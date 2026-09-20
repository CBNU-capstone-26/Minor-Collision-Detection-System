import os
from datetime import datetime
import sys
import csv
import pathlib
import cv2
import torch
import numpy as np
from tqdm import tqdm

import config
from device_utils import is_cuda_like, is_channels_last_3d_supported


def _evaluate_folder_impl(
    model,
    folder_path=config.EVAL_FOLDER_PATH,
    target_id=config.TARGET_ID,
    r_value=config.R_VALUE,
    resize=config.RESIZE,
    clip_length=config.CLIP_LENGTH,
    infer_batch_size=config.EVAL_INFER_BATCH_SIZE,
    window_stride=config.EVAL_WINDOW_STRIDE,
):
    folder_path = str(folder_path)
    print(f"[{folder_path}] 폴더의 영상 평가를 준비합니다...")
    device = next(model.parameters()).device
    model.eval()
    window_stride = max(1, int(window_stride))

    if is_cuda_like(device):
        torch.backends.cudnn.benchmark = True

    # 정규화 통계는 학습과 동일해야 함 — config에서 일괄 관리(Kinetics-400)
    mean = torch.tensor(config.NORM_MEAN,
                        dtype=torch.float32).view(3, 1, 1, 1)
    std = torch.tensor(config.NORM_STD,
                       dtype=torch.float32).view(3, 1, 1, 1)

    def crop_square_and_pad(frame, bbox, r):
        h, w, _ = frame.shape
        x_min, y_min, x_max, y_max = bbox
        vw, vh = (x_max - x_min) * r, (y_max - y_min) * r
        cx, cy = x_min + (x_max - x_min) // 2, y_min + (y_max - y_min) // 2
        side = int(max(vw, vh))
        nx1, ny1 = cx - side // 2, cy - side // 2
        nx2, ny2 = cx + side // 2, cy + side // 2
        v_x1, v_y1 = max(0, nx1), max(0, ny1)
        v_x2, v_y2 = min(w, nx2), min(h, ny2)
        p_l, p_t = max(0, -nx1), max(0, -ny1)
        p_r, p_b = max(0, nx2 - w), max(0, ny2 - h)
        cropped = frame[v_y1:v_y2, v_x1:v_x2]
        if p_l > 0 or p_t > 0 or p_r > 0 or p_b > 0:
            cropped = np.pad(
                cropped, ((p_t, p_b), (p_l, p_r), (0, 0)), mode='constant')
        return cv2.resize(cropped, resize, interpolation=cv2.INTER_LINEAR)

    def frames_to_tensor(frames):
        arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
        video_tensor = torch.from_numpy(arr).permute(3, 0, 1, 2).contiguous()
        return (video_tensor - mean) / std

    all_files = os.listdir(folder_path)
    all_files_set = set(all_files)
    mp4_files = [f for f in all_files if f.endswith('.mp4')]
    valid_pairs = [
        mp4 for mp4 in mp4_files if f"{mp4.rsplit('.', 1)[0]}.txt" in all_files_set]

    if not valid_pairs:
        print("평가할 수 있는 영상-텍스트 짝이 없습니다.")
        return

    # data/eval의 유효한 영상-txt 짝을 모두 평가 (샘플 개수 제한 없음, 정렬로 순서 고정)
    selected_files = sorted(valid_pairs)
    print(f"data/eval 전체 {len(selected_files)}개 영상 쌍을 모두 평가합니다.")

    total_videos = 0
    correct_preds = 0
    tp = tn = fp = fn = 0   # 혼동행렬 (Positive = A/충돌)
    wrong_list = []

    with torch.inference_mode():
        for mp4_file in tqdm(selected_files, desc="평가 진행률"):
            base_name = mp4_file.rsplit('.', 1)[0]
            video_path = os.path.join(folder_path, mp4_file)
            txt_path = os.path.join(folder_path, f"{base_name}.txt")

            parts = base_name.split('_')
            # 클래스 문자는 parts[1]의 '마지막 글자'로 판별한다.
            #   rc: LA/RA/SA/NA(2글자, 마지막=A) → 충돌, LS/LV/LW 등 → 비충돌
            #   real: ReA(3글자, 마지막=A) → 충돌, ReS → 비충돌
            # (TODO: 추후 파일명 대신 txt의 A/S action줄로 읽도록 통일 예정)
            is_accident_gt = len(parts) >= 2 and len(
                parts[1]) >= 2 and parts[1][-1] == 'A'
            gt_label = 1 if is_accident_gt else 0

            bboxes = {}
            with open(txt_path, 'r') as f:
                for line in f:
                    l_parts = line.strip().split(',')
                    if len(l_parts) >= 6 and l_parts[0] == 'car':
                        bboxes[int(l_parts[1])] = [int(l_parts[2]), int(
                            l_parts[3]), int(l_parts[4]), int(l_parts[5])]

            if target_id not in bboxes:
                if not bboxes:
                    continue
                target_bbox = next(iter(bboxes.values()))
            else:
                target_bbox = bboxes[target_id]

            cap = cv2.VideoCapture(video_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(crop_square_and_pad(
                    frame_rgb, target_bbox, r_value))
            cap.release()

            while len(frames) < clip_length:
                frames.append(
                    frames[-1] if frames else np.zeros((resize[1], resize[0], 3), dtype=np.uint8))

            # ⑦ 영상 텐서를 처음부터 GPU에 상주 — 배치마다 .to(device) 전송 제거
            full_video_tensor = frames_to_tensor(frames).to(device)
            predicted_label = 0
            num_windows = len(frames) - (clip_length - 1)
            window_starts = list(range(0, num_windows, window_stride))

            for batch_start in range(0, len(window_starts), infer_batch_size):
                batch_window_starts = window_starts[batch_start:batch_start +
                                                    infer_batch_size]
                # 이미 GPU에 있는 텐서 슬라이싱 — .to() 호출 없음
                clips = torch.stack(
                    [full_video_tensor[:, i:i+clip_length, :, :] for i in batch_window_starts])
                if is_channels_last_3d_supported(device):
                    clips = clips.contiguous(
                        memory_format=torch.channels_last_3d)

                outputs = model(clips)
                pred_classes = outputs.argmax(dim=1)
                if (pred_classes == 1).any().item():
                    predicted_label = 1
                    break

            total_videos += 1
            # 혼동행렬 집계 (Positive = A/충돌)
            if gt_label == 1 and predicted_label == 1:
                tp += 1
            elif gt_label == 0 and predicted_label == 0:
                tn += 1
            elif gt_label == 0 and predicted_label == 1:
                fp += 1
            else:  # gt_label == 1 and predicted_label == 0
                fn += 1

            if predicted_label == gt_label:
                correct_preds += 1
            else:
                wrong_list.append({
                    "file": mp4_file,
                    "gt": "Accident(충돌)" if gt_label == 1 else "Normal(정상)",
                    "pred": "Accident(충돌)" if predicted_label == 1 else "Normal(정상)",
                })

    # ── 논문(Hwang & Lee 2024, Table 6)과 동일한 지표 ──────────────
    #   Positive = A(충돌).  Recall = TP/(TP+FN), False alarm = FP/(FP+TN),
    #   Accuracy = (TP+TN)/전체, Precision = TP/(TP+FP), F1 = 2PR/(P+R)
    def _pct(num, den):
        return (num / den * 100) if den > 0 else 0.0

    accuracy = _pct(tp + tn, total_videos)
    recall = _pct(tp, tp + fn)          # 실제 충돌 중 잡은 비율
    false_alarm = _pct(fp, fp + tn)     # 정상 중 오탐 비율
    precision = _pct(tp, tp + fp)       # A라 판정한 것 중 실제 A
    f1 = (2 * precision * recall / (precision + recall)
          ) if (precision + recall) > 0 else 0.0

    print("\n" + "=" * 50)
    print("[모델 성능 평가 결과]  (Positive = A/충돌)")
    print("=" * 50)
    print(f"총 평가 영상 수 : {total_videos} 개  (A {tp + fn} / S {fp + tn})")
    print(f"혼동행렬        : TP {tp} / FN {fn} / FP {fp} / TN {tn}")
    print("-" * 50)
    print(f"Recall(재현율)      : {recall:.2f}%   (실제 충돌 중 잡음)")
    print(f"False alarm(오탐율) : {false_alarm:.2f}%   (정상 중 오탐)")
    print(f"Accuracy(정확도)    : {accuracy:.2f}%")
    print(f"Precision(정밀도)   : {precision:.2f}%")
    print(f"F1-score            : {f1:.2f}%")
    print("=" * 50)

    if wrong_list:
        print("\n[오답 노트 (틀린 영상 리스트)]")
        for w in wrong_list:
            print(f" - {w['file']} (실제: {w['gt']}  |  모델예측: {w['pred']})")
    else:
        print("\n모든 영상을 완벽하게 맞췄습니다!")

    # 이어서 '서비스와 동일한 경로'로 한 번 더 평가한다
    svc = evaluate_service_path(model, folder_path, selected_files,
                                target_id=target_id, r_value=r_value,
                                resize=resize, clip_length=clip_length)
    return {
        'n_videos': total_videos, 'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
        'recall': recall, 'false_alarm': false_alarm, 'accuracy': accuracy,
        'precision': precision, 'f1': f1, **{f'svc_{k}': v for k, v in svc.items()},
    }


def _read_label(txt_path):
    """라벨 txt → (bboxes, gt_start_f, gt_end_f). A 라인이 없으면 (.., None, None)."""
    bboxes, sf, ef = {}, None, None
    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            p = [t.strip() for t in line.strip().split(',')]
            if len(p) >= 6 and p[0] == 'car':
                try:
                    bboxes[int(p[1])] = [int(float(v)) for v in p[2:6]]
                except ValueError:
                    pass
            elif p and p[0] == 'A' and len(p) > 2:
                try:
                    sf = int(float(p[2]))
                    ef = int(float(p[3])) if len(p) > 3 else sf
                except ValueError:
                    pass
    return bboxes, sf, ef


def evaluate_service_path(
    model,
    folder_path,
    selected_files,
    target_id=config.TARGET_ID,
    r_value=config.R_VALUE,
    resize=config.RESIZE,
    clip_length=config.CLIP_LENGTH,
):
    """서비스가 실제로 쓰는 predict_events_and_clips 로 평가한다.

    위쪽 evaluate_folder_accuracy 는 flow 사전선별·이벤트 상태머신·병합/길이필터를
    모두 건너뛰고 '영상에 A 윈도우가 하나라도 있는가'만 본다. 그래서 서비스가
    내놓는 결과(이벤트 몇 개·언제)와 연결되지 않는다. 여기서는 같은 함수를
    클립 렌더링만 끄고 호출해, 사용자가 실제로 받는 이벤트 목록을 라벨과 대조한다.

    추가로 재는 것:
      · 시점 오차   : 검출 이벤트 시작 − 라벨 start_f (서비스의 핵심 출력)
      · 이벤트 개수 : 한 장면이 여러 클립으로 쪼개지는지 / 오탐이 몇 개인지
      · 구간 IoU    : 검출 구간이 라벨 구간과 얼마나 겹치는지
    """
    import tempfile
    from predict_cam import predict_events_and_clips

    print("\n" + "=" * 50)
    print("[서비스 경로 평가]  prescreen + 상태머신 + 병합/길이필터 포함")
    print("=" * 50)

    tp = tn = fp = fn = 0
    time_errs, ious, n_events, split_cases, miss_list = [], [], [], [], []
    tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix='evalclips_'))

    for mp4_file in tqdm(selected_files, desc="서비스 경로"):
        base = mp4_file.rsplit('.', 1)[0]
        video_path = os.path.join(folder_path, mp4_file)
        bboxes, gt_sf, gt_ef = _read_label(os.path.join(folder_path, f"{base}.txt"))
        if not bboxes:
            continue
        bbox = bboxes.get(target_id, next(iter(bboxes.values())))
        gt_label = 1 if gt_sf is not None else 0

        try:
            events = predict_events_and_clips(
                model, video_path=video_path, bbox=tuple(bbox),
                output_dir=tmp_dir, r_value=r_value, resize=resize,
                clip_length=clip_length, render_clips=False)
        except Exception as exc:  # noqa: BLE001
            print(f"  [경고] {mp4_file} 처리 실패: {exc}")
            continue

        pred_label = 1 if events else 0
        n_events.append(len(events))
        if gt_label == 1 and pred_label == 1:
            tp += 1
        elif gt_label == 0 and pred_label == 0:
            tn += 1
        elif gt_label == 0 and pred_label == 1:
            fp += 1
        else:
            fn += 1
            miss_list.append(mp4_file)

        if gt_label == 1 and events:
            # 라벨 구간과 가장 많이 겹치는 이벤트를 '정답 대응'으로 본다
            def _ov(e):
                return (min(e['end_frame'], gt_ef) - max(e['start_frame'], gt_sf))
            best = max(events, key=_ov)
            time_errs.append(best['start_frame'] - gt_sf)
            inter = max(0, _ov(best) + 1)
            union = (max(best['end_frame'], gt_ef) - min(best['start_frame'], gt_sf) + 1)
            ious.append(inter / union if union > 0 else 0.0)
            if len(events) > 1:
                split_cases.append((mp4_file, len(events)))

    def _pct(a, b):
        return (a / b * 100) if b > 0 else 0.0

    total = tp + tn + fp + fn
    print(f"\n총 평가 영상 수 : {total} 개  (A {tp + fn} / S {fp + tn})")
    print(f"혼동행렬        : TP {tp} / FN {fn} / FP {fp} / TN {tn}")
    print("-" * 50)
    print(f"Recall(재현율)      : {_pct(tp, tp + fn):.2f}%")
    print(f"False alarm(오탐율) : {_pct(fp, fp + tn):.2f}%")
    print(f"Accuracy(정확도)    : {_pct(tp + tn, total):.2f}%")
    print("-" * 50)
    if time_errs:
        s = sorted(time_errs)
        med = s[len(s) // 2]
        within = sum(1 for e in s if abs(e) <= clip_length)
        print(f"시점 오차(검출−라벨): 중앙값 {med:+d}f / 평균 {sum(s)/len(s):+.1f}f "
              f"/ 범위 {s[0]:+d}~{s[-1]:+d}f")
        print(f"                     ±{clip_length}f 이내 {within}/{len(s)}건 "
              f"({_pct(within, len(s)):.0f}%)")
    else:
        print("시점 오차           : 측정 불가 (검출된 충돌 없음)")
    if ious:
        print(f"구간 IoU            : 평균 {sum(ious)/len(ious):.3f} / "
              f"최소 {min(ious):.3f}")
    if n_events:
        print(f"영상당 이벤트 수    : 평균 {sum(n_events)/len(n_events):.2f} / "
              f"최대 {max(n_events)}  (= 사용자가 받는 클립 수)")
    if split_cases:
        print(f"\n[한 충돌이 여러 이벤트로 분할된 영상] {len(split_cases)}건")
        for f, n in split_cases:
            print(f" - {f}: {n}개")
    if miss_list:
        print(f"\n[충돌을 놓친 영상] {len(miss_list)}건")
        for f in miss_list:
            print(f" - {f}")
    print("=" * 50)

    _s = sorted(time_errs)
    return {
        'recall': _pct(tp, tp + fn),
        'false_alarm': _pct(fp, fp + tn),
        'accuracy': _pct(tp + tn, total),
        'time_err_median': (_s[len(_s) // 2] if _s else None),
        'time_err_mean': (round(sum(_s) / len(_s), 2) if _s else None),
        'iou_mean': (round(sum(ious) / len(ious), 4) if ious else None),
        'events_per_video': (round(sum(n_events) / len(n_events), 2)
                             if n_events else None),
        'split_videos': len(split_cases),
        'missed_videos': len(miss_list),
    }


class _Tee:
    """콘솔에 그대로 찍으면서 파일에도 남긴다(평가 결과를 나중에 다시 보려고)."""

    def __init__(self, path):
        self.file = open(path, 'w', encoding='utf-8')
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def evaluate_folder_accuracy(model, folder_path=config.EVAL_FOLDER_PATH, **kwargs):
    """폴더 평가 진입점 — 결과를 콘솔과 파일에 동시에 남긴다.

    남는 것 두 가지:
      1) 콘솔 전문 로그  outputs/evallogs/eval_<모델>_<ptY|ptN>_<시각>.txt
      2) 요약 한 줄      outputs/evallogs/eval_summary.csv  (append)
         → 5가지 학습 조합을 한 표에서 비교하려고 계속 덧붙인다.
    """
    log_dir = pathlib.Path(getattr(config, 'EVAL_LOG_DIR',
                                   pathlib.Path('outputs') / 'evallogs'))
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%y%m%d_%H%M')
    tag = getattr(config, 'PRETRAIN_TAG', '?')
    weights = pathlib.Path(str(config.EVAL_WEIGHTS_PATH)).name
    log_path = log_dir / f"eval_{config.MODEL_NAME}_{tag}_{stamp}.txt"

    tee = _Tee(log_path)
    sys.stdout = tee
    try:
        print(f"[평가 대상] 모델={config.MODEL_NAME} / 사전학습={tag} / 가중치={weights}")
        print(f"[평가 폴더] {folder_path}")
        m = _evaluate_folder_impl(model, folder_path=folder_path, **kwargs)
    finally:
        sys.stdout = tee.stdout
        tee.close()

    if not m:
        print(f"[로그] 평가 로그: {log_path}")
        return m

    # 요약 CSV 에 한 줄 덧붙인다(모델 비교용)
    summary = log_dir / 'eval_summary.csv'
    cols = ['datetime', 'model', 'pretrain', 'weights', 'eval_folder', 'n_videos',
            'tp', 'fn', 'fp', 'tn', 'recall', 'false_alarm', 'accuracy',
            'precision', 'f1',
            'svc_recall', 'svc_false_alarm', 'svc_accuracy',
            'svc_time_err_median', 'svc_time_err_mean', 'svc_iou_mean',
            'svc_events_per_video', 'svc_split_videos', 'svc_missed_videos']
    row = {'datetime': datetime.now().strftime('%Y-%m-%d %H:%M'),
           'model': config.MODEL_NAME, 'pretrain': tag, 'weights': weights,
           'eval_folder': str(folder_path), **m}
    write_header = not summary.exists()
    with open(summary, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(cols)
        w.writerow(['' if row.get(c) is None else row.get(c, '') for c in cols])

    print(f"[로그] 평가 전문: {log_path}")
    print(f"[로그] 요약 누적: {summary}")
    return m
