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

    # 정규화 통계는 학습과 동일해야 함 — config에서 일괄 관리(S3D Kinetics-400)
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
    svc_rows = svc.pop('rows')
    return {
        'n_videos': total_videos, 'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
        'recall': recall, 'false_alarm': false_alarm, 'accuracy': accuracy,
        'precision': precision, 'f1': f1,
        **{(k if k.startswith(('clean_', 'n_in')) else f'svc_{k}'): v
           for k, v in svc.items()},
        '_svc_rows': svc_rows,
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


def _training_video_names():
    """학습 데이터 폴더(config.DATA_DIR) 아래 영상 파일명 집합.

    평가 영상이 여기 있으면 모델이 학습했거나(train 분할) 학습 중 검증에 썼을(val
    분할) 수 있어 성능이 부풀려진다. 실제로 eval 24개 중 18개가 학습에 들어간 채로
    평가한 적이 있어, 파일명으로 겹침을 잡아 경고하고 깨끗한 영상만 따로 집계한다.
    """
    root = pathlib.Path(str(config.DATA_DIR))
    if not root.exists():
        return set()
    return {p.name for p in root.rglob('*.mp4')}


def _summarize(rows):
    """영상별 결과(rows) → 서비스 관점 지표."""
    A = [r for r in rows if r['gt'] == 'A']
    S = [r for r in rows if r['gt'] == 'S']
    hits = sum(1 for r in A if r['hit'])
    errs = sorted(r['err_f'] for r in A if r['err_f'] is not None)
    errs_s = sorted(r['err_sec'] for r in A if r['err_sec'] is not None)
    hours = sum(r['frames'] / r['fps'] for r in rows if r['fps']) / 3600

    def pct(a, b):
        return round(a / b * 100, 2) if b else None

    false_events = sum(r['false_events'] for r in rows)
    return {
        'n': len(rows), 'n_A': len(A), 'n_S': len(S),
        'recall': pct(hits, len(A)),
        'overlap_recall': pct(sum(1 for r in A if r['overlap']), len(A)),
        'missed': sum(1 for r in A if r['n_events'] == 0),
        'wrong_place': sum(1 for r in A if r['n_events'] and not r['hit']),
        'split': sum(1 for r in A if r['hit'] and r['n_events'] > 1),
        'false_events': false_events,
        'false_per_hour': round(false_events / hours, 1) if hours else None,
        'video_false_alarm': pct(sum(1 for r in S if r['n_events']), len(S)),
        'err_median_f': errs[len(errs) // 2] if errs else None,
        'err_median_sec': round(errs_s[len(errs_s) // 2], 3) if errs_s else None,
        'hours': hours,
    }


def _print_summary(s, tol_sec):
    print(f"영상 수            : {s['n']} 개  (A {s['n_A']} / S {s['n_S']}, "
          f"총 {s['hours']*60:.1f}분)")
    if s['n_A']:
        print(f"Recall(시작 ±{tol_sec:g}초) : {s['recall']:.2f}%   "
              f"← 라벨 시작 근처에서 시작한 이벤트가 있어야 검출")
        print(f"  참고: 구간 겹침만  : {s['overlap_recall']:.2f}%   "
              f"(클립에 충돌 장면이 담기기만 한 경우까지 인정)")
        print(f"  놓침 {s['missed']}건 / 엉뚱한 곳 검출 {s['wrong_place']}건 / "
              f"검출했지만 분할 {s['split']}건")
    print(f"오탐 이벤트        : {s['false_events']}개"
          + (f"  (영상 1시간당 {s['false_per_hour']}개)" if s['false_per_hour'] is not None else "")
          + "   ← 충돌 시작과 무관한 클립 전부")
    if s['n_S']:
        print(f"비충돌 영상 오탐율 : {s['video_false_alarm']:.2f}%  "
              f"(S {s['n_S']}개 중 이벤트가 하나라도 나온 비율)")
    if s['err_median_f'] is not None:
        print(f"시작 시점 오차     : 중앙값 {s['err_median_f']:+d}f "
              f"({s['err_median_sec']:+.2f}초)  ← 가장 가까운 이벤트 기준")


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

    위쪽 평가는 거친 탐색 선별·이벤트 상태머신·병합/길이필터를 건너뛰고 '영상에
    A 윈도우가 하나라도 있는가'만 본다(논문과 같은 영상 단위 지표). 여기서는
    같은 함수를 클립 렌더링만 끄고 호출해, 사용자가 실제로 받는 이벤트를 라벨과
    대조한다.

    판정 기준 (이 프로젝트는 충돌 '시작'이 중요하고 끝은 중요하지 않다):
      · 검출      : 라벨 start_f ± EVAL_HIT_TOLERANCE_SEC 안에서 시작한 이벤트가 있음
      · 엉뚱한 곳 : 이벤트는 나왔지만 시작 근처가 아님 → 검출로 치지 않는다.
                    '이벤트가 하나라도 있으면 검출'로 세면 recall 이 부풀려진다.
      · 오탐 이벤트: 충돌 시작 근처가 아닌 모든 이벤트(비충돌 영상의 것 + 충돌
                    영상의 여분). 사용자가 헛걸음하게 되는 클립 수다.
    """
    import tempfile
    from predict_cam import predict_events_and_clips

    tol_sec = float(getattr(config, 'EVAL_HIT_TOLERANCE_SEC', 1.0))
    in_train = _training_video_names()

    print("\n" + "=" * 60)
    print("[서비스 경로 평가]  거친 탐색 선별 + 상태머신 + 병합/길이필터 포함")
    print("=" * 60)

    rows = []
    tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix='evalclips_'))
    for mp4_file in tqdm(selected_files, desc="서비스 경로"):
        base = mp4_file.rsplit('.', 1)[0]
        video_path = os.path.join(folder_path, mp4_file)
        bboxes, gt_sf, gt_ef = _read_label(os.path.join(folder_path, f"{base}.txt"))
        if not bboxes:
            continue
        bbox = bboxes.get(target_id, next(iter(bboxes.values())))

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()

        try:
            events = predict_events_and_clips(
                model, video_path=video_path, bbox=tuple(bbox),
                output_dir=tmp_dir, r_value=r_value, resize=resize,
                clip_length=clip_length, render_clips=False)
        except Exception as exc:  # noqa: BLE001
            print(f"  [경고] {mp4_file} 처리 실패: {exc}")
            continue

        row = {'file': mp4_file, 'gt': 'A' if gt_sf is not None else 'S',
               'in_train': mp4_file in in_train, 'fps': round(fps, 2),
               'frames': frames, 'gt_start': gt_sf, 'gt_end': gt_ef,
               'n_events': len(events),
               'event_starts': ' '.join(str(e['start_frame']) for e in events),
               'hit': None, 'overlap': None, 'err_f': None, 'err_sec': None,
               'false_events': len(events)}
        if gt_sf is not None:
            tol = max(1, int(round(fps * tol_sec)))
            row['hit'] = False
            row['overlap'] = any(e['event_start_frame'] <= gt_ef
                                 and e['event_end_frame'] >= gt_sf for e in events)
            if events:
                best = min(events, key=lambda e: abs(e['start_frame'] - gt_sf))
                err = best['start_frame'] - gt_sf
                row['err_f'] = err
                row['err_sec'] = round(err / fps, 3)
                row['hit'] = abs(err) <= tol
                row['false_events'] = len(events) - (1 if row['hit'] else 0)
        rows.append(row)

    # ── 영상별 결과 ──
    print(f"\n{'영상':<22} {'학습폴더':>6} {'fps':>4} {'라벨시작':>7} {'이벤트':>5} "
          f"{'가장 가까운 시작(오차)':>24}  판정")
    print("-" * 92)
    for r in rows:
        if r['gt'] == 'A':
            if r['n_events'] == 0:
                verdict = "❌ 놓침"
            elif r['hit']:
                verdict = "검출" + (f" +오탐{r['false_events']}" if r['false_events'] else "")
            else:
                verdict = f"❌ 엉뚱한 곳 ({r['n_events']}개)"
            near = (f"{r['gt_start'] + r['err_f']} ({r['err_f']:+d}f, {r['err_sec']:+.2f}s)"
                    if r['err_f'] is not None else "-")
        else:
            verdict = f"❌ 오탐 {r['n_events']}개" if r['n_events'] else "정상"
            near = "-"
        print(f"{r['file']:<22} {'⚠있음' if r['in_train'] else '-':>6} {r['fps']:>4.0f} "
              f"{r['gt_start'] if r['gt_start'] is not None else '-':>7} {r['n_events']:>5} "
              f"{near:>24}  {verdict}")

    # ── 집계 ──
    print("\n" + "-" * 60)
    all_s = _summarize(rows)
    _print_summary(all_s, tol_sec)

    n_contam = sum(1 for r in rows if r['in_train'])
    clean_s = None
    if n_contam:
        clean = [r for r in rows if not r['in_train']]
        print(f"\n⚠️  평가 영상 {len(rows)}개 중 {n_contam}개가 학습 폴더({config.DATA_DIR})에도 "
              f"있습니다.\n    그 영상들은 학습에 쓰였을 수 있어 위 수치는 부풀려져 있습니다. "
              f"학습 폴더에 없는 영상만 따로 집계:")
        print("-" * 60)
        if clean:
            clean_s = _summarize(clean)
            _print_summary(clean_s, tol_sec)
        else:
            print("  (없음 — 모든 평가 영상이 학습 폴더에 있어 신뢰할 수 있는 수치가 없습니다)")
    print("=" * 60)

    out = {'hit_tol_sec': tol_sec,
           **{k: v for k, v in all_s.items() if k not in ('n', 'n_A', 'n_S', 'hours')},
           'n_in_train': n_contam}
    if clean_s:
        out.update({'clean_n': clean_s['n'], 'clean_recall': clean_s['recall'],
                    'clean_false_events': clean_s['false_events'],
                    'clean_video_false_alarm': clean_s['video_false_alarm']})
    out['rows'] = rows
    return out


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

    # 영상별 결과 CSV — 어떤 영상에서 무엇이 틀렸는지 나중에 다시 보려고
    rows = m.pop('_svc_rows', [])
    if rows:
        per_video = log_dir / f"eval_{config.MODEL_NAME}_{tag}_{stamp}_videos.csv"
        with open(per_video, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[로그] 영상별 결과: {per_video}")

    # 요약 CSV 에 한 줄 덧붙인다(모델 비교용)
    summary = log_dir / 'eval_summary.csv'
    cols = ['datetime', 'model', 'pretrain', 'weights', 'eval_folder', 'n_videos',
            'tp', 'fn', 'fp', 'tn', 'recall', 'false_alarm', 'accuracy',
            'precision', 'f1',
            'svc_hit_tol_sec', 'svc_recall', 'svc_overlap_recall', 'svc_missed',
            'svc_wrong_place', 'svc_split', 'svc_false_events', 'svc_false_per_hour',
            'svc_video_false_alarm', 'svc_err_median_f', 'svc_err_median_sec',
            'n_in_train', 'clean_n', 'clean_recall', 'clean_false_events',
            'clean_video_false_alarm']
    # 컬럼 구성이 바뀐 옛 요약 파일에 이어 쓰면 열이 어긋나므로 옆으로 치워둔다
    if summary.exists():
        with open(summary, encoding='utf-8') as f:
            old_header = f.readline().strip().split(',')
        if old_header != cols:
            moved = summary.with_name(f"eval_summary_old_{stamp}.csv")
            summary.rename(moved)
            print(f"[로그] 요약 CSV 컬럼이 바뀌어 기존 파일을 {moved.name} 로 옮겼습니다")
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
