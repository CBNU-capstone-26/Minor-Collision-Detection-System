import os
from datetime import datetime
import sys
import csv
import pathlib
import platform
import statistics
import time
import cv2
import torch
import numpy as np
from tqdm import tqdm

import config
from device_utils import is_cuda_like, is_channels_last_3d_supported
from runlog import make_run_dir, write_run_info


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
    base_rows = {}          # 영상별 ① 판정 — 서비스 판정과 나란히 영상별 CSV에 남긴다

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
            first_a = None          # P(A) > 0.5 가 처음 나온 창의 시작 프레임
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
                    first_a = batch_window_starts[int((pred_classes == 1).nonzero()[0])]
                    break

            total_videos += 1
            base_rows[mp4_file] = {'pred': 'A' if predicted_label else 'S', 'first_a': first_a}
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
    base = _metrics(tp, fn, fp, tn)
    print("\n" + "=" * 60)
    print("[① 기존 판정]  창 하나라도 P(A) > 0.5 면 그 영상은 충돌  (논문 Table 6 방식)")
    print(f"   창 간격 {window_stride}로 모든 창을 본다. 거친 탐색·이벤트 규칙이 없고, "
          "충돌 위치는 따지지 않는다.")
    print("=" * 60)
    _print_metrics(base)
    print("=" * 60)

    if wrong_list:
        print("\n[오답 노트 (틀린 영상 리스트)]")
        for w in wrong_list:
            print(f" - {w['file']} (실제: {w['gt']}  |  모델예측: {w['pred']})")
    else:
        print("\n모든 영상을 완벽하게 맞췄습니다!")

    # 이어서 '서비스와 동일한 경로'로 한 번 더 평가한다
    svc = evaluate_service_path(model, folder_path, selected_files,
                                target_id=target_id, r_value=r_value,
                                resize=resize, clip_length=clip_length,
                                base_rows=base_rows)
    svc_rows = svc.pop('rows')
    _print_comparison(base, svc, svc_rows, window_stride)

    # 추론 속도: ① 모델만 ② 서비스 경로 전체(위에서 영상별로 잰 시간)
    speed = {'device': _device_name(device), 'model': benchmark_model(
        model, clip_length=clip_length, resize=resize), 'service': svc.pop('speed')}
    _print_speed(speed)
    sb = {b['batch']: b for b in speed['model']}
    svc_b = sb.get(int(config.PREDICT_INFER_BATCH_SIZE), speed['model'][-1])
    s = speed['service'] or {}
    return {
        'n_videos': total_videos,
        **{f'base_{k}': v for k, v in base.items()},
        **{(k if k.startswith(('clean_', 'n_in')) else f'svc_{k}'): v
           for k, v in svc.items()},
        'device': speed['device'],
        'speed_proc_fps': round(s['proc_fps'], 2) if s else None,
        'speed_realtime_x': round(s['realtime_x'], 3) if s else None,
        'speed_windows_eval_pct': round(s['windows_eval_pct'], 1) if s else None,
        'speed_model_batch': svc_b['batch'],
        'speed_model_clips_per_sec': svc_b['clips_per_sec'],
        'speed_model_ms_per_clip': svc_b['ms_per_clip'],
        '_speed': speed,
        '_svc_rows': svc_rows,
    }


def _metrics(tp, fn, fp, tn):
    """영상 단위 혼동행렬 → 논문 Table 6 지표(%). 분모가 0이면 None.

    Positive = A(충돌). Recall = TP/(TP+FN), False alarm = FP/(FP+TN),
    Accuracy = (TP+TN)/전체, Precision = TP/(TP+FP), F1 = 2PR/(P+R).
    """
    def pct(a, b):
        return round(a / b * 100, 2) if b else None

    rec, prec = pct(tp, tp + fn), pct(tp, tp + fp)
    if rec is None or prec is None:
        f1 = None
    else:
        f1 = round(2 * prec * rec / (prec + rec), 2) if (prec + rec) else 0.0
    return {'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
            'recall': rec, 'false_alarm': pct(fp, fp + tn),
            'accuracy': pct(tp + tn, tp + fn + fp + tn), 'precision': prec, 'f1': f1}


def _fmt_pct(v):
    return '-' if v is None else f'{v:.2f}%'


def _print_metrics(mt):
    print(f"영상 수            : {mt['tp'] + mt['fn'] + mt['fp'] + mt['tn']} 개  "
          f"(A {mt['tp'] + mt['fn']} / S {mt['fp'] + mt['tn']})")
    print(f"혼동행렬           : TP {mt['tp']} / FN {mt['fn']} / FP {mt['fp']} / TN {mt['tn']}")
    print(f"Recall(재현율)      : {_fmt_pct(mt['recall'])}   (실제 충돌 영상 중 충돌로 판정)")
    print(f"False alarm(오탐율) : {_fmt_pct(mt['false_alarm'])}   (비충돌 영상 중 충돌로 판정)")
    print(f"Accuracy(정확도)    : {_fmt_pct(mt['accuracy'])}")
    print(f"Precision(정밀도)   : {_fmt_pct(mt['precision'])}")
    print(f"F1-score            : {_fmt_pct(mt['f1'])}")


def _print_comparison(base, svc, rows, window_stride):
    """① 기존 판정과 ② 서비스 판정을 같은 지표로 나란히 — 판정 규칙 차이만 비교된다."""
    tol = svc.get('hit_tol_sec', 1.0)
    W = 22

    def line(name, a, b):
        print(f"{name:<{W}}{a:>18}{b:>22}")

    print("\n" + "=" * 64)
    print("[판정 방식 비교]  같은 영상, 영상 단위 (Positive = A/충돌)")
    print("=" * 64)
    line('', '① 기존 판정', '② 서비스 판정')
    line('판정 규칙', '창 P(A)>0.5 1개↑', '이벤트 1개↑')
    line('창 보는 방식', f'간격 {window_stride}, 전부', f'거친 탐색 → 의심 구간만')
    print("-" * 64)
    line('TP / FN / FP / TN', f"{base['tp']}/{base['fn']}/{base['fp']}/{base['tn']}",
         f"{svc['tp']}/{svc['fn']}/{svc['fp']}/{svc['tn']}")
    line('Recall(재현율)', _fmt_pct(base['recall']), _fmt_pct(svc['video_recall']))
    line('False alarm(오탐율)', _fmt_pct(base['false_alarm']), _fmt_pct(svc['video_false_alarm']))
    line('Accuracy(정확도)', _fmt_pct(base['accuracy']), _fmt_pct(svc['accuracy']))
    line('Precision(정밀도)', _fmt_pct(base['precision']), _fmt_pct(svc['precision']))
    line('F1-score', _fmt_pct(base['f1']), _fmt_pct(svc['f1']))
    print("-" * 64)
    line(f'Recall(시작 ±{tol:g}초)', '(위치 안 봄)', _fmt_pct(svc['hit_recall']))
    fph = svc.get('false_per_hour')
    line('오탐 이벤트', '(이벤트 없음)',
         f"{svc['false_events']}개" + (f" (1시간당 {fph})" if fph is not None else ''))
    print("=" * 64)
    print("  · 위 표의 윗부분은 두 판정 모두 '충돌 위치와 무관하게 영상에서 충돌이 나왔나'만 본다.")
    print(f"  · 서비스에서 쓸모 있는 검출은 'Recall(시작 ±{tol:g}초)'이다 — 충돌 시작 근처에서")
    print("    시작한 이벤트만 센다(엉뚱한 곳의 이벤트는 사용자가 헛걸음하는 오탐).")
    diff = [r for r in rows if r.get('base_pred') and r['base_pred'] != r['svc_pred']]
    if diff:
        print(f"\n[두 판정이 다른 영상 {len(diff)}개]")
        for r in diff:
            fa = f" (첫 A 창 {r['base_first_a']}f)" if r.get('base_first_a') is not None else ''
            print(f" - {r['file']} (실제 {r['gt']}) : ① {r['base_pred']}{fa} / "
                  f"② {r['svc_pred']} (이벤트 {r['n_events']}개)")
    else:
        print("\n두 판정이 다른 영상은 없습니다.")


def _device_name(device):
    """속도 결과를 읽을 때 어떤 하드웨어였는지 알 수 있게 장치 이름을 남긴다."""
    if is_cuda_like(device):
        try:
            return f"{device} ({torch.cuda.get_device_name(device)})"
        except Exception:  # noqa: BLE001
            return str(device)
    cpu = platform.processor() or platform.machine()
    try:
        with open('/proc/cpuinfo', encoding='utf-8') as f:
            cpu = next(l.split(':', 1)[1].strip() for l in f if l.startswith('model name'))
    except (OSError, StopIteration):
        pass
    return f"cpu ({cpu}, torch 스레드 {torch.get_num_threads()}개)"


def benchmark_model(model, clip_length=config.CLIP_LENGTH, resize=config.RESIZE,
                    batch_sizes=None, min_sec=3.0, max_iters=30):
    """3D-CNN 순수 추론 속도 — 영상 디코딩·크롭·전처리 없이 모델만 잰다.

    서비스는 창 간격 1(PREDICT_WINDOW_STRIDE)이라 영상을 빠짐없이 훑으려면 프레임
    하나당 창(30프레임 클립) 하나를 추론해야 한다. 그래서 '초당 처리한 창 수'가 곧
    '모델이 초당 분석할 수 있는 영상 프레임 수'다(거친 탐색 없이 전부 볼 때).
    """
    device = next(model.parameters()).device
    batch_sizes = batch_sizes or sorted({1, int(config.PREDICT_INFER_BATCH_SIZE),
                                         int(config.EVAL_INFER_BATCH_SIZE)})
    sync = (torch.cuda.synchronize if is_cuda_like(device) else (lambda: None))
    out = []
    model.eval()
    with torch.inference_mode():
        for bs in batch_sizes:
            x = torch.randn(bs, 3, clip_length, resize[1], resize[0], device=device)
            if is_channels_last_3d_supported(device):
                x = x.contiguous(memory_format=torch.channels_last_3d)
            for _ in range(2):                       # 첫 호출 오버헤드(cuDNN 탐색 등) 제외
                model(x)
            sync()
            n, t0 = 0, time.perf_counter()
            while n < max_iters and (time.perf_counter() - t0) < min_sec:
                model(x)
                n += 1
            sync()
            sec = time.perf_counter() - t0
            clips_per_sec = n * bs / sec
            out.append({'batch': bs, 'clips_per_sec': round(clips_per_sec, 2),
                        'ms_per_clip': round(1000 / clips_per_sec, 2)})
    return out


def _print_speed(speed):
    print("\n" + "=" * 60)
    print("[추론 속도]  장치: " + speed['device'])
    print("=" * 60)
    print("① 모델만 (3D-CNN, 30프레임 클립 1개 = 창 1개)")
    for b in speed['model']:
        mark = '  ← 서비스 배치' if b['batch'] == config.PREDICT_INFER_BATCH_SIZE else ''
        print(f"   배치 {b['batch']:>2}: 초당 {b['clips_per_sec']:7.2f}개 창 "
              f"(창 1개 {b['ms_per_clip']:.1f}ms){mark}")
    print("   → 창 간격 1로 빠짐없이 훑을 때 모델이 초당 분석하는 영상 프레임 수와 같다")
    s = speed.get('service')
    if s:
        print("② 서비스 경로 전체 (디코딩·크롭 + 움직임 계산 + 거친 탐색·정밀 추론, 클립 렌더링 제외)")
        print(f"   영상 {s['n_videos']}개, {s['video_sec']:.1f}초 분량({s['frames']}프레임) → "
              f"처리 {s['proc_sec']:.1f}초")
        print(f"   처리 속도  : 초당 {s['proc_fps']:.1f} 프레임  "
              f"(영상 1초를 {s['sec_per_video_sec']:.2f}초에 처리, 실시간의 {s['realtime_x']:.2f}배)")
        print(f"   3D-CNN 추론 창: 전체 {s['windows_total']}개 중 {s['windows_eval']}개 "
              f"({s['windows_eval_pct']:.0f}%, 거친 탐색으로 줄임), 초당 {s['windows_per_sec']:.1f}개")
        print(f"   시간 비중  : 디코딩·크롭 {s['pct_decode']:.0f}% / 움직임 계산 {s['pct_flow']:.0f}% / "
              f"3D-CNN {s['pct_infer']:.0f}% / 기타 {s['pct_other']:.0f}%")
        print(f"   영상별 처리 속도: 중앙값 초당 {s['proc_fps_median']:.1f} 프레임 "
              f"(최저 {s['proc_fps_min']:.1f} · 최고 {s['proc_fps_max']:.1f})")
    print("=" * 60)


def _speed_summary(rows):
    """서비스 경로 영상별 처리 시간 → 전체 처리 속도."""
    rs = [r for r in rows if r.get('proc_sec')]
    if not rs:
        return None
    frames = sum(r['frames_read'] for r in rs)
    vsec = sum(r['frames_read'] / r['fps'] for r in rs if r['fps'])
    proc = sum(r['proc_sec'] for r in rs)
    dec = sum(r['decode_sec'] for r in rs)
    flow = sum(r['flow_sec'] for r in rs)
    inf = sum(r['infer_sec'] for r in rs)
    wt = sum(r['windows_total'] for r in rs)
    we = sum(r['windows_eval'] for r in rs)
    per = sorted(r['proc_fps'] for r in rs)
    pct = lambda x: x / proc * 100 if proc else 0.0  # noqa: E731
    return {
        'n_videos': len(rs), 'frames': frames, 'video_sec': vsec, 'proc_sec': proc,
        'proc_fps': frames / proc if proc else 0.0,
        'realtime_x': vsec / proc if proc else 0.0,
        'sec_per_video_sec': proc / vsec if vsec else 0.0,
        'windows_total': wt, 'windows_eval': we,
        'windows_eval_pct': we / wt * 100 if wt else 0.0,
        'windows_per_sec': we / inf if inf else 0.0,
        'pct_decode': pct(dec), 'pct_flow': pct(flow), 'pct_infer': pct(inf),
        'pct_other': max(0.0, 100 - pct(dec) - pct(flow) - pct(inf)),
        'proc_fps_median': statistics.median(per), 'proc_fps_min': per[0], 'proc_fps_max': per[-1],
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
    """영상별 결과(rows) → 서비스 판정 지표.

    · 영상 단위(① 기존 판정과 같은 정의): 이벤트가 하나라도 나오면 그 영상은 충돌.
    · 위치 기준: 라벨 시작 ± 허용 오차 안에서 시작한 이벤트가 있어야 검출(hit_recall).
    """
    A = [r for r in rows if r['gt'] == 'A']
    S = [r for r in rows if r['gt'] == 'S']
    hits = sum(1 for r in A if r['hit'])
    errs = sorted(r['err_f'] for r in A if r['err_f'] is not None)
    errs_s = sorted(r['err_sec'] for r in A if r['err_sec'] is not None)
    hours = sum(r['frames'] / r['fps'] for r in rows if r['fps']) / 3600

    def pct(a, b):
        return round(a / b * 100, 2) if b else None

    tp = sum(1 for r in A if r['n_events'])
    fp = sum(1 for r in S if r['n_events'])
    conf = _metrics(tp, len(A) - tp, fp, len(S) - fp)
    false_events = sum(r['false_events'] for r in rows)
    return {
        'n': len(rows), 'n_A': len(A), 'n_S': len(S),
        'tp': conf['tp'], 'fn': conf['fn'], 'fp': conf['fp'], 'tn': conf['tn'],
        'video_recall': conf['recall'], 'video_false_alarm': conf['false_alarm'],
        'accuracy': conf['accuracy'], 'precision': conf['precision'], 'f1': conf['f1'],
        'hit_recall': pct(hits, len(A)),
        'overlap_recall': pct(sum(1 for r in A if r['overlap']), len(A)),
        'missed': sum(1 for r in A if r['n_events'] == 0),
        'wrong_place': sum(1 for r in A if r['n_events'] and not r['hit']),
        'split': sum(1 for r in A if r['hit'] and r['n_events'] > 1),
        'false_events': false_events,
        'false_per_hour': round(false_events / hours, 1) if hours else None,
        'err_median_f': errs[len(errs) // 2] if errs else None,
        'err_median_sec': round(errs_s[len(errs_s) // 2], 3) if errs_s else None,
        'hours': hours,
    }


def _print_summary(s, tol_sec):
    print(f"영상 수            : {s['n']} 개  (A {s['n_A']} / S {s['n_S']}, "
          f"총 {s['hours']*60:.1f}분)")
    print("· 영상 단위 (이벤트가 하나라도 나오면 충돌 — ① 기존 판정과 같은 정의)")
    print(f"  혼동행렬         : TP {s['tp']} / FN {s['fn']} / FP {s['fp']} / TN {s['tn']}")
    print(f"  Recall {_fmt_pct(s['video_recall'])} / 오탐율 {_fmt_pct(s['video_false_alarm'])} / "
          f"정확도 {_fmt_pct(s['accuracy'])} / 정밀도 {_fmt_pct(s['precision'])} / "
          f"F1 {_fmt_pct(s['f1'])}")
    print("· 위치 기준 (서비스에서 쓸모 있는 검출)")
    if s['n_A']:
        print(f"  Recall(시작 ±{tol_sec:g}초) : {s['hit_recall']:.2f}%   "
              f"← 라벨 시작 근처에서 시작한 이벤트가 있어야 검출")
        print(f"  참고: 구간 겹침만  : {s['overlap_recall']:.2f}%   "
              f"(클립에 충돌 장면이 담기기만 한 경우까지 인정)")
        print(f"  놓침 {s['missed']}건 / 엉뚱한 곳 검출 {s['wrong_place']}건 / "
              f"검출했지만 분할 {s['split']}건")
    print(f"  오탐 이벤트      : {s['false_events']}개"
          + (f"  (영상 1시간당 {s['false_per_hour']}개)" if s['false_per_hour'] is not None else "")
          + "   ← 충돌 시작과 무관한 클립 전부")
    if s['err_median_f'] is not None:
        print(f"  시작 시점 오차   : 중앙값 {s['err_median_f']:+d}f "
              f"({s['err_median_sec']:+.2f}초)  ← 가장 가까운 이벤트 기준")


def evaluate_service_path(
    model,
    folder_path,
    selected_files,
    target_id=config.TARGET_ID,
    r_value=config.R_VALUE,
    resize=config.RESIZE,
    clip_length=config.CLIP_LENGTH,
    base_rows=None,
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
    from inference import predict_events_and_clips

    tol_sec = float(getattr(config, 'EVAL_HIT_TOLERANCE_SEC', 1.0))
    in_train = _training_video_names()

    print("\n" + "=" * 60)
    print("[② 서비스 판정]  서비스와 같은 함수: 거친 탐색 + 상태머신 + 병합/길이필터")
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

        stats = {}
        try:
            t0 = time.perf_counter()
            events = predict_events_and_clips(
                model, video_path=video_path, bbox=tuple(bbox),
                output_dir=tmp_dir, r_value=r_value, resize=resize,
                clip_length=clip_length, render_clips=False, stats=stats)
            proc_sec = time.perf_counter() - t0
        except Exception as exc:  # noqa: BLE001
            print(f"  [경고] {mp4_file} 처리 실패: {exc}")
            continue

        frames_read = stats.get('frames', frames)
        base_r = (base_rows or {}).get(mp4_file, {})
        row = {'file': mp4_file, 'gt': 'A' if gt_sf is not None else 'S',
               # 두 판정을 영상별로 나란히 — ① 기존(창 P(A)>0.5 1개↑) / ② 서비스(이벤트 1개↑)
               'base_pred': base_r.get('pred'), 'base_first_a': base_r.get('first_a'),
               'svc_pred': 'A' if events else 'S',
               'in_train': mp4_file in in_train, 'fps': round(fps, 2),
               'frames': frames, 'gt_start': gt_sf, 'gt_end': gt_ef,
               'n_events': len(events),
               'event_starts': ' '.join(str(e['start_frame']) for e in events),
               'hit': None, 'overlap': None, 'err_f': None, 'err_sec': None,
               'false_events': len(events),
               # ── 속도 (클립 렌더링 제외: render_clips=False) ──
               'frames_read': frames_read,
               'proc_sec': round(proc_sec, 3),
               'proc_fps': round(frames_read / proc_sec, 2) if proc_sec else None,
               'realtime_x': round(frames_read / fps / proc_sec, 3) if proc_sec and fps else None,
               'decode_sec': round(stats.get('decode_sec', 0.0), 3),
               'flow_sec': round(stats.get('flow_sec', 0.0), 3),
               'infer_sec': round(stats.get('infer_sec', 0.0), 3),
               'windows_total': stats.get('windows_total', 0),
               'windows_eval': stats.get('windows_eval', 0)}
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
          f"{'가장 가까운 시작(오차)':>24} {'처리fps':>7}  판정")
    print("-" * 100)
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
              f"{near:>24} {r['proc_fps'] or 0:>7.1f}  {verdict}")

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
           'n_in_train': n_contam, 'speed': _speed_summary(rows)}
    if clean_s:
        out.update({'clean_n': clean_s['n'], 'clean_svc_hit_recall': clean_s['hit_recall'],
                    'clean_svc_false_events': clean_s['false_events'],
                    'clean_svc_video_false_alarm': clean_s['video_false_alarm']})
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


def _info_metrics(m, prefix, rec_key, fa_key):
    """run_info 한 줄: 재현율·오탐율·정확도·정밀도·F1 + 혼동행렬."""
    g = lambda k: m.get(prefix + k)  # noqa: E731
    return (f"재현율 {_fmt_pct(g(rec_key))} / 오탐율 {_fmt_pct(g(fa_key))} / "
            f"정확도 {_fmt_pct(g('accuracy'))} / 정밀도 {_fmt_pct(g('precision'))} / "
            f"F1 {_fmt_pct(g('f1'))} (TP {g('tp')} FN {g('fn')} FP {g('fp')} TN {g('tn')})")


def evaluate_folder_accuracy(model, folder_path=config.EVAL_FOLDER_PATH, **kwargs):
    """폴더 평가 진입점 — 결과를 콘솔과 파일에 동시에 남긴다.

    남는 것:
      1) 회차 폴더       outputs/evallogs/<YYMMDD_HHMM>_<모델>_<ptY|ptN>[_span]/
           · 콘솔 전문 로그  eval_<모델>_<ptY|ptN>_<시각>.txt
           · 영상별 결과     eval_<모델>_<ptY|ptN>_<시각>_videos.csv
      2) 요약 한 줄      outputs/evallogs/eval_summary.csv  (append, run 열 = 회차 폴더 이름)
         → 5가지 학습 조합을 한 표에서 비교하려고 계속 덧붙인다.
    """
    log_dir = pathlib.Path(getattr(config, 'EVAL_LOG_DIR',
                                   pathlib.Path('outputs') / 'evallogs'))
    log_dir.mkdir(parents=True, exist_ok=True)
    weights = pathlib.Path(str(config.EVAL_WEIGHTS_PATH)).name
    # 사전학습 여부는 평가하는 가중치의 이름(_ptY_/_ptN_)으로 정한다. 이름에 없으면 config 값.
    tag = next((t for t in ('ptY', 'ptN') if f'_{t}_' in weights),
               getattr(config, 'PRETRAIN_TAG', '?'))
    # A 클립 방식 span 으로 학습한 가중치면 회차 폴더·파일 이름에 _span 을 붙여 'pre' 결과와 구분한다
    name_tag = f'{tag}_span' if '_span_' in weights else tag
    run_dir, stamp = make_run_dir(log_dir, config.MODEL_NAME, name_tag)
    log_path = run_dir / f"eval_{config.MODEL_NAME}_{name_tag}_{stamp}.txt"
    write_run_info(run_dir, [
        ('종류', '평가'),
        ('모델', config.MODEL_NAME),
        ('사전학습', tag),
        ('시작', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
        ('평가 가중치', str(config.EVAL_WEIGHTS_PATH)),
        ('평가 폴더', str(folder_path)),
    ])

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

    speed = m.pop('_speed', None) or {}
    s = speed.get('service') or {}
    info = (pathlib.Path(run_dir, 'run_info.txt').read_text(encoding='utf-8')
            .rstrip('\n').split('\n'))
    write_run_info(run_dir, [tuple(l.split(': ', 1)) for l in info] + [
        ('종료', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
        ('평가 영상 수', m.get('n_videos')),
        ('① 기존 판정(창 P(A)>0.5 1개↑)', _info_metrics(m, 'base_', 'recall', 'false_alarm')),
        ('② 서비스 판정(이벤트 1개↑)', _info_metrics(m, 'svc_', 'video_recall', 'video_false_alarm')),
        (f"② 서비스 위치 기준 재현율(시작 ±{m.get('svc_hit_tol_sec')}초)", _fmt_pct(m.get('svc_hit_recall'))),
        ('② 서비스 오탐 이벤트', m.get('svc_false_events')),
        ('학습 폴더와 겹친 영상', m.get('n_in_train')),
        ('추론 장치', speed.get('device')),
        ('처리 속도(서비스 경로)', f"초당 {s['proc_fps']:.1f} 프레임, 실시간의 {s['realtime_x']:.2f}배 "
                            f"(클립 렌더링 제외)" if s else None),
        ('모델 추론 속도', ' / '.join(f"배치 {b['batch']}: 초당 {b['clips_per_sec']:.1f}개 창"
                                     for b in speed.get('model', [])) or None),
    ])

    # 영상별 결과 CSV — 어떤 영상에서 무엇이 틀렸는지 나중에 다시 보려고
    rows = m.pop('_svc_rows', [])
    if rows:
        per_video = run_dir / f"eval_{config.MODEL_NAME}_{name_tag}_{stamp}_videos.csv"
        with open(per_video, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[로그] 영상별 결과: {per_video}")

    # 요약 CSV 에 한 줄 덧붙인다(모델 비교용)
    summary = log_dir / 'eval_summary.csv'
    # base_* = ① 기존 판정(창 P(A)>0.5 1개↑), svc_* = ② 서비스 판정(이벤트 1개↑).
    # 둘의 tp~f1 은 같은 정의라 그대로 비교된다. svc_hit_recall 은 위치 기준(시작 ±허용 오차).
    # ⚠️ 2026-10-12 이전 요약 파일의 'recall' 은 지금의 base_recall, 'svc_recall' 은
    #    지금의 svc_hit_recall 이다(이름만 바뀜).
    cols = ['datetime', 'run', 'model', 'pretrain', 'weights', 'eval_folder', 'n_videos',
            'base_tp', 'base_fn', 'base_fp', 'base_tn', 'base_recall', 'base_false_alarm',
            'base_accuracy', 'base_precision', 'base_f1',
            'svc_tp', 'svc_fn', 'svc_fp', 'svc_tn', 'svc_video_recall', 'svc_video_false_alarm',
            'svc_accuracy', 'svc_precision', 'svc_f1',
            'svc_hit_tol_sec', 'svc_hit_recall', 'svc_overlap_recall', 'svc_missed',
            'svc_wrong_place', 'svc_split', 'svc_false_events', 'svc_false_per_hour',
            'svc_err_median_f', 'svc_err_median_sec',
            'n_in_train', 'clean_n', 'clean_svc_hit_recall', 'clean_svc_false_events',
            'clean_svc_video_false_alarm',
            # 추론 속도 — 장치가 다르면 비교할 수 없으니 device 와 함께 본다
            'device', 'speed_proc_fps', 'speed_realtime_x', 'speed_windows_eval_pct',
            'speed_model_batch', 'speed_model_clips_per_sec', 'speed_model_ms_per_clip']
    # 컬럼 구성이 바뀐 옛 요약 파일에 이어 쓰면 열이 어긋나므로 옆으로 치워둔다
    if summary.exists():
        with open(summary, encoding='utf-8') as f:
            old_header = f.readline().strip().split(',')
        if old_header != cols:
            moved = summary.with_name(f"eval_summary_old_{stamp}.csv")
            summary.rename(moved)
            print(f"[로그] 요약 CSV 컬럼이 바뀌어 기존 파일을 {moved.name} 로 옮겼습니다")
    row = {'datetime': datetime.now().strftime('%Y-%m-%d %H:%M'), 'run': run_dir.name,
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
