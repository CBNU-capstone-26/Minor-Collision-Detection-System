"""GPU FP16 추론 확인 (학습 서버용) — FP32 와 비교해 속도와 결과 차이를 잰다.

    CUDA_VISIBLE_DEVICES=0 python -u -m model.fp16_check               # 속도 + 실제 창 확률 비교
    CUDA_VISIBLE_DEVICES=0 python -u -m model.fp16_check --full-eval   # + 평가 세트를 FP32·FP16 으로 각각 평가

결과는 화면과 outputs/fp16_check/<시각>/report.txt 에 남는다. --full-eval 의 평가 로그는
평소처럼 outputs/evallogs 에 두 회차로 남는다.

FP16 을 켜도 되는 기준 (셋 다):
  ① 실제 창에서 판정(P(A) 0.5)·거친 탐색(0.3) 기준이 바뀐 창이 없다
  ② --full-eval 에서 기존 판정·서비스 판정 지표가 FP32 와 같다
  ③ 속도 이득이 있다
켜는 법: 서비스(워커)를 띄우는 환경에 INFER_FP16=1  (config.INFER_FP16)
"""
import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import config
from device_utils import is_channels_last_3d_supported
from evaluate import _Tee, evaluate_folder_accuracy
from main import _load_model


def _windows_of(video, txt, n, clip_length, resize):
    """영상 하나에서 고르게 n 개 창(정규화 전 uint8) — 0번 차량 박스 r=1 정사각 크롭(서비스와 같음)."""
    box = None
    for line in open(txt, encoding='utf-8', errors='ignore'):
        p = [t.strip() for t in line.split(',')]
        if len(p) >= 6 and p[0] == 'car' and p[1] == '0':
            box = list(map(int, map(float, p[2:6])))
    if box is None:
        return None, None
    x1, y1, x2, y2 = box
    side = max(x2 - x1, y2 - y1)
    cx, cy = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2
    cap, frames = cv2.VideoCapture(video), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        h, w, _ = f.shape
        a1, b1, a2, b2 = cx - side // 2, cy - side // 2, cx + side // 2, cy + side // 2
        c = f[max(0, b1):min(h, b2), max(0, a1):min(w, a2)]
        c = np.pad(c, ((max(0, -b1), max(0, b2 - h)), (max(0, -a1), max(0, a2 - w)), (0, 0)))
        frames.append(cv2.resize(c, resize))
    cap.release()
    if len(frames) < clip_length:
        return None, None
    frames = np.stack(frames)
    starts = np.unique(np.linspace(0, len(frames) - clip_length, n).round().astype(int))
    return frames, starts


def main():
    ap = argparse.ArgumentParser(description='GPU FP16 추론 확인')
    ap.add_argument('--weights', default=str(config.EVAL_WEIGHTS_PATH), help='확인할 가중치(기본 평가 가중치)')
    ap.add_argument('--per-video', type=int, default=24, help='영상마다 비교할 창 수')
    ap.add_argument('--full-eval', action='store_true', help='평가 세트를 FP32·FP16 으로 각각 평가해 지표 비교')
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print('GPU(CUDA)가 없어 FP16 을 확인할 수 없습니다. GPU 서버에서 실행하세요.')
        sys.exit(1)

    out_dir = Path(config._ROOT) / 'outputs' / 'fp16_check' / datetime.now().strftime('%y%m%d_%H%M')
    out_dir.mkdir(parents=True, exist_ok=True)
    tee = _Tee(out_dir / 'report.txt')
    sys.stdout = tee
    try:
        _run(args)
    finally:
        sys.stdout = tee.stdout
        tee.close()
    print(f'\n[로그] {out_dir / "report.txt"}')


def _run(args):
    config.INFER_DEVICE_TYPE = 'cuda'
    config.EVAL_WEIGHTS_PATH = Path(args.weights)   # --full-eval 평가 로그에 이 가중치 이름이 남게
    model, device = _load_model(args.weights)
    model.eval()
    cap = torch.cuda.get_device_capability(device)
    print(f'GPU: {torch.cuda.get_device_name(device)} (연산능력 {cap[0]}.{cap[1]}) | torch {torch.__version__} | '
          f'CUDA {torch.version.cuda} | cuDNN {torch.backends.cudnn.version()}')
    print(f'가중치: {args.weights}')
    if cap < (7, 0):
        print('⚠️ 연산능력 7.0 미만(Tensor Core 없음) — FP16 이 빠르지 않을 수 있고 서비스는 이 GPU 에서 FP16 을 켜지 않습니다.')
    cl = is_channels_last_3d_supported(device) and config.USE_CHANNELS_LAST

    def run(x, fp16, tf32=True):
        old = torch.backends.cudnn.allow_tf32
        torch.backends.cudnn.allow_tf32 = tf32
        try:
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16, enabled=fp16):
                return model(x).float()
        finally:
            torch.backends.cudnn.allow_tf32 = old

    shape = (3, config.CLIP_LENGTH, config.RESIZE[1], config.RESIZE[0])

    # ── ① 속도 ──
    print('\n[속도] 초당 처리 창 수 (서비스 배치 = '
          f'{config.PREDICT_INFER_BATCH_SIZE_GPU}, 평가 배치 = {config.EVAL_INFER_BATCH_SIZE})')
    print(f"{'배치':>4} {'FP32':>9} {'FP16':>9} {'FP16/FP32':>10}")
    for bs in (1, 2, 4, 8, 16):
        x = torch.randn((bs, *shape), device=device)
        if cl:
            x = x.contiguous(memory_format=torch.channels_last_3d)
        res = {}
        for fp16 in (False, True):
            for _ in range(3):
                run(x, fp16)
            torch.cuda.synchronize()
            n, t0 = 0, time.perf_counter()
            while time.perf_counter() - t0 < 3.0 or n < 5:
                run(x, fp16)
                n += 1
            torch.cuda.synchronize()
            res[fp16] = n * bs / (time.perf_counter() - t0)
        print(f'{bs:>4} {res[False]:>9.1f} {res[True]:>9.1f} {res[True] / res[False]:>9.2f}x')

    # ── ② 실제 창 결과 비교 ──
    mean = torch.tensor(config.NORM_MEAN, device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor(config.NORM_STD, device=device).view(1, 3, 1, 1, 1)
    folder = Path(config.EVAL_FOLDER_PATH)
    vids = sorted(p for p in folder.glob('*.mp4') if p.with_suffix('.txt').exists())
    print(f'\n[결과 비교] {folder} 영상 {len(vids)}개 × 최대 {args.per_video}창, FP32 기준')
    all32, all16, allstrict, flips, per_video = [], [], [], [], []
    for v in vids:
        frames, starts = _windows_of(str(v), str(v.with_suffix('.txt')), args.per_video,
                                     config.CLIP_LENGTH, tuple(config.RESIZE))
        if frames is None:
            continue
        p32, p16, pst = [], [], []
        for i in range(0, len(starts), 8):
            s = starts[i:i + 8]
            u8 = torch.from_numpy(np.stack([frames[k:k + config.CLIP_LENGTH] for k in s])).to(device)
            x = (u8.permute(0, 4, 1, 2, 3).float().div(255) - mean) / std
            if cl:
                x = x.contiguous(memory_format=torch.channels_last_3d)
            p32.append(F.softmax(run(x, False), 1)[:, 1].cpu())
            p16.append(F.softmax(run(x, True), 1)[:, 1].cpu())
            pst.append(F.softmax(run(x, False, tf32=False), 1)[:, 1].cpu())
        p32, p16, pst = torch.cat(p32), torch.cat(p16), torch.cat(pst)
        all32.append(p32); all16.append(p16); allstrict.append(pst)
        d = (p16 - p32).abs()
        per_video.append((float(d.max()), v.name))
        for k, s in enumerate(starts):
            for th in (0.5, 0.3):
                if (p32[k] > th) != (p16[k] > th):
                    flips.append((v.name, int(s), th, float(p32[k]), float(p16[k])))
    p32, p16, pst = torch.cat(all32), torch.cat(all16), torch.cat(allstrict)
    d16, dtf = (p16 - p32).abs(), (pst - p32).abs()
    print(f'  창 {len(p32)}개')
    print(f'  FP16 vs FP32       : P(A) 차이 최대 {d16.max():.2e} / 평균 {d16.mean():.2e}')
    print(f'  (참고) TF32 끈 FP32 vs 지금 FP32 : 최대 {dtf.max():.2e} / 평균 {dtf.mean():.2e}  '
          f'← GPU FP32 도 기본 TF32 라 이미 이 정도 차이가 있다')
    print('  차이가 큰 영상: ' + ', '.join(f'{n}({m:.1e})' for m, n in sorted(per_video, reverse=True)[:5]))
    f05 = [f for f in flips if f[2] == 0.5]
    f03 = [f for f in flips if f[2] == 0.3]
    print(f'  판정이 바뀐 창: 판정 기준 0.5 → {len(f05)}개 / 거친 탐색 기준 0.3 → {len(f03)}개')
    for name, s, th, a, b in flips[:20]:
        print(f'    {name} 창 시작 {s}f: 기준 {th}  FP32 {a:.4f} → FP16 {b:.4f}')

    # ── ③ 평가 세트 전체 (선택) ──
    same = None
    if args.full_eval:
        keys = [('base_recall', '① 기존 판정 재현율'), ('base_false_alarm', '① 기존 판정 오탐율'),
                ('base_f1', '① 기존 판정 F1'), ('svc_video_recall', '② 서비스 판정 재현율'),
                ('svc_video_false_alarm', '② 서비스 판정 오탐율'), ('svc_f1', '② 서비스 판정 F1'),
                ('svc_hit_recall', '② 위치 기준 재현율'), ('svc_false_events', '② 오탐 이벤트'),
                ('svc_err_median_f', '② 시작 오차 중앙값(f)'), ('speed_proc_fps', '처리 속도(fps)')]
        got = {}
        for fp16 in (False, True):
            config.INFER_FP16 = fp16
            print(f"\n{'=' * 60}\n[평가 세트] {'FP16' if fp16 else 'FP32'}\n{'=' * 60}")
            got[fp16] = evaluate_folder_accuracy(model, folder_path=config.EVAL_FOLDER_PATH) or {}
        config.INFER_FP16 = False
        print(f"\n[평가 지표 비교]\n{'지표':<24}{'FP32':>10}{'FP16':>10}")
        for k, label in keys:
            a, b = got[False].get(k), got[True].get(k)
            mark = '' if a == b or k == 'speed_proc_fps' else '  ← 다름'
            print(f'{label:<24}{str(a):>10}{str(b):>10}{mark}')
        same = all(got[False].get(k) == got[True].get(k) for k, _ in keys if k != 'speed_proc_fps')

    # ── 판단 ──
    print('\n[판단]')
    ok1 = not flips
    print(f"  ① 판정이 바뀐 창 없음: {'예' if ok1 else f'아니오({len(flips)}건)'}")
    if same is not None:
        print(f"  ② 평가 지표 같음: {'예' if same else '아니오'}")
    else:
        print('  ② 평가 지표: --full-eval 로 확인하세요')
    print('  ③ 속도 이득: 위 [속도] 표의 서비스 배치 줄을 보세요')
    if ok1 and same:
        print('  → FP16 을 켜도 됩니다(속도 이득이 있다면). 서비스 환경에 INFER_FP16=1')
    elif not ok1 or same is False:
        print('  → FP16 은 보류하세요(결과가 달라지는 경우가 있음).')


if __name__ == '__main__':
    main()
