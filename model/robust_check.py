"""변형 견딤 시험 — 학습한 증강과 학습하지 않은 변형(저해상도·압축)에 모델 판단이 얼마나 흔들리는지 잰다.

    python -u -m model.robust_check                      # 평가 폴더 전부, 평가 가중치(GPU 권장, 약 5분)
    python -u -m model.robust_check --weights <가중치.pth>
    python -u -m model.robust_check --videos 4           # 사고·정상 절반씩 일부만(코드 점검용)

평가 영상(config.EVAL_FOLDER_PATH)마다 창을 고른다 — 사고 영상은 충돌 시작 앞뒤(시작 25·20·15·10·5·0프레임 전 시작),
정상 영상은 영상 전체에 고르게 12개. 창마다 30프레임을 학습·서비스와 같이 자르고 줄인 뒤(0번 차량 박스, r=1.0)
변형별로 P(A)를 잰다. 변형은 클립 전체에 같은 값이다(노이즈만 프레임마다, 시드 고정).
  · 학습한 증강(dataset.py 와 같은 연산): 좌우 뒤집기 · 밝게 1.4 · 어둡게 0.6 · 대비 0.7 · 채도 0.7 · 색조 +0.1 ·
    흑백 · 흐림(3×3) · 노이즈 σ8
  · 학습하지 않은 변형: 저해상도(차량 크롭을 64·40px 로 줄였다 다시 키움 = 화면 속 작은 차) · 압축(JPEG 화질 25) ·
    저해상도 64px + 압축(작은 차를 압축이 심한 CCTV 로 본 것)
학습한 증강에는 판단이 덜 흔들리고 학습하지 않은 변형에 크게 흔들리면 증강이 모델에 반영된 것이다. 저해상도에서
정상 창의 P(A)가 오르면 '흐리면 사고' 지름길을 의심한다(학습 데이터의 저화질 영상이 사고 쪽에 몰려 있음).
판정은 창 단위 P(A)>기준(config.A_THRESHOLD, 기본 0.5) 과, 영상의 고른 창 중 하나라도 기준을 넘으면 '사고'로 보는 축약 판정이다(서비스 판정과
다르다 — 변형끼리 비교용). 결과는 화면과 outputs/robust_check/<시각>/report.txt · windows.csv 에 남는다.
"""
import argparse
import csv
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

import config
from dataset import HitAndRunDataset
from device_utils import is_channels_last_3d_supported, is_cuda_like
from evaluate import _read_label
from main import _load_model

A_OFFSETS = (25, 20, 15, 10, 5, 0)            # 사고 창 시작 = 충돌 시작 - 이 값(프레임)
NOISE_SIGMA = 8.0


class _Tee:
    """화면과 파일에 함께 쓴다."""
    def __init__(self, path):
        self.stdout, self.f = sys.stdout, open(path, 'w', encoding='utf-8')

    def write(self, s):
        self.stdout.write(s)
        self.f.write(s)

    def flush(self):
        self.stdout.flush()
        self.f.flush()

    def close(self):
        self.f.close()


# ───────────────────────── 변형 ─────────────────────────
def _pil(fn):
    """PIL 변환을 클립의 모든 프레임에 같은 값으로 — dataset._apply_augmentation 과 같은 연산."""
    return lambda frames, seed: [np.array(fn(Image.fromarray(f))) for f in frames]


def _noise(frames, seed):
    # 학습과 같이 프레임마다 새로 뽑는다(창마다 시드 고정 → 다시 돌려도 같다)
    rng = np.random.default_rng(seed)
    return [np.clip(f.astype(np.float32) + rng.normal(0.0, NOISE_SIGMA, f.shape), 0, 255).astype(np.uint8)
            for f in frames]


def _low(size, jpeg=None):
    """224 크롭을 size 로 줄였다(카메라가 작게 찍은 차) 다시 키운다 — 서비스가 작은 크롭을 키우는 것과 같은 보간."""
    def fn(frames, seed):
        out = []
        for f in frames:
            s = cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA)
            if jpeg:
                s = _jpeg_one(s, jpeg)
            out.append(cv2.resize(s, (f.shape[1], f.shape[0]), interpolation=cv2.INTER_LINEAR))
        return out
    return fn


def _jpeg_one(f, q):
    ok, buf = cv2.imencode('.jpg', cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


TRAINED, UNTRAINED = '학습한 증강', '학습 안 한 변형'
CONDITIONS = [
    ('orig', '원본', None, lambda frames, seed: frames),
    ('flip', '좌우 뒤집기', TRAINED, _pil(TF.hflip)),
    ('bright', '밝게 ×1.4', TRAINED, _pil(lambda im: TF.adjust_brightness(im, 1.4))),
    ('dark', '어둡게 ×0.6', TRAINED, _pil(lambda im: TF.adjust_brightness(im, 0.6))),
    ('contrast', '대비 ×0.7', TRAINED, _pil(lambda im: TF.adjust_contrast(im, 0.7))),
    ('sat', '채도 ×0.7', TRAINED, _pil(lambda im: TF.adjust_saturation(im, 0.7))),
    ('hue', '색조 +0.1', TRAINED, _pil(lambda im: TF.adjust_hue(im, 0.1))),
    ('gray', '흑백', TRAINED, _pil(lambda im: TF.rgb_to_grayscale(im, num_output_channels=3))),
    ('blur', '흐림 3×3', TRAINED, _pil(lambda im: TF.gaussian_blur(im, kernel_size=3))),
    ('noise', f'노이즈 σ{NOISE_SIGMA:g}', TRAINED, _noise),
    ('low64', '저해상도 64px', UNTRAINED, _low(64)),
    ('low40', '저해상도 40px', UNTRAINED, _low(40)),
    ('jpeg25', 'JPEG 화질 25', UNTRAINED, lambda frames, seed: [_jpeg_one(f, 25) for f in frames]),
    ('low64j', '저해상도 64px+압축', UNTRAINED, _low(64, jpeg=25)),
]


# ───────────────────────── 창 고르기·읽기 ─────────────────────────
def _windows(total, gt_sf, n_s):
    L = config.CLIP_LENGTH
    last = max(0, total - L)
    if gt_sf is not None:
        return sorted({min(max(0, gt_sf - k), last) for k in A_OFFSETS})
    return sorted({int(round(v)) for v in np.linspace(0, last, n_s)})


def _read_clips(path, bbox, starts):
    """창 시작점들 → [(T,H,W,3) uint8] — 처음부터 차례로 읽으며(서비스처럼) 필요한 프레임만 자르고 줄인다."""
    L = config.CLIP_LENGTH
    need = {s + i for s in starts for i in range(L)}
    crop_self = SimpleNamespace(resize=tuple(config.RESIZE))
    cap = cv2.VideoCapture(str(path))
    crops, idx = {}, 0
    while idx <= max(need):
        ok, frame = cap.read()
        if not ok:
            break
        if idx in need:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            crops[idx] = HitAndRunDataset._crop_and_pad(crop_self, rgb, bbox, config.R_VALUE)
        idx += 1
    cap.release()
    clips = []
    for s in starts:
        fr = [crops[s + i] for i in range(L) if s + i in crops]
        if not fr:
            continue
        while len(fr) < L:                       # 영상이 짧으면 마지막 프레임으로 채운다(dataset.py 와 같다)
            fr.append(fr[-1])
        clips.append((s, np.stack(fr)))
    return clips


# ───────────────────────── 추론 ─────────────────────────
def _infer(model, device, clips, batch, cl):
    mean = torch.tensor(config.NORM_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor(config.NORM_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1, 1)
    out = []
    with torch.no_grad():
        for i in range(0, len(clips), batch):
            x = torch.from_numpy(np.stack(clips[i:i + batch])).to(device)        # (B,T,H,W,3) uint8
            x = (x.permute(0, 4, 1, 2, 3).float() / 255.0 - mean) / std           # dataset._frames_to_tensor 와 같다
            if cl:
                x = x.contiguous(memory_format=torch.channels_last_3d)
            out += torch.softmax(model(x).float(), 1)[:, 1].cpu().tolist()
    return out


def _auc(pos, neg):
    """사고 창이 정상 창보다 P(A)가 높을 확률(동점 반) — 판정 기준과 무관한 구분력."""
    if not pos or not neg:
        return float('nan')
    p, n = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float((p > n).mean() + 0.5 * (p == n).mean())


# ───────────────────────── 실행 ─────────────────────────
def main():
    ap = argparse.ArgumentParser(description='변형 견딤 시험')
    ap.add_argument('--weights', default=str(config.EVAL_WEIGHTS_PATH), help='시험할 가중치(기본 평가 가중치)')
    ap.add_argument('--folder', default=str(config.EVAL_FOLDER_PATH), help='평가 영상 폴더(mp4 + txt)')
    ap.add_argument('--videos', type=int, default=0, help='사고·정상 절반씩 이 수만(0 = 전부)')
    ap.add_argument('--s-windows', type=int, default=12, help='정상 영상마다 고를 창 수')
    ap.add_argument('--batch', type=int, default=0, help='추론 묶음(0 = GPU 16 · CPU 4)')
    args = ap.parse_args()

    out_dir = Path(config._ROOT) / 'outputs' / 'robust_check' / datetime.now().strftime('%y%m%d_%H%M%S')
    out_dir.mkdir(parents=True, exist_ok=True)
    tee = _Tee(out_dir / 'report.txt')
    sys.stdout = tee
    try:
        return _run(args, out_dir)
    finally:
        sys.stdout = tee.stdout
        tee.close()
        print(f'\n[저장] {out_dir / "report.txt"}')


def _run(args, out_dir):
    model, device = _load_model(args.weights)
    model.eval()
    cl = is_channels_last_3d_supported(device) and bool(config.USE_CHANNELS_LAST)
    if is_cuda_like(device):
        torch.backends.cudnn.benchmark = True
    batch = args.batch or (16 if is_cuda_like(device) else 4)
    print(f'[변형 견딤 시험] {datetime.now():%Y-%m-%d %H:%M} · 장치 {device} · FP32 · 묶음 {batch}')
    print(f'가중치: {args.weights}')

    folder = Path(args.folder)
    vids = []
    for mp4 in sorted(folder.glob('*.mp4')):
        txt = mp4.with_suffix('.txt')
        if not txt.exists():
            continue
        bboxes, gt_sf, _ = _read_label(str(txt))
        if bboxes:
            vids.append((mp4, bboxes.get(config.TARGET_ID, next(iter(bboxes.values()))), gt_sf))
    if args.videos:
        a = [v for v in vids if v[2] is not None][:math.ceil(args.videos / 2)]
        s = [v for v in vids if v[2] is None][:args.videos // 2]
        vids = a + s
    n_a = sum(v[2] is not None for v in vids)
    print(f'영상: {folder} — 사고 {n_a}개 · 정상 {len(vids) - n_a}개 · 변형 {len(CONDITIONS) - 1}가지 + 원본')

    rows = []                                    # (영상, gt, 창 시작, 변형, P(A))
    t0 = time.perf_counter()
    for vi, (mp4, bbox, gt_sf) in enumerate(vids):
        cap = cv2.VideoCapture(str(mp4))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        clips = _read_clips(mp4, bbox, _windows(total, gt_sf, args.s_windows))
        gt = 'A' if gt_sf is not None else 'S'
        for key, _, _, fn in CONDITIONS:
            xs = [np.stack(fn(list(c), seed=vi * 1000 + s)) for s, c in clips]
            for (s, _), p in zip(clips, _infer(model, device, xs, batch, cl)):
                rows.append((mp4.name, gt, s, key, p))
        print(f'  {vi + 1}/{len(vids)} {mp4.name} ({gt}) 창 {len(clips)}개 · {time.perf_counter() - t0:.0f}초', flush=True)

    with open(out_dir / 'windows.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['video', 'gt', 'window_start', 'condition', 'p_a'])
        w.writerows([r[:4] + (f'{r[4]:.6f}',) for r in rows])
    _report(rows)
    return 0


def _report(rows):
    base = {(v, s): p for v, gt, s, k, p in rows if k == 'orig'}
    gt_of = {v: gt for v, gt, *_ in rows}
    n_av = sum(g == 'A' for g in gt_of.values())
    n_sv = len(gt_of) - n_av
    stats = {}
    th = config.A_THRESHOLD     # 창 판정 기준(기본 0.5)
    for key, name, group, _ in CONDITIONS:
        sub = [(v, gt, s, p) for v, gt, s, k, p in rows if k == key]
        pa = [p for v, gt, s, p in sub if gt == 'A']
        ps = [p for v, gt, s, p in sub if gt == 'S']
        d = [abs(p - base[(v, s)]) for v, gt, s, p in sub]
        miss = sum(gt == 'A' and base[(v, s)] > th >= p for v, gt, s, p in sub)     # 사고 창: 맞던 것을 놓침
        fa = sum(gt == 'S' and base[(v, s)] <= th < p for v, gt, s, p in sub)       # 정상 창: 새로 오탐
        vmax = {}
        for v, gt, s, p in sub:
            vmax[v] = max(vmax.get(v, 0.0), p)
        stats[key] = dict(name=name, group=group, pa=np.mean(pa) if pa else float('nan'),
                          ps=np.mean(ps) if ps else float('nan'), d=float(np.mean(d)), miss=miss, fa=fa,
                          auc=_auc(pa, ps), hit=sum(gt_of[v] == 'A' and m > th for v, m in vmax.items()),
                          vfa=sum(gt_of[v] == 'S' and m > th for v, m in vmax.items()),
                          n_a=len(pa), n_s=len(ps))
    o = stats['orig']
    print(f'\n[결과] 창: 사고 {o["n_a"]}개 · 정상 {o["n_s"]}개 (사고 영상 {n_av}개 · 정상 영상 {n_sv}개)')
    print(f'  {"변형":<18}{"사고창 P(A)":>11}{"정상창 P(A)":>11}{"평균 변화":>9}{"놓침+":>6}{"오탐+":>6}'
          f'{"구분력":>8}{"사고영상":>9}{"정상오탐":>9}')
    group = None
    for key, name, g, _ in CONDITIONS:
        if g != group:
            group = g
            if g:
                print(f'  ── {g}')
        s = stats[key]
        print(f'  {name:<18}{s["pa"]:>11.3f}{s["ps"]:>11.3f}{s["d"]:>9.3f}{s["miss"]:>6}{s["fa"]:>6}'
              f'{s["auc"]:>8.3f}{s["hit"]:>6}/{n_av:<2}{s["vfa"]:>6}/{n_sv:<2}')
    print(f'  · 평균 변화 = 원본 대비 |P(A) 차이| 평균 · 놓침+ = 원본에선 {th:g} 를 넘던 사고 창이 넘지 못함 · '
          f'오탐+ = 원본에선 {th:g} 이하던 정상 창이 넘음')
    print('  · 구분력 = 사고 창이 정상 창보다 P(A)가 높을 확률(1 이 완벽, 0.5 는 무작위) · '
          f'사고영상/정상오탐 = 영상의 고른 창 중 하나라도 {th:g} 초과(축약 판정)')

    tr = [stats[k] for k, _, g, _ in CONDITIONS if g == TRAINED]
    un = [stats[k] for k, _, g, _ in CONDITIONS if g == UNTRAINED]
    print('\n[판단]')
    d_tr, d_un = np.mean([s['d'] for s in tr]), np.mean([s['d'] for s in un])
    a_tr, a_un = np.mean([s['auc'] for s in tr]), np.mean([s['auc'] for s in un])
    print(f'  1) 평균 변화: 학습한 증강 {d_tr:.3f} · 학습 안 한 변형 {d_un:.3f} / '
          f'구분력: 원본 {o["auc"]:.3f} · 학습한 증강 평균 {a_tr:.3f} · 학습 안 한 변형 평균 {a_un:.3f}')
    worst_tr = max(tr, key=lambda s: s['d'])
    if d_un > d_tr * 1.5:
        print('     → 학습한 증강에는 덜 흔들리고 학습 안 한 변형에 더 흔들립니다 — 증강이 모델에 반영된 것으로 보입니다.')
    else:
        print('     → 학습한 증강과 학습 안 한 변형의 흔들림이 비슷합니다 — 증강의 효과가 뚜렷하지 않습니다.')
    print(f'     학습한 증강 중 가장 흔들리는 것: {worst_tr["name"]}(평균 변화 {worst_tr["d"]:.3f}, '
          f'놓침+ {worst_tr["miss"]} · 오탐+ {worst_tr["fa"]})')
    low = [stats[k] for k in ('low64', 'low40', 'jpeg25', 'low64j')]
    rise_s = max(s['ps'] - o['ps'] for s in low)
    rise_a = max(s['pa'] - o['pa'] for s in low)
    fa_low = max(s['fa'] for s in low)
    print('  2) 화질을 낮추면(저해상도·압축) P(A)가:')
    print(f'     정상 창 원본 {o["ps"]:.3f} → ' + ' / '.join(f'{s["name"]} {s["ps"]:.3f}' for s in low)
          + f' · 새 오탐 창 최대 {fa_low}개')
    print(f'     사고 창 원본 {o["pa"]:.3f} → ' + ' / '.join(f'{s["name"]} {s["pa"]:.3f}' for s in low))
    if rise_s > 0.05 or fa_low >= max(2, 0.05 * o['n_s']):
        print('     → 화질이 나빠지면 정상 창도 사고 쪽으로 기웁니다 — "흐리면 사고" 지름길을 의심할 만합니다'
              '(저화질 증강·데이터로 보완).')
    elif rise_a > 0.1:
        print('     → 정상 창은 거의 그대로지만 사고 창의 P(A)가 화질이 나빠질수록 오릅니다 — 같은 방향의 기울기가 있어,'
              ' 정상 창이 조금이라도 애매하면 오탐으로 이어질 수 있습니다(저화질 증강으로 보완할 만함).')
    elif max(s['pa'] for s in low) < o['pa'] - 0.1:
        print('     → 화질이 나빠지면 사고 창을 놓치는 쪽입니다(지름길보다는 저해상도에서 정보가 부족한 것).')
    else:
        print('     → 화질이 나빠져도 판단이 크게 기울지 않습니다 — 지름길 징후는 약합니다.')


if __name__ == '__main__':
    sys.exit(main())
