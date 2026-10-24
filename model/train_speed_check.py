"""학습 속도 병목 찾기 — GPU 계산 · 클립 만들기 단계별 · 데이터 로더 처리량 · 실제 학습 단계를 잰다.

    python -u -m model.train_speed_check                # 전부(약 5~10분)
    python -u -m model.train_speed_check --part gpu     # ① GPU 계산만(channels_last 켬/끔)
    python -u -m model.train_speed_check --part data    # ②③ 데이터 준비만(GPU 없어도 됨)

  ① GPU 계산: 학습·검증 한 단계를 channels_last 켬/끔으로(무작위 입력, 데이터 안 읽음).
     AMP·cudnn.benchmark·gradient checkpointing·기울기 자르기(1.0)는 train.py 와 같다.
  ② 클립 하나 만들기 단계별 시간(한 프로세스): 영상 열기 / 시작 프레임으로 이동 / 프레임 풀기 /
     전체 화면 색 변환 / 자르기·줄이기 / 증강 / 텐서 변환 — dataset.py __getitem__ 과 같은 순서.
     '자른 뒤 색 변환'이면 얼마나 줄어드는지도 잰다(결과는 같다).
  ③ 데이터 로더 처리량: 학습과 같은 설정(묶음 15·셔플·prefetch 4)의 DataLoader 로 묶음 하나
     준비 시간을 작업자 수별로(기본: 학습과 같은 min(8, CPU) · CPU 코어 수).
  ④ 실제 학습 단계: ③의 로더(학습과 같은 작업자 수) + 지금 설정의 모델로 묶음당 시간과
     그중 GPU 가 데이터를 기다린 비율.
마지막에 병목(GPU 계산 vs 데이터 준비)을 판정한다.

⚠️ 학습을 멈춘 뒤(GPU·CPU 가 빌 때) 돌린다 — 학습 중이면 둘 다 나눠 써서 시간이 틀린다.
결과는 화면과 outputs/speed_check/<시각>/report.txt 에 남는다.
"""
import argparse
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

import config
import hitandrun_model
from device_utils import is_channels_last_3d_supported
from hitandrun_model import HitAndRun3DCNN


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


# ───────────────────────── ① GPU 계산 ─────────────────────────
def _step_times(model, x, y, channels_last, dev, amp, steps, warmup, train):
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last_3d)
    crit = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=1e-5) if train else None
    scaler = torch.amp.GradScaler('cuda') if (train and amp) else None
    model.train(train)
    times = []
    for i in range(warmup + steps):
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        if train:
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev.type, enabled=amp):
                loss = crit(model(x), y)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        else:
            with torch.no_grad(), torch.autocast(dev.type, enabled=amp):
                model(x)
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        if i >= warmup:
            times.append(time.perf_counter() - t0)
    return np.array(times)


def _gpu_run(channels_last, state, x, y, dev, amp, steps, warmup):
    """같은 초기 가중치로 학습 단계·검증 단계 시간, 최대 메모리, 검증 P(A)를 잰다."""
    model = HitAndRun3DCNN(num_classes=2, pretrained=False)
    model.load_state_dict(state)
    model = model.to(dev)
    if channels_last:
        model = model.to(memory_format=torch.channels_last_3d)
    model.eval()
    xe = x.contiguous(memory_format=torch.channels_last_3d) if channels_last else x
    with torch.no_grad(), torch.autocast(dev.type, enabled=amp):
        p = torch.softmax(model(xe).float(), 1)[:, 1].cpu().numpy()
    if dev.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    t_val = _step_times(model, x, y, channels_last, dev, amp, steps, warmup, train=False)
    t_train = _step_times(model, x, y, channels_last, dev, amp, steps, warmup, train=True)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == 'cuda' else float('nan')
    del model
    if dev.type == 'cuda':
        torch.cuda.empty_cache()
    return t_train, t_val, peak, p


def part_gpu(args, dev, amp):
    """① → {channels_last 켬/끔: 학습 1단계 평균 시간(초)}."""
    print(f'\n① GPU 계산 — 묶음 {args.batch} · {config.CLIP_LENGTH}프레임 · {config.RESIZE[0]}×{config.RESIZE[1]} · '
          f'AMP {"켬" if amp else "끔"} · gradient checkpointing {"켬" if hitandrun_model.GRAD_CHECKPOINT else "끔"} · '
          f'준비 {args.warmup} + 측정 {args.steps}단계 (무작위 입력)', flush=True)
    torch.manual_seed(0)
    state = HitAndRun3DCNN(num_classes=2, pretrained=False).state_dict()
    x = torch.randn(args.batch, 3, config.CLIP_LENGTH, config.RESIZE[1], config.RESIZE[0], device=dev)
    y = torch.randint(0, 2, (args.batch,), device=dev)
    res = {}
    for cl in (True, False):
        name = 'channels_last 켬' if cl else 'channels_last 끔'
        print(f'  {name} 재는 중…', flush=True)
        try:
            res[cl] = _gpu_run(cl, state, x, y, dev, amp, args.steps, args.warmup)
        except torch.cuda.OutOfMemoryError:
            print(f'  {name}: GPU 메모리 부족(OOM) — 이 설정은 이 묶음 크기로 학습할 수 없음')
            torch.cuda.empty_cache()
    if not res:
        return None
    print(f'  {"설정":<18}{"학습 1단계":>14}{"검증 1단계":>14}{"최대 메모리":>12}')
    for cl, (tt, tv, peak, _) in res.items():
        print(f'  {"channels_last " + ("켬" if cl else "끔"):<18}{tt.mean():>9.3f}±{tt.std():.3f}초'
              f'{tv.mean():>9.3f}±{tv.std():.3f}초{peak:>10.2f}GB')
    if len(res) == 2 and not args.smoke:
        (on_t, on_v, _, p_on), (off_t, off_v, _, p_off) = res[True], res[False]
        ep_on, ep_off = on_t.mean() + on_v.mean() / 4, off_t.mean() + off_v.mean() / 4
        faster = 'channels_last 끔' if ep_off < ep_on else 'channels_last 켬'
        d_t = (off_t.mean() - on_t.mean()) / on_t.mean() * 100
        d_v = (off_v.mean() - on_v.mean()) / on_v.mean() * 100
        print(f'  → 빠른 쪽: {faster} — GPU 계산 시간 약 {abs(ep_on - ep_off) / max(ep_on, ep_off) * 100:.0f}% 단축')
        print(f'    끔이 켬보다 학습 단계 {abs(d_t):.0f}% {"빠름" if d_t < 0 else "느림"} · '
              f'검증 단계 {abs(d_v):.0f}% {"빠름" if d_v < 0 else "느림"}')
        print(f'    같은 가중치·입력의 P(A) 최대 차이 {np.abs(p_on - p_off).max():.1e} (반올림 수준이면 결과 같음)')
    print(f'  지금 설정: config.USE_CHANNELS_LAST = {bool(config.USE_CHANNELS_LAST)}')
    return {cl: float(v[0].mean()) for cl, v in res.items()}


# ───────────────────────── ② 클립 하나 만들기 단계별 ─────────────────────────
def part_clip(ds, idxs):
    """② 학습 데이터셋과 같은 순서로 클립을 만들며 단계별 시간을 잰다 → 클립 하나 평균 시간(초)."""
    keys = ['열기', '이동', '풀기', '색 변환(전체 화면)', '자르기·줄이기', '증강', '텐서 변환']
    tot = dict.fromkeys(keys, 0.0)
    color_small, n_frames, n_clips = 0.0, 0, 0
    pc = time.perf_counter
    for i in idxs:
        s = ds.samples[i]
        starts = list(s.get('win_starts') or [s['start_f']])
        s0, n_read = min(starts), max(starts) - min(starts) + ds.clip_length
        t = pc()
        cap = cv2.VideoCapture(s['mp4_path'])
        tot['열기'] += pc() - t
        t = pc()
        cap.set(cv2.CAP_PROP_POS_FRAMES, s0)
        tot['이동'] += pc() - t
        frames = []
        for _ in range(n_read):
            t = pc()
            ok, f = cap.read()
            tot['풀기'] += pc() - t
            if not ok:
                break
            t = pc()
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            tot['색 변환(전체 화면)'] += pc() - t
            t = pc()
            crop = ds._crop_and_pad(rgb, s['bbox'], ds.r_value)
            tot['자르기·줄이기'] += pc() - t
            t = pc()
            cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)       # 비교용: 자른 뒤(224) 색 변환이면 드는 시간
            color_small += pc() - t
            frames.append(crop)
            n_frames += 1
        cap.release()
        if not frames:
            continue
        t = pc()
        frames = ds._apply_augmentation(frames)
        tot['증강'] += pc() - t
        t = pc()
        ds._frames_to_tensor(frames)
        tot['텐서 변환'] += pc() - t
        n_clips += 1
    if not n_clips:
        print('  클립을 읽지 못했습니다')
        return None
    per = {k: v / n_clips for k, v in tot.items()}
    total = sum(per.values())
    print(f'\n② 클립 하나 만들기 (한 프로세스, 학습 클립 {n_clips}개 평균 · 클립당 프레임 {n_frames / n_clips:.0f}장)')
    for k in keys:
        print(f'  {k:<16}{per[k] * 1000:>9.1f} ms  {per[k] / total * 100:>5.1f}%')
    print(f'  {"합계":<16}{total * 1000:>9.1f} ms')
    small = color_small / n_clips
    save = (per['색 변환(전체 화면)'] - small) / total
    print(f'  ※ 색 변환을 자른 뒤(224)에 하면 {per["색 변환(전체 화면)"] * 1000:.1f} → {small * 1000:.1f} ms '
          f'(클립당 {save * 100:.0f}% 단축, 결과는 같음)')
    return {'total': total, 'per': per, 'color_save': save}


# ───────────────────────── ③ 데이터 로더 처리량 ─────────────────────────
def _loader(ds, idxs, batch, workers, pin):
    kw = dict(batch_size=batch, shuffle=True, num_workers=workers, pin_memory=pin)
    if workers > 0:
        kw.update(persistent_workers=True, prefetch_factor=4)
    return DataLoader(Subset(ds, idxs), **kw)


def part_loader(ds, pool, batch, workers_list, pin):
    """③ 작업자 수별 묶음 하나 준비 시간(초). 로더를 만든 순간부터 N번째 묶음까지(작업자 시작 포함)."""
    print(f'\n③ 데이터 로더 처리량 (묶음 {batch}, 학습과 같은 설정 · 작업자 시작부터 잼)')
    out = {}
    for w in workers_list:
        n = max(24, 4 * w)                 # 작업자 수의 4배 묶음 — 첫 묶음들이 동시에 나오는 효과를 묻힌다
        idxs = pool[:batch * n]
        t0 = time.perf_counter()
        dl = _loader(ds, idxs, batch, w, pin)
        it = iter(dl)
        got = 0
        for _ in range(n):
            try:
                next(it)
            except StopIteration:
                break
            got += 1
        el = time.perf_counter() - t0
        del it, dl
        out[w] = el / max(got, 1)
        print(f'  작업자 {w:>2}개: 묶음 하나 {out[w]:.3f}초 (초당 클립 {batch / out[w]:.1f}개, {got}묶음)', flush=True)
    return out


# ───────────────────────── ④ 실제 학습 단계 ─────────────────────────
def part_e2e(ds, pool, batch, workers, dev, amp, n, cl):
    """④ 실제 로더 + 모델(channels_last cl)로 묶음당 시간과 데이터 기다린 시간 → (기다림, 계산) 초."""
    cl = bool(cl) and is_channels_last_3d_supported(dev)
    model = HitAndRun3DCNN(num_classes=2, pretrained=False).to(dev)
    if cl:
        model = model.to(memory_format=torch.channels_last_3d)
    model.train()
    opt = optim.AdamW(model.parameters(), lr=1e-5)
    scaler = torch.amp.GradScaler('cuda') if amp else None
    crit = nn.CrossEntropyLoss()
    dl = _loader(ds, pool[:batch * (n + 2)], batch, workers, dev.type == 'cuda')
    it = iter(dl)
    waits, comps = [], []
    for i in range(n + 1):
        t0 = time.perf_counter()
        try:
            x, y = next(it)
        except StopIteration:
            break
        t1 = time.perf_counter()
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        if cl:
            x = x.contiguous(memory_format=torch.channels_last_3d)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(dev.type, enabled=amp):
            loss = crit(model(x), y)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        loss.item()                                  # 학습 코드처럼 매 단계 값을 꺼낸다(동기화)
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        if i > 0:                                    # 첫 묶음(작업자 시작·cudnn 탐색) 제외
            waits.append(t1 - t0)
            comps.append(t2 - t1)
    del it, dl, model
    if dev.type == 'cuda':
        torch.cuda.empty_cache()
    w, c = float(np.mean(waits)), float(np.mean(comps))
    print(f'  channels_last {"켬" if cl else "끔"}: 묶음 하나 {w + c:.3f}초 = 데이터 기다림 {w:.3f}초'
          f'({w / (w + c) * 100:.0f}%) + GPU 계산 {c:.3f}초  ({len(waits)}묶음)', flush=True)
    return w, c


def _mem_available_gb():
    """남은 메모리(GB). 리눅스(/proc/meminfo)에서만, 못 읽으면 None."""
    try:
        for line in open('/proc/meminfo', encoding='utf-8'):
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) / 2 ** 20
    except OSError:
        pass
    return None


def main():
    ap = argparse.ArgumentParser(description='학습 속도 병목 찾기')
    ap.add_argument('--part', choices=['all', 'gpu', 'data'], default='all',
                    help='all: ①~④ / gpu: ① GPU 계산만 / data: ②③ 데이터 준비만(GPU 불필요)')
    ap.add_argument('--batch', type=int, default=config.TRAIN_BATCH_SIZE, help='묶음 크기(기본 학습과 같음)')
    ap.add_argument('--steps', type=int, default=20, help='① 측정 단계 수(준비 단계 제외)')
    ap.add_argument('--warmup', type=int, default=5, help='① 준비 단계(cudnn 최적 알고리즘 찾기) — 시간에서 뺌')
    ap.add_argument('--clips', type=int, default=40, help='② 단계별로 잴 클립 수')
    ap.add_argument('--workers', default='',
                    help='③ 작업자 수 목록(쉼표). 기본: 학습과 같은 min(8, CPU) 와 CPU 코어 수')
    ap.add_argument('--e2e-batches', type=int, default=30, help='④ 잴 묶음 수')
    ap.add_argument('--data-dir', default=str(config.DATA_DIR), help='학습 데이터 폴더(②~④)')
    ap.add_argument('--force', action='store_true', help='GPU 메모리가 이미 쓰이고 있어도 진행')
    ap.add_argument('--smoke', action='store_true', help='코드 점검용 — CPU·묶음 1·1단계(① 시간 의미 없음)')
    args = ap.parse_args()

    out_dir = Path(config._ROOT) / 'outputs' / 'speed_check' / datetime.now().strftime('%y%m%d_%H%M%S')
    out_dir.mkdir(parents=True, exist_ok=True)
    tee = _Tee(out_dir / 'report.txt')
    sys.stdout = tee
    try:
        return _main(args)
    finally:
        sys.stdout = tee.stdout
        tee.close()
        print(f'\n[저장] {out_dir / "report.txt"}')


def _main(args):
    use_gpu = args.part in ('all', 'gpu') and not args.smoke
    if args.smoke:
        dev, amp = torch.device('cpu'), False
        args.batch, args.steps, args.warmup = 1, 1, 0
    elif use_gpu:
        if not torch.cuda.is_available():
            print('GPU 가 잡히지 않습니다 — 학습 서버에서 실행하세요(데이터 준비만: --part data, 점검: --smoke).')
            return 1
        dev, amp = torch.device('cuda'), True
        torch.backends.cudnn.benchmark = True
        free, total = (v / 2 ** 30 for v in torch.cuda.mem_get_info())
        print(f'[GPU] {torch.cuda.get_device_name(0)} · 빈 메모리 {free:.1f} / {total:.1f} GB')
        if free < 0.85 * total and not args.force:
            print('⚠️ GPU 메모리를 다른 프로세스(학습 등)가 쓰고 있습니다. 학습이 끝난 뒤 실행하세요(--force 로 무시).')
            return 1
    else:
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        amp = dev.type == 'cuda'
    n_cpu = os.cpu_count() or 2
    print(f'[측정] {datetime.now():%Y-%m-%d %H:%M} · CPU 코어 {n_cpu}개 · 부분 {args.part}'
          f'{" (점검용)" if args.smoke else ""}')

    gpu = part_gpu(args, dev, amp) if args.part in ('all', 'gpu') else None
    if args.part == 'gpu':
        return 0

    # ②~④ — 학습과 같은 데이터셋(증강 켬)에서 무작위로 고른 클립
    from dataset import HitAndRunDataset
    print(f'\n[데이터] {args.data_dir} 의 학습 클립 목록을 만드는 중…', flush=True)
    ds = HitAndRunDataset(data_dir=args.data_dir, augment=True)
    if not len(ds):
        print('  학습 클립이 없습니다')
        return 1
    rng = random.Random(0)
    pool = list(range(len(ds)))
    rng.shuffle(pool)
    clip = part_clip(ds, pool[:args.clips])
    clip_sec = clip['total'] if clip else None

    train_w = min(8, n_cpu)                       # train.py _make_loader 와 같은 작업자 수
    if args.workers:
        workers_list = sorted({int(w) for w in args.workers.split(',') if w.strip()})
    else:
        workers_list = sorted({train_w, n_cpu})
    # 작업자 하나가 묶음 하나(약 0.55GB, 묶음 15)를 들고 있으므로 남은 메모리로 상한을 둔다
    mem = _mem_available_gb()
    per_w = 0.55 * args.batch / 15
    if mem is not None:
        cap = max(1, int((mem - 2.0) / per_w))
        if max(workers_list) > cap:
            print(f'\n[메모리] 남은 메모리 {mem:.1f}GB — 작업자 {cap}개까지만 잽니다(작업자당 약 {per_w:.2f}GB)')
            workers_list = sorted({min(w, cap) for w in workers_list})
            train_w = min(train_w, cap)
    loader = part_loader(ds, pool, args.batch, workers_list, pin=dev.type == 'cuda')

    e2e = {}
    if args.part == 'all' and not args.smoke and dev.type == 'cuda':
        print(f'\n④ 실제 학습 단계 — 학습과 같은 로더(작업자 {train_w}개) + 모델, channels_last 켬/끔 각각')
        for cl in (True, False):
            e2e[cl] = part_e2e(ds, pool, args.batch, train_w, dev, amp, args.e2e_batches, cl)

    _verdict(args, gpu, clip, loader, train_w, e2e)
    return 0


def _verdict(args, gpu, clip, loader, train_w, e2e):
    """결과만 보고 판단할 수 있게: 무엇이 오래 걸리나 / channels_last 가 원인이었나 / 지금 병목은 무엇이고 어떻게 줄이나."""
    print('\n' + '=' * 72 + '\n[판정]')
    lb = loader.get(train_w)
    cur = bool(config.USE_CHANNELS_LAST)
    # 1) 무엇이 오래 걸리나
    print('1) 묶음 하나(학습 단계 한 번)에 드는 시간')
    if e2e:
        for cl, (w, c) in e2e.items():
            mark = '  ← 지금 설정' if cl == cur else ''
            print(f'   channels_last {"켬" if cl else "끔"}: {w + c:.3f}초 = 데이터 기다림 {w:.3f}초({w / (w + c) * 100:.0f}%)'
                  f' + GPU 계산 {c:.3f}초{mark}')
    if lb is not None:
        print(f'   (참고) 데이터 준비만: 로더 작업자 {train_w}개로 묶음 하나 {lb:.3f}초'
              + (f' · GPU 계산만(①): 켬 {gpu[True]:.3f}초 / 끔 {gpu[False]:.3f}초' if gpu and len(gpu) == 2 else ''))
    if clip:
        top = sorted(((k, v) for k, v in clip['per'].items()), key=lambda kv: -kv[1])[:3]
        print(f'   클립 하나 만들기 {clip["total"]:.3f}초 중 오래 걸리는 단계: '
              + ', '.join(f'{k} {v * 1000:.0f}ms({v / clip["total"] * 100:.0f}%)' for k, v in top))

    # 2) channels_last 가 원인이었나
    if e2e and len(e2e) == 2:
        on, off = sum(e2e[True]), sum(e2e[False])
        real_x = on / off
        gpu_x = gpu[True] / gpu[False] if gpu and len(gpu) == 2 else None
        print('2) channels_last 가 느린 원인이었나')
        print(f'   실제 학습에서 끔이 켬보다 {real_x:.2f}배 빠름'
              + (f' (GPU 계산만 보면 {gpu_x:.2f}배)' if gpu_x else ''))
        if real_x < 1.1:
            print('   → 아니요. 실제 학습 속도에는 거의 영향이 없습니다(다른 곳이 병목).')
        elif gpu_x and real_x < gpu_x * 0.8:
            print(f'   → 예, 하지만 일부만 이득입니다. 끈 뒤에는 데이터 준비가 병목이 되어 GPU 계산 이득({gpu_x:.2f}배)을'
                  f' 다 못 씁니다 — 체감이 측정보다 작은 이유.')
        else:
            print('   → 예. 실제 학습에서도 GPU 계산 이득이 거의 그대로 납니다.')

    # 3) 지금 설정의 병목과 줄이는 법
    if e2e and cur in e2e:
        w, c = e2e[cur]
        print(f'3) 지금 설정(channels_last {"켬" if cur else "끔"})의 병목')
        if w / (w + c) > 0.2:
            print(f'   → 데이터 준비: GPU 가 묶음마다 {w:.3f}초({w / (w + c) * 100:.0f}%) 데이터를 기다립니다.')
            best_w = min(loader, key=loader.get)
            if best_w != train_w and lb and loader[best_w] < lb * 0.85:
                print(f'     · 작업자 {train_w} → {best_w}개: 로더 묶음 {lb:.3f} → {loader[best_w]:.3f}초 (train.py _make_loader)')
            if clip:
                print(f'     · 색 변환을 자른 뒤에: 클립당 약 {clip["color_save"] * 100:.0f}% 단축(결과 같음)')
                print('     · 그 밖에 ②의 큰 단계(풀기·이동이면 영상 디코딩 자체 — 짧은 키프레임 간격 재인코딩 등)')
        else:
            print(f'   → GPU 계산(데이터 기다림 {w / (w + c) * 100:.0f}%) — 데이터 쪽을 고쳐도 빨라지지 않습니다.')
        print(f'   학습 클립 1,000개당 약 {(w + c) * 1000 / args.batch / 60:.1f}분(검증 제외)')
    elif lb is not None and gpu and cur in gpu:
        print('3) ' + ('데이터 준비가 GPU 계산보다 느립니다(병목 가능성 — ④는 GPU 에서 --part all 로)'
                       if lb > gpu[cur] else 'GPU 계산이 데이터 준비보다 느립니다'))


if __name__ == '__main__':
    sys.exit(main())
