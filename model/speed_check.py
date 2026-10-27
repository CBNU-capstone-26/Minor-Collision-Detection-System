"""학습 속도 점검 — 병목을 찾고, 결과를 바꾸지 않고 빠르게 하는 설정 후보를 잰다.

    python -u -m model.speed_check                # 전부(약 10~15분)
    python -u -m model.speed_check --part gpu     # ①⑤⑥ GPU 계산만
    python -u -m model.speed_check --part data    # ②③⑦ 데이터 준비만(GPU 없어도 됨)

  ① GPU 계산: 학습·검증 한 단계를 channels_last 켬/끔으로(무작위 입력, 데이터 안 읽음).
     AMP·cudnn.benchmark·gradient checkpointing·기울기 자르기(1.0)는 train.py 와 같다.
  ② 클립 하나 만들기 단계별 시간(한 프로세스): 영상 열기 / 시작 프레임으로 이동 / 프레임 풀기 /
     전체 화면 색 변환 / 자르기·줄이기 / 증강 / 텐서 변환 — dataset.py __getitem__ 과 같은 순서.
     '자른 뒤 색 변환'이면 얼마나 줄어드는지도 잰다(결과는 같다).
  ③ 데이터 로더 처리량: 학습과 같은 설정(묶음 15·셔플·prefetch 4)의 DataLoader 로 묶음 하나
     준비 시간을 작업자 수별(기본: 학습과 같은 min(8, CPU) · CPU 코어 수) × 작업자당 OpenCV 스레드
     (기본 / 1개 — config.TRAIN_CV2_THREADS)로.
  ④ 실제 학습 단계: 학습과 같은 로더 + 지금 설정의 모델로, OpenCV 스레드 기본 / 1개 각각
     묶음당 시간과 그중 GPU 가 데이터를 기다린 비율.
  ⑤ gradient checkpointing 범위: backward 때 다시 계산하는 앞쪽 블록 수(지금 4 = 줄기~res4 / 3 / 2 / 0 = 끔)별
     학습 한 단계 시간과 최대 메모리(hitandrun_model.GRAD_CHECKPOINT_BLOCKS). 기울기는 같고 BN 이동평균도 맞춘다.
  ⑥ torch.compile: 컴파일 시간, 학습·검증 한 단계 시간, 같은 가중치·입력의 P(A) 차이.
  ⑦ 검증 클립 저장본(config.TRAIN_VAL_CACHE): 저장본 없이 / 처음(만들며 저장) / 다음부터(읽기) 묶음 하나
     시간과 결과가 같은지. 임시 폴더에 만들고 끝나면 지운다.
마지막에 병목과 바꿀 만한 설정을 판정한다.

⚠️ 학습을 멈춘 뒤(GPU·CPU 가 빌 때) 돌린다 — 학습 중이면 둘 다 나눠 써서 시간이 틀린다.
결과는 화면과 outputs/speed_check/<시각>/report.txt 에 남는다.
"""
import argparse
import copy
import functools
import os
import random
import shutil
import sys
import tempfile
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
from dataset import limit_cv2_threads
from device_utils import is_channels_last_3d_supported
from hitandrun_model import HitAndRun3DCNN

BLOCK_NAMES = ['줄기', 'res2', 'res3', 'res4', 'res5']
CV2_DEFAULT_THREADS = cv2.getNumThreads()       # 바꾸기 전 OpenCV 기본 스레드 수(③④⑦ 뒤 되돌림)


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


def _cur_cl(dev):
    """지금 설정(config.USE_CHANNELS_LAST)으로 channels_last 를 쓰는지."""
    return bool(config.USE_CHANNELS_LAST) and is_channels_last_3d_supported(dev)


def _build(state, dev, cl):
    model = HitAndRun3DCNN(num_classes=2, pretrained=False)
    model.load_state_dict(state)
    model = model.to(dev)
    if cl:
        model = model.to(memory_format=torch.channels_last_3d)
    return model


def _p_a(model, x, dev, amp, cl):
    """검증 모드 P(A)."""
    model.eval()
    xe = x.contiguous(memory_format=torch.channels_last_3d) if cl else x
    with torch.no_grad(), torch.autocast(dev.type, enabled=amp):
        return torch.softmax(model(xe).float(), 1)[:, 1].cpu().numpy()


def _free_gpu(dev):
    if dev.type == 'cuda':
        torch.cuda.empty_cache()


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
    model = _build(state, dev, channels_last)
    p = _p_a(model, x, dev, amp, channels_last)
    if dev.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    t_val = _step_times(model, x, y, channels_last, dev, amp, steps, warmup, train=False)
    t_train = _step_times(model, x, y, channels_last, dev, amp, steps, warmup, train=True)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == 'cuda' else float('nan')
    del model
    _free_gpu(dev)
    return t_train, t_val, peak, p


def part_gpu(args, dev, amp, state, x, y):
    """① → {channels_last 켬/끔: 학습 1단계 평균 시간(초)}."""
    print(f'\n① GPU 계산 — 묶음 {args.batch} · {config.CLIP_LENGTH}프레임 · {config.RESIZE[0]}×{config.RESIZE[1]} · '
          f'AMP {"켬" if amp else "끔"} · gradient checkpointing {"켬" if hitandrun_model.GRAD_CHECKPOINT else "끔"} · '
          f'준비 {args.warmup} + 측정 {args.steps}단계 (무작위 입력)', flush=True)
    res = {}
    for cl in (True, False):
        name = 'channels_last 켬' if cl else 'channels_last 끔'
        print(f'  {name} 재는 중…', flush=True)
        try:
            res[cl] = _gpu_run(cl, state, x, y, dev, amp, args.steps, args.warmup)
        except torch.cuda.OutOfMemoryError:
            print(f'  {name}: GPU 메모리 부족(OOM) — 이 설정은 이 묶음 크기로 학습할 수 없음')
            _free_gpu(dev)
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


# ───────────────────────── ⑤ gradient checkpointing 범위 ─────────────────────────
def part_ckpt(args, dev, amp, state, x, y):
    """⑤ 다시 계산하는 앞쪽 블록 수별 학습 한 단계 시간·최대 메모리 → {블록 수: 결과}, 기본 블록 수."""
    if not hitandrun_model.GRAD_CHECKPOINT:
        print('\n⑤ gradient checkpointing 이 꺼져 있어(환경변수 GRAD_CHECKPOINT=0) 범위 비교를 건너뜁니다')
        return None
    cl = _cur_cl(dev)
    default = HitAndRun3DCNN(num_classes=2, pretrained=False).default_checkpoint_blocks()
    choices = sorted({default, 3, 2, 0}, reverse=True)
    print(f'\n⑤ gradient checkpointing 범위 — 학습 한 단계, 묶음 {args.batch} · AMP {"켬" if amp else "끔"} · '
          f'channels_last {"켬" if cl else "끔"}(지금 설정) · 준비 {args.warmup} + 측정 {args.steps}단계', flush=True)
    total = torch.cuda.get_device_properties(dev).total_memory / 2 ** 30 if dev.type == 'cuda' else float('nan')
    saved = hitandrun_model.GRAD_CHECKPOINT_BLOCKS
    out = {}
    try:
        for k in choices:
            name = _ckpt_name(k, default)
            print(f'  {name} 재는 중…', flush=True)
            hitandrun_model.GRAD_CHECKPOINT_BLOCKS = k    # 모델을 만들 때 BN momentum 도 맞춘다
            model = None
            try:
                model = _build(state, dev, cl)
                if dev.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats()
                t = _step_times(model, x, y, cl, dev, amp, args.steps, args.warmup, train=True)
                if dev.type == 'cuda':
                    peak = torch.cuda.max_memory_allocated() / 2 ** 30
                    resv = torch.cuda.max_memory_reserved() / 2 ** 30
                else:
                    peak = resv = float('nan')
                out[k] = {'t': float(t.mean()), 'sd': float(t.std()), 'peak': peak, 'resv': resv, 'oom': False}
            except torch.cuda.OutOfMemoryError:
                out[k] = {'oom': True}
            del model
            _free_gpu(dev)
    finally:
        hitandrun_model.GRAD_CHECKPOINT_BLOCKS = saved
    base = out.get(default)
    print(f'  {"다시 계산하는 블록":<22}{"학습 1단계":>14}{"지금 대비":>10}{"최대 메모리(예약)":>20}')
    for k in choices:
        r = out[k]
        if r['oom']:
            print(f'  {_ckpt_name(k, default):<22}{"GPU 메모리 부족(OOM)":>24}')
            continue
        rel = f'{(r["t"] / base["t"] - 1) * 100:+.0f}%' if base and not base['oom'] else '-'
        print(f'  {_ckpt_name(k, default):<22}{r["t"]:>9.3f}±{r["sd"]:.3f}초{rel:>10}'
              f'{r["peak"]:>11.2f}GB({r["resv"]:.2f})')
    print(f'  GPU 전체 메모리 {total:.1f}GB · 기울기는 모두 같고, 범위가 바뀌는 블록은 BN momentum 을 맞춰 결과가 같다')
    return {'res': out, 'default': default, 'total': total}


def _ckpt_name(k, default):
    if k == 0:
        name = '없음(끔)'
    else:
        name = f'{BLOCK_NAMES[0]}~{BLOCK_NAMES[k - 1]}' if k > 1 else BLOCK_NAMES[0]
    return f'{k}개: {name}' + (' ← 지금' if k == default else '')


# ───────────────────────── ⑥ torch.compile ─────────────────────────
def part_compile(args, dev, amp, state, x, y):
    """⑥ 컴파일 전후 학습·검증 한 단계 시간과 P(A) 차이 → 결과 dict, 실패하면 {'error': ...}."""
    cl = _cur_cl(dev)
    print(f'\n⑥ torch.compile — 묶음 {x.shape[0]} · 입력 {tuple(x.shape[2:])} · channels_last {"켬" if cl else "끔"}(지금 설정)'
          f'{" · 점검용 작은 입력" if args.smoke else ""}', flush=True)
    model = None
    try:
        import torch._dynamo
        torch._dynamo.reset()
        model = _build(state, dev, cl)
        p_eager = _p_a(model, x, dev, amp, cl)
        cmodel = torch.compile(model)
        print('  컴파일 중(검증 모드)…', flush=True)
        t0 = time.perf_counter()
        p_comp = _p_a(cmodel, x, dev, amp, cl)            # 첫 호출 = 컴파일
        c_val = time.perf_counter() - t0
        print('  컴파일 중(학습 모드)…', flush=True)
        t0 = time.perf_counter()
        _step_times(cmodel, x, y, cl, dev, amp, 1, 0, train=True)
        c_train = time.perf_counter() - t0
        print('  재는 중…', flush=True)
        # 같은 모델의 컴파일 전(model)·후(cmodel)를 번갈아 잰다 — 학습 단계가 가중치를 바꿔도 시간과는 무관
        e_val = _step_times(model, x, y, cl, dev, amp, args.steps, args.warmup, train=False)
        k_val = _step_times(cmodel, x, y, cl, dev, amp, args.steps, args.warmup, train=False)
        e_tr = _step_times(model, x, y, cl, dev, amp, args.steps, args.warmup, train=True)
        k_tr = _step_times(cmodel, x, y, cl, dev, amp, args.steps, args.warmup, train=True)
    except Exception as e:                                  # 컴파일러 없음·지원 안 되는 연산 등
        msg = f'{type(e).__name__}: {str(e).strip().splitlines()[0] if str(e).strip() else ""}'[:300]
        print(f'  실패 — {msg}')
        print('  (이 환경에서는 torch.compile 을 쓸 수 없음 — 학습은 지금처럼)')
        return {'error': msg}
    finally:
        del model
        _free_gpu(dev)
    diff = float(np.abs(p_eager - p_comp).max())
    print(f'  {"":<10}{"학습 1단계":>14}{"검증 1단계":>14}')
    print(f'  {"컴파일 전":<10}{e_tr.mean():>9.3f}±{e_tr.std():.3f}초{e_val.mean():>9.3f}±{e_val.std():.3f}초')
    print(f'  {"컴파일 후":<10}{k_tr.mean():>9.3f}±{k_tr.std():.3f}초{k_val.mean():>9.3f}±{k_val.std():.3f}초')
    print(f'  → 학습 {e_tr.mean() / k_tr.mean():.2f}배 · 검증 {e_val.mean() / k_val.mean():.2f}배 · '
          f'컴파일 시간 학습 {c_train:.0f}초 + 검증 {c_val:.0f}초(학습 시작 때 한 번)')
    print(f'    같은 가중치·입력의 P(A) 최대 차이 {diff:.1e} (반올림 수준이면 결과 같음)')
    return {'e_tr': float(e_tr.mean()), 'k_tr': float(k_tr.mean()), 'e_val': float(e_val.mean()),
            'k_val': float(k_val.mean()), 'c_sec': c_train + c_val, 'diff': diff}


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
def _loader(ds, idxs, batch, workers, pin, threads=None, shuffle=True):
    """학습과 같은 설정의 로더. threads = 작업자마다 OpenCV 스레드 수(None = 기본, train.py _make_loader 와 같은 방식)."""
    ds.cv2_threads = threads                     # 영상 풀기 스레드(작업자에 데이터셋이 복사될 때 함께 간다)
    # OpenCV 계산 스레드는 부모에서 먼저 정해 작업자가 물려받게 한다(작업자 안에서 처음 바꾸면 멈춤 —
    # dataset.limit_cv2_threads 주석). 작업자는 이 로더를 처음 돌릴 때 생긴다.
    cv2.setNumThreads(threads or CV2_DEFAULT_THREADS)
    kw = dict(batch_size=batch, shuffle=shuffle, num_workers=workers, pin_memory=pin)
    if workers > 0:
        kw.update(persistent_workers=True, prefetch_factor=4)
        if threads:
            kw['worker_init_fn'] = functools.partial(limit_cv2_threads, threads)
    return DataLoader(Subset(ds, idxs), **kw)


def _tname(threads):
    return '작업자당 1스레드' if threads == 1 else (f'작업자당 {threads}스레드' if threads else 'OpenCV 기본')


def _time_batches(dl, n):
    """로더를 만든 순간부터 n번째 묶음까지(작업자 시작 포함) → (묶음당 초, 받은 묶음 수)."""
    t0 = time.perf_counter()
    it = iter(dl)
    got = 0
    for _ in range(n):
        try:
            next(it)
        except StopIteration:
            break
        got += 1
    el = time.perf_counter() - t0
    del it
    return el / max(got, 1), got


def part_loader(ds, pool, batch, workers_list, pin):
    """③ (작업자 수, OpenCV 스레드)별 묶음 하나 준비 시간(초)."""
    print(f'\n③ 데이터 로더 처리량 (묶음 {batch}, 학습과 같은 설정 · 작업자 시작부터 잼)')
    out = {}
    for w in workers_list:
        n = max(24, 4 * w)                 # 작업자 수의 4배 묶음 — 첫 묶음들이 동시에 나오는 효과를 묻힌다
        idxs = pool[:batch * n]
        # 1스레드를 먼저 잰다 — 같은 영상을 두 번째로 읽을 때 디스크 캐시 이득이 지금 설정(기본) 쪽에 가도록(보수적)
        for threads in (1, None):
            dl = _loader(ds, idxs, batch, w, pin, threads)
            sec, got = _time_batches(dl, n)
            del dl
            out[(w, threads)] = sec
            print(f'  작업자 {w:>2}개 · {_tname(threads):<10}: 묶음 하나 {sec:.3f}초 '
                  f'(초당 클립 {batch / sec:.1f}개, {got}묶음)', flush=True)
    _restore_threads(ds)
    return out


def _restore_threads(ds):
    ds.cv2_threads = getattr(config, 'TRAIN_CV2_THREADS', None)
    cv2.setNumThreads(CV2_DEFAULT_THREADS)


# ───────────────────────── ⑦ 검증 클립 저장본 ─────────────────────────
def part_val_cache(ds, pool, batch, workers, pin, n_batches):
    """⑦ 검증처럼(증강 없음·순서대로) 묶음 하나 시간: 저장본 없이 / 처음(만들며 저장) / 다음부터(읽기)."""
    idxs = pool[:batch * n_batches]
    need = len(idxs) * config.CLIP_LENGTH * config.RESIZE[0] * config.RESIZE[1] * 3 / 2 ** 30
    root = Path(config.TRAIN_VAL_CACHE_DIR).parent
    print(f'\n⑦ 검증 클립 저장본 (검증처럼 증강 없이 순서대로 · 작업자 {workers}개 · 클립 {len(idxs)}개, 임시 약 {need:.1f}GB)')
    try:
        root.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(root).free / 2 ** 30
    except OSError as e:
        print(f'  저장 폴더를 쓸 수 없어 건너뜁니다: {e}')
        return None
    if free < need * 2 + 1:
        print(f'  디스크 여유 {free:.1f}GB — 모자라 건너뜁니다')
        return None
    tmp = Path(tempfile.mkdtemp(prefix='speed_check_', dir=root))
    try:
        # 학습 데이터셋(증강 켬)을 복사해 검증처럼 바꾼다 — 클립 목록을 다시 만들지 않으려고
        plain = copy.copy(ds)
        plain.augment, plain.cache_dir, plain._cache_off = False, None, False
        cached = copy.copy(plain)
        cached.cache_dir = tmp
        res = {}
        for name, d in (('none', plain), ('first', cached), ('cached', cached)):
            dl = _loader(d, idxs, batch, workers, pin, getattr(config, 'TRAIN_CV2_THREADS', None), shuffle=False)
            res[name], _ = _time_batches(dl, n_batches)
            del dl
        same = all(torch.equal(plain[i][0], cached[i][0]) for i in idxs[:5])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)    # 이 점검이 만든 임시 폴더만 지운다
        cv2.setNumThreads(CV2_DEFAULT_THREADS)
    print(f'  저장본 없이(지금까지)   : 묶음 하나 {res["none"]:.3f}초')
    print(f'  처음(만들며 저장)       : 묶음 하나 {res["first"]:.3f}초  ← 첫 학습의 첫 검증 때 한 번')
    print(f'  다음부터(저장본 읽기)   : 묶음 하나 {res["cached"]:.3f}초')
    print(f'  저장본으로 만든 입력이 영상에서 만든 것과 같은가(클립 5개): {"예" if same else "아니오 ⚠️"}')
    print('  ※ 방금 쓴 파일이라 메모리(디스크 캐시)에서 읽혔을 수 있다 — 서버 메모리가 저장본(약 5GB)보다 넉넉하면 실제도 같다')
    res['same'] = same
    return res


# ───────────────────────── ④ 실제 학습 단계 ─────────────────────────
def part_e2e(ds, pool, batch, workers, dev, amp, n, threads):
    """④ 실제 로더 + 모델(지금 설정)로 묶음당 시간과 데이터 기다린 시간 → (기다림, 계산) 초."""
    cl = _cur_cl(dev)
    model = HitAndRun3DCNN(num_classes=2, pretrained=False).to(dev)
    if cl:
        model = model.to(memory_format=torch.channels_last_3d)
    model.train()
    opt = optim.AdamW(model.parameters(), lr=1e-5)
    scaler = torch.amp.GradScaler('cuda') if amp else None
    crit = nn.CrossEntropyLoss()
    dl = _loader(ds, pool[:batch * (n + 2)], batch, workers, dev.type == 'cuda', threads)
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
    _free_gpu(dev)
    _restore_threads(ds)
    w, c = float(np.mean(waits)), float(np.mean(comps))
    print(f'  {_tname(threads):<10}: 묶음 하나 {w + c:.3f}초 = 데이터 기다림 {w:.3f}초'
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
    ap = argparse.ArgumentParser(description='학습 속도 점검')
    ap.add_argument('--part', choices=['all', 'gpu', 'data'], default='all',
                    help='all: ①~⑦ / gpu: ①⑤⑥ GPU 계산만 / data: ②③⑦ 데이터 준비만(GPU 불필요)')
    ap.add_argument('--batch', type=int, default=config.TRAIN_BATCH_SIZE, help='묶음 크기(기본 학습과 같음)')
    ap.add_argument('--steps', type=int, default=20, help='①⑤⑥ 측정 단계 수(준비 단계 제외)')
    ap.add_argument('--warmup', type=int, default=5, help='①⑤⑥ 준비 단계(cudnn 최적 알고리즘 찾기) — 시간에서 뺌')
    ap.add_argument('--clips', type=int, default=40, help='② 단계별로 잴 클립 수')
    ap.add_argument('--workers', default='',
                    help='③ 작업자 수 목록(쉼표). 기본: 학습과 같은 min(8, CPU) 와 CPU 코어 수')
    ap.add_argument('--e2e-batches', type=int, default=30, help='④ 잴 묶음 수')
    ap.add_argument('--val-batches', type=int, default=16, help='⑦ 잴 묶음 수')
    ap.add_argument('--no-compile', action='store_true', help='⑥ torch.compile 을 건너뜀')
    ap.add_argument('--data-dir', default=str(config.DATA_DIR), help='학습 데이터 폴더(②③④⑦)')
    ap.add_argument('--force', action='store_true', help='GPU 메모리가 이미 쓰이고 있어도 진행')
    ap.add_argument('--smoke', action='store_true', help='코드 점검용 — CPU·묶음 1·1단계(시간 의미 없음)')
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

    gpu = ckpt = comp = None
    if args.part in ('all', 'gpu'):
        torch.manual_seed(0)
        state = HitAndRun3DCNN(num_classes=2, pretrained=False).state_dict()
        x = torch.randn(args.batch, 3, config.CLIP_LENGTH, config.RESIZE[1], config.RESIZE[0], device=dev)
        y = torch.randint(0, 2, (args.batch,), device=dev)
        gpu = part_gpu(args, dev, amp, state, x, y)
        ckpt = part_ckpt(args, dev, amp, state, x, y)
        if args.no_compile:
            print('\n⑥ torch.compile — 건너뜀(--no-compile)')
        else:
            # 점검용(CPU)은 작은 입력으로 코드 경로만 확인한다 — 원래 크기는 CPU 컴파일·메모리가 너무 크다
            xc = torch.randn(1, 3, 8, 64, 64, device=dev) if args.smoke else x
            comp = part_compile(args, dev, amp, state, xc, y[:xc.shape[0]])
        del x, y
        _free_gpu(dev)
    if args.part == 'gpu':
        _verdict(args, gpu, None, None, None, {}, ckpt, comp, None)
        return 0

    # ②③④⑦ — 학습과 같은 데이터셋(증강 켬)에서 무작위로 고른 클립
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
    pin = dev.type == 'cuda'
    loader = part_loader(ds, pool, args.batch, workers_list, pin)
    val = part_val_cache(ds, pool, args.batch, train_w, pin, args.val_batches)

    e2e = {}
    if args.part == 'all' and not args.smoke and dev.type == 'cuda':
        print(f'\n④ 실제 학습 단계 — 학습과 같은 로더(작업자 {train_w}개) + 지금 설정의 모델, OpenCV 스레드별')
        for threads in (1, None):                 # ③과 같은 이유로 1스레드 먼저
            e2e[threads] = part_e2e(ds, pool, args.batch, train_w, dev, amp, args.e2e_batches, threads)

    _verdict(args, gpu, clip, loader, train_w, e2e, ckpt, comp, val)
    return 0


def _verdict(args, gpu, clip, loader, train_w, e2e, ckpt, comp, val):
    """결과만 보고 판단할 수 있게: 무엇이 오래 걸리나 / 바꿀 만한 설정은 무엇인가."""
    print('\n' + '=' * 72 + '\n[판정]')
    cur_t = getattr(config, 'TRAIN_CV2_THREADS', None)
    no = 0

    # 1) 무엇이 오래 걸리나
    if e2e or clip or gpu:
        no += 1
        print(f'{no}) 학습 묶음 하나(학습 단계 한 번)에 드는 시간')
        for t, (w, c) in e2e.items():
            mark = '  ← 지금 설정' if t == cur_t else ''
            print(f'   {_tname(t)}: {w + c:.3f}초 = 데이터 기다림 {w:.3f}초({w / (w + c) * 100:.0f}%)'
                  f' + GPU 계산 {c:.3f}초{mark}')
        if loader and (train_w, cur_t) in loader:
            print(f'   (참고) 데이터 준비만: 작업자 {train_w}개로 묶음 하나 {loader[(train_w, cur_t)]:.3f}초')
        if gpu and len(gpu) == 2:
            cur_cl = bool(config.USE_CHANNELS_LAST)
            print(f'   (참고) GPU 계산만(①): channels_last 켬 {gpu[True]:.3f}초 / 끔 {gpu[False]:.3f}초'
                  f' — 지금 설정 {"켬" if cur_cl else "끔"}'
                  + ('' if args.smoke or gpu[cur_cl] <= min(gpu.values()) * 1.05
                     else ' ⚠️ 다른 쪽이 빠름(config.USE_CHANNELS_LAST)'))
        if clip:
            top = sorted(((k, v) for k, v in clip['per'].items()), key=lambda kv: -kv[1])[:3]
            print(f'   클립 하나 만들기 {clip["total"]:.3f}초 중 오래 걸리는 단계: '
                  + ', '.join(f'{k} {v * 1000:.0f}ms({v / clip["total"] * 100:.0f}%)' for k, v in top))
        if e2e and cur_t in e2e:
            w, c = e2e[cur_t]
            if w / (w + c) > 0.2:
                print(f'   → 병목은 데이터 준비: GPU 가 묶음마다 {w:.3f}초({w / (w + c) * 100:.0f}%) 데이터를 기다립니다.')
            else:
                print(f'   → 병목은 GPU 계산(데이터 기다림 {w / (w + c) * 100:.0f}%) — 데이터 쪽을 고쳐도 빨라지지 않습니다.')
            print(f'   학습 클립 1,000개당 약 {(w + c) * 1000 / args.batch / 60:.1f}분(검증 제외)')

    # 2) 작업자당 OpenCV 스레드
    if e2e and 1 in e2e and None in e2e:
        a, b = sum(e2e[None]), sum(e2e[1])
        src = f'실제 학습 묶음 {a:.3f} → {b:.3f}초'
    elif loader and (train_w, 1) in loader and (train_w, None) in loader:
        a, b = loader[(train_w, None)], loader[(train_w, 1)]
        src = f'로더 묶음(작업자 {train_w}개) {a:.3f} → {b:.3f}초'
    else:
        a = b = None
    if a is not None:
        no += 1
        print(f'{no}) 작업자당 OpenCV 1스레드: {src} ({a / b:.2f}배)')
        if b < a * 0.9:
            print('   → 빨라집니다. config.TRAIN_CV2_THREADS = 1 (또는 환경변수 TRAIN_CV2_THREADS=1) — 화소는 같습니다.')
        else:
            print('   → 차이가 10% 미만입니다. 지금처럼 둡니다(작업자끼리 CPU 다툼이 원인이 아님).')
        if loader:
            best = min(loader, key=loader.get)
            if best[0] != train_w and (train_w, best[1]) in loader and loader[best] < loader[(train_w, best[1])] * 0.85:
                print(f'   · 작업자 {train_w} → {best[0]}개({_tname(best[1])})면 로더 묶음 '
                      f'{loader[(train_w, best[1])]:.3f} → {loader[best]:.3f}초 (train.py _make_loader)')

    # 3) gradient checkpointing 범위
    if ckpt:
        no += 1
        res, d, total = ckpt['res'], ckpt['default'], ckpt['total']
        base = res.get(d)
        print(f'{no}) gradient checkpointing 범위')
        if not base or base['oom']:
            print('   지금 범위도 메모리가 모자랍니다 — 묶음 크기를 확인하세요.')
        else:
            # 학습 중 다른 것(검증 로더·조각난 메모리)도 있으니 GPU 전체의 80% 안만 안전하다고 본다
            limit = total * 0.8 if total == total else float('inf')
            ok = {k: r for k, r in res.items() if not r['oom'] and (r['resv'] != r['resv'] or r['resv'] <= limit)}
            ok[d] = base                                   # 지금 범위는 이미 쓰고 있으니 후보에 둔다
            best = min(ok, key=lambda k: ok[k]['t'])
            gain = 1 - ok[best]['t'] / base['t']
            if best != d and gain >= 0.05:
                print(f'   → GRAD_CHECKPOINT_BLOCKS={best} ({_ckpt_name(best, d).split(": ")[1]}): 학습 GPU 계산 '
                      f'{base["t"]:.3f} → {ok[best]["t"]:.3f}초({gain * 100:.0f}% 단축), 최대 메모리 '
                      f'{ok[best]["resv"]:.1f}GB / {total:.1f}GB.')
                print('     학습 명령 앞에 환경변수로: GRAD_CHECKPOINT_BLOCKS=%d python -u -m model.main --mode train ...' % best)
                print('     기울기는 같고 BN 이동평균도 맞춰 결과가 같습니다. ⚠️ span 학습(--a-mode span)은 한 단계에 창이')
                print('     최대 2배라 메모리가 더 듭니다 — span 은 지금 범위로 두거나 따로 잽니다.')
            else:
                print('   → 지금 범위(줄기~res4)대로 둡니다(더 빠른 안전한 범위가 없거나 5% 미만 차이)'
                      + (f'. 안전 기준: 예약 메모리 {limit:.1f}GB 이하' if limit != float('inf') else ''))

    # 4) torch.compile
    if comp:
        no += 1
        print(f'{no}) torch.compile')
        if 'error' in comp:
            print(f'   → 이 환경에서 쓸 수 없습니다({comp["error"][:80]}). 지금처럼 둡니다.')
        else:
            g_tr, g_val = comp['e_tr'] / comp['k_tr'], comp['e_val'] / comp['k_val']
            print(f'   학습 {g_tr:.2f}배 · 검증 {g_val:.2f}배 · 컴파일 {comp["c_sec"]:.0f}초 · P(A) 차이 {comp["diff"]:.1e}')
            if g_tr >= 1.1 and comp['diff'] < 1e-2:
                print('   → 효과가 있습니다. 학습 코드에 켜는 설정을 추가할 만합니다(아직 없음 — 결과는 반올림 수준만 다름).')
            else:
                print('   → 효과가 10% 미만이거나 결과 차이가 큽니다. 지금처럼 둡니다.')

    # 5) 검증 클립 저장본
    if val:
        no += 1
        n, c = val['none'], val['cached']
        print(f'{no}) 검증 클립 저장본(config.TRAIN_VAL_CACHE = {bool(getattr(config, "TRAIN_VAL_CACHE", False))})')
        print(f'   검증 묶음 하나 {n:.3f} → {c:.3f}초 ({n / max(c, 1e-9):.1f}배) · 처음 한 번 {val["first"]:.3f}초 · '
              f'결과 같음 {"예" if val["same"] else "아니오 ⚠️"}')
        print(f'   검증 클립 1,000개당 약 {n * 1000 / args.batch / 60:.1f}분 → {c * 1000 / args.batch / 60:.1f}분(에포크마다)')


if __name__ == '__main__':
    sys.exit(main())
