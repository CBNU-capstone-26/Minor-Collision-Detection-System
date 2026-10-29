"""지난 학습의 에포크별 가중치로 검증 지표를 다시 계산한다 — 2026-10-29 에 추가한 지표를 그 전 학습에도 채운다.

학습 중 검증(train.validate)과 같은 함수·같은 검증 영상(회차 폴더 split.json)으로 잰다.
  · 기존 지표(균형 손실·재현율)도 다시 내서 학습 CSV 와 맞춰 본다(검산 — 맞으면 같은 검증을 재현한 것)
  · 첫 창 재현율 : 충돌 시작 TRAIN_A_PRE_FRAMES 전부터 창 1개('pre' 학습의 val 재현율과 같은 정의)
  · 구간 재현율 : 접근~충돌 종료 창(최대 TRAIN_SPAN_MAX_WINDOWS) 중 하나라도 A — 서비스 판정에 가깝다
  · 구분력      : 충돌 샘플(구간 창 중 최고 점수)이 비충돌 클립보다 P(A)가 높을 확률 — 판정 기준과 무관
학습 방식(pre/span)과 상관없이 정의가 같아, pre 학습과 span 학습을 이 지표로 서로 비교할 수 있다.
--threshold 로 판정 기준을 바꿔 다시 셀 수 있다(검증 데이터로 기준을 정할 때 — 평가 영상으로 정하지 않는다).

    python -m model.val_recheck --run outputs/trainlogs/<회차 폴더>                  # 에포크 전부(epochs/)
    python -m model.val_recheck --run outputs/trainlogs/<회차 폴더> --epochs 35,41,43 --domain real

결과: 회차 폴더의 val_recheck_<시각>[_th<기준>].csv · .txt. 에포크별 가중치가 없는 학습(10-22 이전)은
work_best·work_cand_ep* 만 잰다.
"""
import argparse
import csv
import importlib.util
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import Subset

import config
from dataset import HitAndRunDataset, span_collate
from device_utils import get_device, is_channels_last_3d_supported, is_cuda_like
from evaluate import _Tee
from hitandrun_model import HitAndRun3DCNN
from train import _make_loader, validate

# 검증 클립 구성에 영향을 주는 설정 — 학습 당시(config_snapshot.py)와 다르면 검산이 맞지 않을 수 있다
_SAMPLE_KEYS = ('CLIP_LENGTH', 'RESIZE', 'R_VALUE', 'NORM_MEAN', 'NORM_STD', 'TRAIN_A_PRE_FRAMES',
                'TRAIN_A_SLICE_ENABLED', 'TRAIN_A_SLICE_STRIDE', 'TRAIN_S_SLICE_ENABLED', 'TRAIN_S_SLICE_STRIDE',
                'TRAIN_S_MAX_CLIPS_PER_VIDEO', 'TRAIN_S_REAL_USE_CAR1', 'TRAIN_SPAN_WINDOW_STRIDE',
                'TRAIN_SPAN_MAX_WINDOWS', 'TRAIN_SPAN_FIRST_WEIGHT', 'TRAIN_CLASS_WEIGHTS')
_METRICS = ('bal', 'rec', 'prec', 'preRec', 'prePrec', 'spanRec', 'spanPrec', 'auc')


def _snapshot_diff(run_dir):
    """학습 당시 설정과 지금 설정이 다른 키 [(키, 당시, 지금)]. 스냅숏이 없으면 None."""
    snap = run_dir / 'config_snapshot.py'
    if not snap.exists():
        return None
    spec = importlib.util.spec_from_file_location('_run_config_snapshot', snap)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # noqa: BLE001 — 옛 스냅숏을 못 읽어도 다시 계산은 할 수 있다
        print(f'⚠️ 설정 스냅숏을 읽지 못해 비교를 건너뜁니다: {exc}')
        return None
    return [(k, getattr(mod, k, None), getattr(config, k, None)) for k in _SAMPLE_KEYS
            if getattr(mod, k, None) != getattr(config, k, None)]


def _weights(run_dir, epochs):
    """[(에포크 번호 또는 이름, 가중치 경로)]. epochs/ 가 없으면 work_best·work_cand_ep*."""
    found = []
    for p in sorted((run_dir / 'epochs').glob('*_ep[0-9][0-9][0-9].pth')):
        found.append((int(re.search(r'_ep(\d+)\.pth$', p.name).group(1)), p))
    if not found:
        for p in sorted(run_dir.glob('work_cand_ep*.pth')):
            found.append((int(re.search(r'_ep(\d+)\.pth$', p.name).group(1)), p))
        if (run_dir / 'work_best.pth').exists():
            found.append(('best', run_dir / 'work_best.pth'))
    if epochs:
        found = [(e, p) for e, p in found if e in epochs]
    return found


def _val_indices(dataset, split, data_dir):
    """split.json 의 검증 영상 → (rc 샘플 번호, real 샘플 번호, 못 찾은 검증 영상 수).

    split.json 은 train.py 가 데이터 폴더 기준 상대경로(os.path.relpath)로 남기므로 같은 방식으로 맞춘다 —
    학습한 컴퓨터와 데이터 폴더 위치가 달라도 된다.
    """
    def _key(p):
        return str(p).replace('\\', '/')     # 다른 OS(Windows)에서 학습한 회차도 읽게

    want = {_key(v) for v in split['val']}
    rc, real, seen = [], [], set()
    for i, s in enumerate(dataset.samples):
        key = _key(os.path.relpath(s.get('video_key', s['mp4_path']), data_dir))
        if key in want:
            seen.add(key)
            (real if 'realdata' in Path(s['mp4_path']).parts else rc).append(i)
    return rc, real, len(want - seen)


def _f(v):
    return '-' if v is None else f'{v:.3f}'


def main():
    ap = argparse.ArgumentParser(description='지난 학습의 검증 지표 다시 계산')
    ap.add_argument('--run', required=True, help='학습 회차 폴더(outputs/trainlogs/<회차>)')
    ap.add_argument('--epochs', default='', help='이 에포크만(쉼표로, 예: 35,41,43). 비우면 전부')
    ap.add_argument('--domain', choices=('all', 'real', 'rc'), default='all',
                    help='검증 영상 — real 만 보면 몇 배 빠르다')
    ap.add_argument('--threshold', type=float, default=None,
                    help=f'판정 기준 P(A)(기본 config.A_THRESHOLD = {config.A_THRESHOLD:g})')
    args = ap.parse_args()

    run_dir = Path(args.run).resolve()
    if not (run_dir / 'split.json').exists():
        print(f'에러: 회차 폴더에 split.json 이 없습니다 -> {run_dir}')
        return 1
    th = config.A_THRESHOLD if args.threshold is None else args.threshold
    stamp = datetime.now().strftime('%y%m%d_%H%M%S')
    th_tag = '' if abs(th - 0.5) < 1e-9 else f'_th{th:g}'
    out_base = run_dir / f'val_recheck_{stamp}{th_tag}'
    tee = _Tee(out_base.with_suffix('.txt'))
    sys.stdout = tee
    try:
        return _run(args, run_dir, th, out_base)
    finally:
        sys.stdout = tee.stdout
        tee.close()
        print(f'\n[저장] {out_base}.csv · .txt')


def _run(args, run_dir, th, out_base):
    span = '_span' in run_dir.name           # 학습 방식 — 기존 지표(손실·재현율)를 학습 때처럼 내려고
    epochs = {int(e) for e in args.epochs.split(',') if e.strip()} if args.epochs else None
    weights = _weights(run_dir, epochs)
    if not weights:
        print(f'에러: 잴 가중치가 없습니다(epochs/ · work_best.pth) -> {run_dir}')
        return 1
    split = json.loads((run_dir / 'split.json').read_text(encoding='utf-8'))
    data_dir = Path(split.get('data_dir') or config.DATA_DIR)
    if not data_dir.exists():
        data_dir = Path(config.DATA_DIR)

    print(f'[검증 다시 계산] {datetime.now():%Y-%m-%d %H:%M} · 회차 {run_dir.name} · '
          f'A 방식 {"span" if span else "pre"} · 판정 기준 P(A) {th:g} · 가중치 {len(weights)}개')
    diff = _snapshot_diff(run_dir)
    if diff:
        print('⚠️ 검증 클립에 영향을 주는 설정이 학습 당시와 다릅니다 — 검산이 맞지 않을 수 있습니다:')
        for k, old, new in diff:
            print(f'    {k}: 당시 {old!r} → 지금 {new!r}')

    device = get_device(config.TRAIN_DEVICE_TYPE)
    if is_cuda_like(device):
        torch.backends.cudnn.benchmark = True
    channels_last = bool(config.USE_CHANNELS_LAST) and is_channels_last_3d_supported(device)
    dataset = HitAndRunDataset(
        data_dir=data_dir, augment=False, span_windows=True,
        cache_dir=config.TRAIN_VAL_CACHE_DIR if getattr(config, 'TRAIN_VAL_CACHE', False) else None)
    rc_idx, real_idx, missing = _val_indices(dataset, split, data_dir)
    print(f'[검증 영상] {data_dir} — 회차의 검증 영상 {len(split["val"])}개 중 못 찾음 {missing}개 · '
          f'클립 rc {len(rc_idx)} / real {len(real_idx)}')
    if missing:
        print('⚠️ 없는 검증 영상이 있어 학습 때와 검증 세트가 다릅니다(검산이 맞지 않을 수 있음)')
    doms = [d for d in ('rc', 'real') if args.domain in ('all', d)]
    idx = {'rc': rc_idx, 'real': real_idx}
    loaders = {d: _make_loader(Subset(dataset, idx[d]), batch_size=config.TRAIN_BATCH_SIZE, shuffle=False,
                               device=device, collate_fn=span_collate) if idx[d] else None for d in doms}

    rows_csv = {}
    log = next(iter(sorted(run_dir.glob('trainlog_*.csv'))), None)
    if log is not None:
        with open(log, encoding='utf-8') as fh:
            rows_csv = {r['epoch']: r for r in csv.DictReader(fh) if r.get('epoch')}

    model = HitAndRun3DCNN(num_classes=config.MODEL_NUM_CLASSES).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last_3d)
    out_rows, n_check, n_ok = [], 0, 0
    t0 = time.perf_counter()
    for k, (ep, path) in enumerate(weights):
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        row = {'epoch': ep, 'weights': path.name}
        old = rows_csv.get(str(ep), {})
        notes = []
        for d in doms:
            m = validate(model, loaders[d], device, span=span, criterion=torch.nn.CrossEntropyLoss(),
                         batch_size=config.TRAIN_BATCH_SIZE, first_w=float(config.TRAIN_SPAN_FIRST_WEIGHT),
                         channels_last=channels_last, threshold=th) if loaders[d] else None
            for key in _METRICS:
                row[f'{d}_{key}'] = None if m is None else m[key]
            # 검산: 학습 CSV 의 균형 손실·재현율과 같은가(기준이 0.5 가 아니면 재현율은 빼고)
            if m is not None and old.get(f'val_{d}_bal'):
                ok = abs(float(old[f'val_{d}_bal']) - m['bal']) < 2e-3
                if abs(th - 0.5) < 1e-9 and old.get(f'val_{d}_recallA') and m['rec'] is not None:
                    ok = ok and abs(float(old[f'val_{d}_recallA']) - m['rec']) < 1e-6
                row[f'{d}_check'] = 'ok' if ok else 'diff'
                n_check += 1
                n_ok += ok
                if not ok:
                    notes.append(f"{d} 균형 기록 {float(old[f'val_{d}_bal']):.4f}/재계산 {m['bal']:.4f}")
        out_rows.append(row)
        el = time.perf_counter() - t0
        print(f'  [{k + 1}/{len(weights)}] ep {ep}: '
              + ' | '.join(f"{d} 첫창 {_f(row[f'{d}_preRec'])} 구간 {_f(row[f'{d}_spanRec'])} "
                           f"구간정밀 {_f(row[f'{d}_spanPrec'])} 구분력 {_f(row[f'{d}_auc'])}" for d in doms)
              + (f'  ⚠️ {"; ".join(notes)}' if notes else '')
              + f'  · {el / (k + 1):.0f}초/개, 남은 약 {el / (k + 1) * (len(weights) - k - 1) / 60:.1f}분')

    cols = ['epoch', 'weights'] + [f'{d}_{c}' for d in doms for c in (*_METRICS, 'check')]
    with open(f'{out_base}.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in out_rows:
            w.writerow(['' if r.get(c) is None else (f'{r[c]:.6f}' if isinstance(r[c], float) else r[c])
                        for c in cols])
    print(f'\n[검산] 학습 CSV 와 같음 {n_ok}/{n_check}'
          + ('' if n_check else ' (비교할 기록 없음)')
          + ('' if n_ok == n_check else ' — 다른 것은 검증 영상·설정·장치(CPU/GPU) 차이를 확인'))
    dom = 'real' if 'real' in doms and real_idx else doms[0]
    top = sorted((r for r in out_rows if r.get(f'{dom}_spanRec') is not None),
                 key=lambda r: (-r[f'{dom}_spanRec'], -(r[f'{dom}_auc'] or 0)))[:5]
    if top:
        print(f'[{dom} 구간 재현율 상위] ' + ', '.join(
            f"ep {r['epoch']}({_f(r[f'{dom}_spanRec'])}·구분력 {_f(r[f'{dom}_auc'])})" for r in top)
            + ' — 후보일 뿐이고 최종 선택은 평가 영상으로')
    return 0


if __name__ == '__main__':
    sys.exit(main())
