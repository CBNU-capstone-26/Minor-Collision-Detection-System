"""학습 회차의 에포크를 '검증 영상 서비스 판정'으로 순위 매긴다(2026-11-22).

suggest 의 학습 중 검증 지표(판단 지표·균형손실·F1·구간 재현율·구분력)는 후보끼리의 실제 평가 순서를 무작위 수준으로만
맞혔다(outputs/analysis/suggest_audit_261122 — 17묶음, 기준 1등 = 평가 1등 4~9개 vs 무작위 기댓값 5.1개). 검증 사고가
첫 묶음 11개뿐이고, 서비스 판정(간격 1 · 연속 묶음 · 정상 영상에 한 번만 튀어도 오탐)과 계산이 달라서다.
그래서 학습이 끝난 뒤 후보 에포크마다 검증 실제 영상의 간격 1 P(A)를 재고 서비스와 같은 이벤트 판정
(postproc_check._events = 지금 설정의 판정 기준·거친 탐색·최소 길이·fps 맞춤)으로 센다.
  · 후보: 기본은 학습한 에포크의 후반 절반(마지막 에포크 번호가 30 이면 16~30 — 조기 종료를 기다린 10에포크 포함, 유예 기간 제외).
    지난 17회차에서 평가 1등 에포크가 후반 절반 안에 15회, 판단 지표 상위 10개 안에 13회 있었다(11-22 사용자 결정).
    --top N 이면 판단 지표(3에포크 평균) 상위 N개, --epochs 로 직접 고를 수도.
  · 순위: 놓친 사고 적음 → 오탐 정상 영상 적음 → 오탐 이벤트 적음 → 가장 약하게 잡은 사고의 여유 큼 → 가장 센 오탐의 여유 큼.
    여유 = 판정 기준을 0.05 단위로 바꿔 가며 그 사고를 잡는 가장 높은 기준 / 그 정상에 이벤트가 생기는 가장 높은 기준.
  · 검증 영상 크롭은 한 번만 읽어 모든 에포크에 다시 쓴다(검증 실제 19개 약 0.74GB).
  · 검증 영상도 19개뿐이라 이 순위도 흔들린다. 상위 1~2개만 평가 영상으로 확인한다(평가 영상으로 여러 개를 고르면 점수가 부풀려짐).
결과: outputs/epoch_rank/<시각>_<회차>/report.txt · probs/(postproc_check 와 같은 형식)

    python -u -m model.epoch_rank --run outputs/trainlogs/<회차>               # 후반 절반, GPU 권장(서버 RTX 5090 후보당 약 20초 예상)
    python -u -m model.epoch_rank --run ... --top 10 | --epochs 30-45,48 | --videos 4(코드 점검)
    python -u -m model.epoch_rank --again outputs/epoch_rank/<폴더>            # 저장한 P(A)로 다시 집계(모델 안 씀)
"""
import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import config
from evaluate import _Tee
from postproc_check import _events, _load_probs, _read_crops, _score, _short, _val_videos, _window_probs, _wtag

GRID = [round(x, 2) for x in np.arange(0.05, 0.951, 0.05)]
THS = (0.35, 0.4, 0.45, 0.5, 0.55)


def _trainlog(run_dir):
    p = next(iter(sorted(Path(run_dir).glob('trainlog_*.csv'))), None)
    if p is None:
        return {}
    with open(p, encoding='utf-8') as fh:
        return {int(r['epoch']): r for r in csv.DictReader(fh) if r.get('epoch')}


def _parse_epochs(spec):
    """'30-45,48' → [30, …, 45, 48]."""
    out = set()
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-', 1)
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return sorted(out)


def _pick(rows, top):
    """top 이 없으면 후반 절반(마지막 에포크 번호 // 2 개), 있으면 판단 지표(3에포크 평균) 상위 top 개.
    둘 다 유예 기간(grace=1) 에포크는 뺀다."""
    cand = [r for r in rows.values() if r.get('grace') != '1']
    if not top:
        last = max(rows)
        start = last - last // 2 + 1
        return sorted(int(r['epoch']) for r in cand if int(r['epoch']) >= start)
    cand = [r for r in cand if r.get('monitor') not in (None, '')]
    cand.sort(key=lambda r: float(r['monitor']))
    return sorted(int(r['epoch']) for r in cand[:top])


def _epoch_file(run_dir, e):
    hit = sorted((Path(run_dir) / 'epochs').glob(f'*_ep{e:03d}.pth'))
    return hit[0] if hit else None


def _margins(data):
    """영상별 여유 — 사고: 잡히는 가장 높은 기준(못 잡으면 0) · 정상: 이벤트가 생기는 가장 높은 기준(없으면 0)."""
    catch, fp = {}, {}
    for name, (pa, m) in data.items():
        best = 0.0
        for t in GRID:
            evs, _ = _events(pa, m, t)
            hit = (any(s <= m['gt_ef'] and e >= m['gt_sf'] for s, e in evs) if m['gt_sf'] is not None else bool(evs))
            if hit:
                best = t
        (catch if m['gt_sf'] is not None else fp)[name] = best
    return catch, fp


def _summarize(ep, data, t, tr, w):
    """tr = 그 에포크의 학습 곡선 한 줄, w = 가중치 경로."""
    sc, _ = _score(data, t)
    catch, fp = _margins(data)
    weak = min(catch.values()) if catch else 0.0
    strong = max(fp.values()) if fp else 0.0
    by_t = {x: _score(data, x)[0] for x in THS}
    return dict(ep=ep, sc=sc, weak=weak, weak_v=min(catch, key=catch.get) if catch else '-',
                strong=strong, strong_v=max(fp, key=fp.get) if fp and strong > 0 else '-', by_t=by_t,
                tr=tr or {}, w=str(w) if w else None, run=_run_id(w),
                key=(sc['miss'], sc['fa'], sc['false_ev'], -weak, strong, ep))


def _run_id(w):
    m = re.search(r'(\d{6}_\d{4})', Path(str(w)).name) if w else None
    return m.group(1) if m else ''


def _report(results, t, n_val):
    results.sort(key=lambda r: r['key'])
    multi = len({r['run'] for r in results}) > 1          # 다시 집계할 때 여러 회차가 섞였으면 회차 시각도 붙인다
    lines = [f"\n[순위] 검증 실제 {n_val}개 · 서비스 판정 기준 {t:g} — 놓침 → 오탐 영상 → 오탐 이벤트 → 약한 사고 여유 → 센 오탐 여유",
             f"  {'순위':>4}  {'에포크':<8}{'놓침':>3}{'오탐영상':>7}{'오탐이벤트':>8}{'약한사고 여유':>16}{'센 오탐 여유':>16}"
             f"   기준별 놓침/오탐영상 {' '.join(f'{x:g}' for x in THS)}   │ 학습 중: 판단지표 · 균형손실 · F1"]
    for i, r in enumerate(results, 1):
        sc = r['sc']
        tr = r['tr']
        name = f"{r['run'][-4:]}·{r['ep']}" if multi else str(r['ep'])
        rc, pc = tr.get('val_real_recallA'), tr.get('val_real_precA')
        try:
            f1 = 2 * float(rc) * float(pc) / (float(rc) + float(pc))
        except (TypeError, ValueError, ZeroDivisionError):
            f1 = float('nan')
        bt = ' '.join(f"{r['by_t'][x]['miss']}/{r['by_t'][x]['fa']}" for x in THS)
        lines.append(f"  {i:>4}  {name:<9}{sc['miss']:>3}{sc['fa']:>7}{sc['false_ev']:>8}"
                     f"{r['weak']:>8.2f}({_short(r['weak_v'])[:8]:>8}){r['strong']:>6.2f}({_short(r['strong_v'])[:8]:>8})"
                     f"   {bt:<28}│ {tr.get('monitor', '-')[:6]} · {tr.get('val_real_bal', '-')[:6]} · {f1:.2f}")
    best = results[0]
    lines.append(f"\n  1위 {'' if not multi else best['run'] + ' '}ep{best['ep']} — 놓침 {', '.join(_short(x) for x in best['sc']['miss_list']) or '없음'} · "
                 f"오탐 {', '.join(_short(x) for x in best['sc']['fa_list']) or '없음'}")
    have = [r['w'] for r in results[:2] if r['w']]
    if have:
        from hitandrun_model import weights_head_pool, weights_input_size
        size = weights_input_size(Path(have[0]).name)
        pool = weights_head_pool(Path(have[0]).name)
        env = (f"INPUT_SIZE={size} " if size else "") + (f"HEAD_TEMPORAL_POOL={pool} " if pool != 'avg' else "")
        lines.append("  평가 명령(상위 2개만 — 평가 영상으로 여러 개를 고르면 점수가 부풀려짐):")
        lines.append("    mkdir -p outputs/console && for w in \\")
        lines += [f"      {p} \\" for p in have[:-1]] + [f"      {have[-1]}; do"]
        lines.append(f"      {env}python -u -m model.main --mode eval --weights \"$w\"; done 2>&1 "
                     f"| tee outputs/console/eval_rank_$(date +%y%m%d_%H%M).log")
    return lines


def _again(folder, t):
    folder = Path(folder)
    results, logs, n_val = [], {}, 0
    for wdir in sorted((folder / 'probs').iterdir()):
        m = re.search(r'_ep(\d+)', wdir.name)
        if not m:
            continue
        data = _load_probs(wdir, 'val', 'none')
        if not data:
            continue
        ep = int(m.group(1))
        n_val = len(data)
        w = next(iter(data.values()))[1].get('weights')
        rd = Path(w).resolve().parent.parent if w else None
        if rd not in logs:
            logs[rd] = _trainlog(rd) if rd else {}
        results.append(_summarize(ep, data, t, logs[rd].get(ep), w))
    return results, n_val


def main():
    ap = argparse.ArgumentParser(description='에포크를 검증 영상 서비스 판정으로 순위 매김')
    ap.add_argument('--run', help='학습 회차 폴더(outputs/trainlogs/<회차>)')
    ap.add_argument('--top', type=int, default=0, help='판단 지표 상위 몇 개만 잴지(기본 0 = 후반 절반, 유예 기간 제외)')
    ap.add_argument('--epochs', help='잴 에포크를 직접(예: 30-45,48) — 주면 --top 무시')
    ap.add_argument('--threshold', type=float, default=None, help='서비스 판정 기준(기본 config.A_THRESHOLD)')
    ap.add_argument('--batch', type=int, default=0, help='추론 묶음(0 = 서비스와 같게 GPU 8 · CPU 1)')
    ap.add_argument('--videos', type=int, default=0, help='사고·정상 절반씩 이 수만(0 = 전부, 코드 점검용)')
    ap.add_argument('--again', help='이전 결과 폴더 — 저장한 P(A)로 다시 집계(모델 안 씀)')
    args = ap.parse_args()
    if not args.run and not args.again:
        ap.error('--run 또는 --again 이 필요합니다')
    t = config.A_THRESHOLD if args.threshold is None else args.threshold

    if args.again:
        results, n_val = _again(args.again, t)
        out_dir = Path(args.again)
        tee = _Tee(out_dir / f"report_again_{datetime.now():%y%m%d_%H%M%S}.txt")
        sys.stdout = tee
        try:
            print(f"[에포크 순위 — 다시 집계] {out_dir} · 기준 {t:g}")
            print('\n'.join(_report(results, t, n_val)))
        finally:
            sys.stdout = tee.stdout
            tee.close()
        return 0

    run_dir = Path(args.run)
    rows = _trainlog(run_dir)
    if not rows:
        ap.error(f'학습 곡선 CSV 가 없습니다: {run_dir}')
    eps = _parse_epochs(args.epochs) if args.epochs else _pick(rows, args.top)
    weights_of = {e: _epoch_file(run_dir, e) for e in eps}
    missing = [e for e in eps if weights_of[e] is None]
    eps = [e for e in eps if weights_of[e] is not None]
    if not eps:
        ap.error(f'잴 에포크 가중치가 없습니다(epochs/ 확인): {run_dir}')
    split = run_dir / 'split.json'
    vids = _val_videos(split)
    if args.videos:
        vids = ([v for v in vids if v[2] is not None][:(args.videos + 1) // 2]
                + [v for v in vids if v[2] is None][:args.videos // 2])

    out_dir = Path(config._ROOT) / 'outputs' / 'epoch_rank' / f"{datetime.now():%y%m%d_%H%M%S}_{run_dir.name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tee = _Tee(out_dir / 'report.txt')
    sys.stdout = tee
    done = False
    try:
        print(f"[에포크 순위] {datetime.now():%Y-%m-%d %H:%M} · 회차 {run_dir.name} · 판정 기준 {t:g}")
        print(f"후보 에포크 {len(eps)}개: {', '.join(map(str, eps))}"
              + (f" (가중치 없음: {', '.join(map(str, missing))})" if missing else '')
              + (' — 직접 고름' if args.epochs else f' — 판단 지표 상위 {args.top}개(유예 기간 제외)' if args.top
                 else f' — 후반 절반(마지막 ep{max(rows)}, 유예 기간 제외)'))
        n_a = sum(v[2] is not None for v in vids)
        print(f"검증 실제 영상 {len(vids)}개(사고 {n_a} · 정상 {len(vids) - n_a}) — {split}")
        t0 = time.perf_counter()
        crops = []
        for mp4, bbox, sf, ef in vids:
            frames, fps, size = _read_crops(mp4, bbox)
            if frames:
                crops.append((mp4, bbox, sf, ef, frames, fps, size))
        nbytes = sum(len(c[4]) for c in crops) * int(np.prod(crops[0][4][0].shape)) if crops else 0
        print(f"검증 크롭 읽기 {len(crops)}개 · {time.perf_counter() - t0:.0f}초 · 약 {nbytes / 2 ** 30:.2f}GB", flush=True)

        from device_utils import is_cuda_like, predict_batch_size
        from infer_runtime import get_runner
        from main import _load_model
        results = []
        for ep in eps:
            t1 = time.perf_counter()
            w = weights_of[ep]
            model, device = _load_model(str(w))
            model.eval()
            if is_cuda_like(device):
                torch.backends.cudnn.benchmark = True      # 실행기를 고르기 전에 켠다(CLAUDE.md 참고)
            runner = get_runner(model)
            batch = args.batch or predict_batch_size(device)
            wdir = out_dir / 'probs' / _wtag(w) / 'val_none'
            wdir.mkdir(parents=True, exist_ok=True)
            data = {}
            for mp4, bbox, sf, ef, frames, fps, size in crops:
                pa = _window_probs(runner, device, frames, batch)
                meta = dict(file=mp4.name, set='val', resample='none', fps=fps, factor=1, n_orig=len(frames),
                            width=size[0], height=size[1], bbox=[int(v) for v in bbox], gt_sf=sf, gt_ef=ef,
                            weights=str(w))
                np.savez(wdir / f'{mp4.stem}.npz', pa=pa, meta=json.dumps(meta))
                data[mp4.stem] = (pa, meta)
            r = _summarize(ep, data, t, rows.get(ep), w)
            results.append(r)
            sc = r['sc']
            print(f"  ep{ep}: 놓침 {sc['miss']} · 오탐 영상 {sc['fa']} · 오탐 이벤트 {sc['false_ev']} · "
                  f"약한 사고 여유 {r['weak']:.2f} · 센 오탐 여유 {r['strong']:.2f} · {runner.name} · "
                  f"{time.perf_counter() - t1:.0f}초", flush=True)
            del model, runner
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print('\n'.join(_report(results, t, len(crops))))
        print(f"\n전체 {time.perf_counter() - t0:.0f}초")
        done = True
        return 0
    finally:
        sys.stdout = tee.stdout
        tee.close()
        print(f'\n[저장] {out_dir}' if done else f'\n[중단] 오류로 멈춤 — 일부만 남음: {out_dir}')


if __name__ == '__main__':
    sys.exit(main())
