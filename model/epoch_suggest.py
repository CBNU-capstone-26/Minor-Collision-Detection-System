"""학습 곡선 CSV 로 '평가 영상으로 비교해 볼 에포크'를 고른다.

판단 지표(검증 균형 손실의 3에포크 평균)는 실제 검증 충돌이 적어(약 12개) 흔들리고, 평균이라
저장이 약 1에포크 늦게 따라간다. 또 손실 기준은 판정(P(A) 0.5)이 바뀌지 않아도 확률만 오르면
좋아져서, 재현율은 그대로인데 오탐이 는 에포크를 고르기도 한다(2026-10-22 학습: 판단 지표
1위 44에포크 7/12·오탐 6 vs 38에포크 7/12·오탐 2). 그래서 세 기준의 상위를 합쳐 후보로 낸다.
  ① 판단 지표(3에포크 평균) ② 그 에포크 균형손실 ③ 그 에포크 F1(서비스와 같은 0.5 판정의 재현율·정밀도)
구간 재현율(접근~충돌 종료 창 중 하나라도 A)·구분력은 2026-10-29 부터 학습 CSV 에 있고, 보여 주기만 하며 고르는 기준엔
넣지 않는다(그 전 학습은 '-' — python -m model.val_recheck 로 다시 계산).
기준마다 상위 k개(config.TRAIN_SUGGEST_TOPK)를 서로 TRAIN_TOPK_MIN_GAP 에포크 이상 떨어진 것만
고른다(붙은 에포크는 사실상 같은 모델). 검증 충돌이 적어 어느 기준도 흔들리므로 이것은 '후보'일 뿐이고,
최종 선택은 평가 영상(서비스 판정·정상 영상 오탐)으로 한다. 에포크별 가중치는 회차 폴더 epochs/ 에 있다.

    python -m model.main --mode suggest --run outputs/trainlogs/<회차 폴더>   # 학습 중·중단 후에도
"""
import csv
from pathlib import Path

import config


def _f(row, col):
    try:
        return float(row[col])
    except (KeyError, TypeError, ValueError):
        return None


def _f1(row, dom):
    """그 에포크 검증의 F1. 충돌로 판정한 클립이 없으면(정밀도 빈칸) 0."""
    r, p = _f(row, f'val_{dom}_recallA'), _f(row, f'val_{dom}_precA')
    if not r or not p:
        return 0.0
    return 2 * p * r / (p + r)


def _top(rows, key, k, gap):
    """key 가 작은 순으로, 이미 고른 에포크와 gap 이상 떨어진 것만 k 개."""
    picked = []
    for r in sorted((r for r in rows if key(r) is not None), key=key):
        e = int(r['epoch'])
        if all(abs(e - q) >= gap for q in picked):
            picked.append(e)
        if len(picked) == k:
            break
    return picked


def suggest(run_dir, k=None, gap=None):
    """(추천 에포크 목록, 출력 줄들). run_dir 은 회차 폴더(또는 그 안의 trainlog CSV)."""
    k = int(k or getattr(config, 'TRAIN_SUGGEST_TOPK', 2))
    gap = int(gap or getattr(config, 'TRAIN_TOPK_MIN_GAP', 3))
    run_dir = Path(run_dir)
    csv_path = run_dir if run_dir.suffix == '.csv' else next(iter(sorted(run_dir.glob('trainlog_*.csv'))), None)
    if csv_path is None or not csv_path.exists():
        return [], [f"학습 곡선 CSV 가 없습니다: {run_dir}"]
    run_dir = csv_path.parent
    with open(csv_path, encoding='utf-8') as fh:
        rows = [r for r in csv.DictReader(fh) if r.get('epoch')]
    if not rows:
        return [], ["학습 곡선이 아직 비어 있습니다"]
    # 실제 영상 검증이 있으면 그쪽, 없으면 RC 검증
    dom = 'real' if any(_f(r, 'val_real_bal') is not None for r in rows) else 'rc'
    bal = f'val_{dom}_bal'
    picks = {
        '판단지표': _top(rows, lambda r: _f(r, 'monitor'), k, gap),
        '균형손실': _top(rows, lambda r: _f(r, bal), k, gap),
        # F1 이 같으면 균형손실이 낮은 쪽
        'F1': _top(rows, lambda r: (-_f1(r, dom), _f(r, bal) if _f(r, bal) is not None else 9.0), k, gap),
    }
    pick = sorted({e for v in picks.values() for e in v})
    by_ep = {int(r['epoch']): r for r in rows}
    ep_dir = run_dir / 'epochs'

    def _file(e):
        hit = sorted(ep_dir.glob(f'*_ep{e:03d}.pth')) if ep_dir.exists() else []
        return hit[0] if hit else None

    def _n(v):
        return '-' if v is None else f'{v:.2f}'

    lines = [f"[평가 추천] 기준마다 상위 {k}개(서로 {gap}에포크 이상 떨어진 것) — ① 판단 지표 ② 그 에포크 {dom} "
             f"균형손실 ③ 그 에포크 {dom} F1. 후보일 뿐이고 최종 선택은 평가 영상으로",
             f"  {'에포크':>6}{'판단지표':>10}{dom + ' 균형':>10}{dom + ' 재현율':>11}{dom + ' 정밀도':>11}"
             f"{dom + ' F1':>8}{'구간 재현율':>9}{'구분력':>7}{'rc 균형':>9}  {'뽑힌 기준':<16}가중치"]
    files = []
    for e in pick:
        r = by_ep[e]
        fp = _file(e)
        files.append(fp)
        why = '·'.join(name for name, eps in picks.items() if e in eps)
        lines.append(f"  {e:>6}{_n(_f(r, 'monitor')):>10}{_n(_f(r, bal)):>10}"
                     f"{_n(_f(r, f'val_{dom}_recallA')):>11}{_n(_f(r, f'val_{dom}_precA')):>11}"
                     f"{_n(_f1(r, dom)):>8}{_n(_f(r, f'val_{dom}_spanRecA')):>9}{_n(_f(r, f'val_{dom}_auc')):>7}"
                     f"{_n(_f(r, 'val_rc_bal')):>9}  {why:<16}"
                     f"{fp.name if fp else '(없음 — 에포크별 저장 전 학습)'}")
    have = [p for p in files if p]
    if have:
        lines.append("  평가 명령(하나씩 차례로):")
        lines.append("    for w in " + " ".join(str(p) for p in have) + "; do")
        lines.append("      python -u -m model.main --mode eval --weights \"$w\"; done")
    return pick, lines
