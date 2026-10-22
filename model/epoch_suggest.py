"""학습 곡선 CSV 로 '평가 영상으로 비교해 볼 에포크'를 고른다.

판단 지표(검증 균형 손실의 3에포크 평균)는 실제 검증 충돌이 적어(약 12개) 흔들리고, 평균이라
저장이 약 1에포크 늦게 따라간다. 그래서 두 기준의 상위를 합쳐 후보로 내고, 최종 선택은
평가 영상(서비스 판정·오탐)으로 한다. 에포크별 가중치는 회차 폴더의 epochs/ 에 있다
(config.TRAIN_SAVE_ALL_EPOCHS, 2026-10-22 부터).

    python -m model.main --mode suggest --run outputs/trainlogs/<회차 폴더>   # 학습 중·중단 후에도
"""
import csv
from pathlib import Path


def _f(row, col):
    try:
        return float(row[col])
    except (KeyError, TypeError, ValueError):
        return None


def suggest(run_dir, k=3):
    """(추천 에포크 목록, 출력 줄들). run_dir 은 회차 폴더(또는 그 안의 trainlog CSV)."""
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
    by_mon = sorted((r for r in rows if _f(r, 'monitor') is not None), key=lambda r: _f(r, 'monitor'))[:k]
    by_bal = sorted((r for r in rows if _f(r, bal) is not None), key=lambda r: _f(r, bal))[:k]
    pick = sorted({int(r['epoch']) for r in by_mon + by_bal})
    by_ep = {int(r['epoch']): r for r in rows}
    ep_dir = run_dir / 'epochs'

    def _file(e):
        hit = sorted(ep_dir.glob(f'*_ep{e:03d}.pth')) if ep_dir.exists() else []
        return hit[0] if hit else None

    def _pct(v):
        return '-' if v is None else f'{v:.2f}'

    lines = [f"[평가 추천] 판단 지표 상위 {k} + 그 에포크 {dom} 균형손실 상위 {k} — 평가 영상으로 비교해 고른다",
             f"  {'에포크':>6}{'판단지표':>10}{dom + ' 균형':>10}{dom + ' 재현율':>11}{dom + ' 정밀도':>11}"
             f"{'rc 균형':>9}  가중치"]
    files = []
    for e in pick:
        r = by_ep[e]
        fp = _file(e)
        files.append(fp)
        lines.append(f"  {e:>6}{_pct(_f(r, 'monitor')):>10}{_pct(_f(r, bal)):>10}"
                     f"{_pct(_f(r, f'val_{dom}_recallA')):>11}{_pct(_f(r, f'val_{dom}_precA')):>11}"
                     f"{_pct(_f(r, 'val_rc_bal')):>9}  {fp.name if fp else '(없음 — 에포크별 저장 전 학습)'}")
    have = [p for p in files if p]
    if have:
        lines.append("  평가 명령(하나씩 차례로):")
        lines.append("    for w in " + " ".join(str(p) for p in have) + "; do")
        lines.append("      python -u -m model.main --mode eval --weights \"$w\"; done")
    return pick, lines
