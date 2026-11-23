"""학습 곡선 CSV로 평가해 볼 후보 에포크를 고릅니다.

세 기준의 상위를 합쳐 후보로 냅니다.
    1) 판단 지표(검증 균형 손실의 3에포크 평균)
    2) 그 에포크의 균형 손실
    3) 그 에포크의 F1(서비스와 같은 판정 기준의 재현율과 정밀도)
손실 기준은 판정이 바뀌지 않아도 확률만 오르면 좋아지므로 F1을 함께 봅니다.
구간 재현율과 구분력은 보여 주기만 하고 고르는 기준에는 넣지 않습니다.
기준마다 상위 k개(config.TRAIN_SUGGEST_TOPK)를 서로 TRAIN_TOPK_MIN_GAP 에포크 이상 떨어진 것만 고릅니다.

이 기준들은 검증 충돌이 적어 후보끼리의 실제 순서를 잘 맞히지 못합니다.
그래서 마지막에 검증 영상 서비스 판정으로 순위를 매기는 epoch_rank 명령도 함께 안내합니다.

    python -m model.main --mode suggest --run outputs/trainlogs/<회차 폴더>   # 학습 중이나 중단한 뒤에도 쓸 수 있습니다
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
    """그 에포크 검증의 F1을 돌려줍니다. 충돌로 판정한 클립이 없으면(정밀도 빈칸) 0입니다."""
    r, p = _f(row, f'val_{dom}_recallA'), _f(row, f'val_{dom}_precA')
    if not r or not p:
        return 0.0
    return 2 * p * r / (p + r)


def _top(rows, key, k, gap):
    """key가 작은 순으로, 이미 고른 에포크와 gap 이상 떨어진 것만 k개 고릅니다."""
    picked = []
    for r in sorted((r for r in rows if key(r) is not None), key=key):
        e = int(r['epoch'])
        if all(abs(e - q) >= gap for q in picked):
            picked.append(e)
        if len(picked) == k:
            break
    return picked


def _epoch_file(ep_dir, e):
    hit = sorted(ep_dir.glob(f'*_ep{e:03d}.pth')) if ep_dir.exists() else []
    return hit[0] if hit else None


def candidate_paths(run_dir):
    """추천 에포크 가중치의 절대 경로(파일이 있는 것만)를 돌려줍니다. main --mode suggest --paths가 한 줄에 하나씩 출력합니다.
    이어서 돌리는 명령이 출력 문장에서 경로를 뽑지 않아도 되게 따로 둡니다.
    """
    pick, _ = suggest(run_dir)
    run_dir = Path(run_dir)
    ep_dir = (run_dir.parent if run_dir.suffix == '.csv' else run_dir) / 'epochs'
    return [p.resolve() for p in (_epoch_file(ep_dir, e) for e in pick) if p]


def suggest(run_dir, k=None, gap=None):
    """(추천 에포크 목록, 출력 줄들)을 돌려줍니다. run_dir은 회차 폴더(또는 그 안의 trainlog CSV)입니다."""
    k = int(k or getattr(config, 'TRAIN_SUGGEST_TOPK', 2))
    gap = int(gap or getattr(config, 'TRAIN_TOPK_MIN_GAP', 3))
    run_dir = Path(run_dir)
    csv_path = run_dir if run_dir.suffix == '.csv' else next(iter(sorted(run_dir.glob('trainlog_*.csv'))), None)
    if csv_path is None or not csv_path.exists():
        return [], [f"학습 곡선 CSV 가 없습니다: {run_dir}"]
    run_dir = csv_path.parent
    with open(csv_path, encoding='utf-8') as fh:
        # 유예 기간(grace=1) 에포크는 추천하지 않습니다. best와 조기 종료도 세지 않는 구간입니다.
        rows = [r for r in csv.DictReader(fh) if r.get('epoch') and r.get('grace') != '1']
    if not rows:
        return [], ["학습 곡선이 아직 비어 있습니다"]
    # 실제 영상 검증이 있으면 그것을, 없으면 RC 검증을 씁니다.
    dom = 'real' if any(_f(r, 'val_real_bal') is not None for r in rows) else 'rc'
    bal = f'val_{dom}_bal'
    picks = {
        '판단지표': _top(rows, lambda r: _f(r, 'monitor'), k, gap),
        '균형손실': _top(rows, lambda r: _f(r, bal), k, gap),
        # F1이 같으면 균형 손실이 낮은 쪽
        'F1': _top(rows, lambda r: (-_f1(r, dom), _f(r, bal) if _f(r, bal) is not None else 9.0), k, gap),
    }
    pick = sorted({e for v in picks.values() for e in v})
    by_ep = {int(r['epoch']): r for r in rows}
    ep_dir = run_dir / 'epochs'

    def _file(e):
        return _epoch_file(ep_dir, e)

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
        # 224가 아닌 입력 크기로 학습한 회차(_s256 등)는 평가도 같은 크기로 해야 합니다. 다르면 가중치를 읽을 때 멈춥니다.
        from hitandrun_model import weights_head_pool, weights_input_size
        size = weights_input_size(have[0].name)
        pool = weights_head_pool(have[0].name)      # avg가 아닌 헤드 풀링(_top3 등)도 같은 값으로 맞춥니다.
        env = (f"INPUT_SIZE={size} " if size else "") + (f"HEAD_TEMPORAL_POOL={pool} " if pool != 'avg' else "")
        # 화면 로그도 남깁니다. tee는 폴더가 없으면 조용히 실패하므로 먼저 만듭니다(프로젝트 루트에서 실행).
        lines.append("  평가 명령(하나씩 차례로 · 화면 로그는 outputs/console/):")
        lines.append("    mkdir -p outputs/console && for w in \\")
        lines += [f"      {p} \\" for p in have[:-1]] + [f"      {have[-1]}; do"]
        lines.append(f"      {env}python -u -m model.main --mode eval --weights \"$w\"; done 2>&1 "
                     f"| tee outputs/console/eval_{run_dir.name}_$(date +%y%m%d_%H%M).log")
        # 위 기준들은 후보끼리의 실제 평가 순서를 잘 맞히지 못하므로,
        # 검증 영상 서비스 판정으로 순위를 다시 매긴 뒤 상위 1~2개만 평가하는 쪽을 권합니다.
        lines.append("  권장: 검증 영상 서비스 판정으로 순위(후반 절반 에포크를 재고 상위 2개 평가 명령을 냄):")
        lines.append(f"    {env}python -u -m model.epoch_rank --run {run_dir} 2>&1 "
                     f"| tee outputs/console/rank_{run_dir.name}_$(date +%y%m%d_%H%M).log")
    return pick, lines
