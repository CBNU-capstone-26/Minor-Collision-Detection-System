"""학습 곡선 CSV → PNG 그래프.

train.py 가 매 에포크 끝에 호출해 `outputs/trainlogs/<회차>/<로그이름>.png` 를 갱신한다
(학습이 중간에 끊겨도 그때까지의 곡선이 남는다). 지난 로그도 그릴 수 있다:

    python trainplot.py ../outputs/trainlogs/*/*.csv

세 칸으로 그린다 (x축 = 에포크, 칸마다 y축 하나):
  1) 손실 — train / val_rc / val_real, 그리고 best 판단 지표(있으면)
  2) 검증 충돌(A) recall(실선)·precision(점선) — val_real / val_rc
  3) 헤드 학습률(로그 스케일) — ReduceLROnPlateau 가 언제 깎았는지
best 에포크는 실선, 후보 에포크는 점선 세로줄로 표시한다.

그래프 안 글자는 한글로 쓴다. 한글 글꼴(맑은 고딕·애플 SD 고딕 Neo·나눔고딕·Noto Sans CJK)이
없는 컴퓨터(예: 글꼴을 깔지 않은 리눅스 서버)에선 글자가 네모로 깨지므로 영문으로 쓴다.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

# 색은 '대상'을 따른다 — 세 칸 모두 같은 대상은 같은 색.
# (검증된 기본 범주 팔레트 1~3번. 3번 청록은 배경 대비가 2.7:1 로 낮아 선 끝에
#  이름을 직접 붙여 보완한다.)
C_REAL = '#2a78d6'    # val_real — 서비스 대상(실차)이라 가장 먼저
C_RC = '#eb6834'      # val_rc
C_TRAIN = '#1baf7a'   # train
INK = '#0b0b0b'       # 글자·best 표시
INK2 = '#52514e'      # 보조 글자·후보 표시·학습률
GRID = '#e4e3df'
SURFACE = '#fcfcfb'

# 한글 글꼴 — 앞에서부터 찾는다(Windows → macOS → 리눅스)
KO_FONTS = ('Malgun Gothic', 'Apple SD Gothic Neo', 'AppleGothic',
            'NanumGothic', 'Noto Sans CJK KR', 'Noto Sans KR', 'Noto Sans CJK JP')
# WSL 은 Windows 글꼴이 자동 등록되지 않아 파일로 직접 넣는다
KO_FONT_FILES = ('/mnt/c/Windows/Fonts/malgun.ttf',)

# 그래프 안 글자 — 한글 글꼴이 있으면 KO, 없으면 EN
LABELS = {
    'ko': {
        'train': '학습 (증강, 드롭아웃 켬)', 'train_end': '학습',
        'val_rc': 'RC 검증', 'val_real': '실제 검증',
        'monitor': '판단 지표: 실제 검증 균형 손실 이동평균',
        'loss': '손실', 'recall': '{} 재현율', 'precision': '{} 정밀도',
        'collision': '충돌(A) 판정\n임계값 0.5', 'lr': '헤드 학습률', 'epoch': '에포크',
        'best': 'best {}에포크', 'cand': '후보 {}에포크', 'early_stop': '조기 종료',
        'why_metric': '판단 지표 최소',
        'why_old_real': '옛 기준: 실제 검증 손실 최소',
        'why_old_rc': '옛 기준: RC 검증 손실 최소',
    },
    'en': {
        'train': 'train (aug + dropout on)', 'train_end': 'train',
        'val_rc': 'val_rc', 'val_real': 'val_real',
        'monitor': 'selection metric: val_real class-balanced loss, moving avg',
        'loss': 'loss', 'recall': '{} recall', 'precision': '{} precision',
        'collision': 'collision (A)\n@ threshold 0.5', 'lr': 'head LR', 'epoch': 'epoch',
        'best': 'best ep{}', 'cand': 'cand ep{}', 'early_stop': 'early stop',
        'why_metric': 'min selection metric',
        'why_old_real': 'old rule: min val_real loss',
        'why_old_rc': 'old rule: min val_rc loss',
    },
}


def _korean_font():
    """쓸 수 있는 한글 글꼴 이름. 없으면 None."""
    from matplotlib import font_manager as fm
    names = {f.name for f in fm.fontManager.ttflist}
    for name in KO_FONTS:
        if name in names:
            return name
    for path in KO_FONT_FILES:
        if Path(path).exists():
            fm.fontManager.addfont(path)
            return fm.FontProperties(fname=path).get_name()
    return None


def _load(csv_path: Path):
    with open(csv_path, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    def col(key):
        out = []
        for r in rows:
            v = r.get(key)
            out.append(float(v) if v not in (None, '') else math.nan)
        return out
    return rows, col


def _pick_best(rows, col, result):
    """best 에포크: 학습 결과 파일(.result.json) → 판단 지표 열 → 옛 기준 순으로.
    (에포크, 근거 'metric'|'old_real'|'old_rc') 를 돌려준다."""
    if result.get('best_epoch'):
        return int(result['best_epoch']), 'metric'
    ep = [int(r['epoch']) for r in rows]
    for key, why in (('monitor', 'metric'),
                     ('val_real_loss', 'old_real'),
                     ('val_rc_loss', 'old_rc')):
        vals = col(key) if key in rows[0] else []
        pairs = [(v, e) for v, e in zip(vals, ep) if not math.isnan(v)]
        if pairs:
            return min(pairs)[1], why
    return None, ''


def _end_label(ax, xs, ys, text, color):
    """선 끝에 이름을 붙인다(범례와 함께 — 색만으로 구분하지 않게)."""
    pts = [(x, y) for x, y in zip(xs, ys) if not math.isnan(y)]
    if not pts:
        return
    x, y = pts[-1]
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords='offset points',
                va='center', fontsize=8.5, color=INK2)
    ax.plot([x], [y], 'o', ms=4, color=color, mec=SURFACE, mew=1.5, zorder=4)


def plot_trainlog(csv_path, out_path=None):
    """CSV 하나를 그려 PNG 경로를 돌려준다. 행이 없으면 None."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    csv_path = Path(csv_path)
    rows, col = _load(csv_path)
    if not rows:
        return None
    out_path = Path(out_path) if out_path else csv_path.with_suffix('.png')
    side = csv_path.with_name(csv_path.stem + '.result.json')
    result = json.loads(side.read_text(encoding='utf-8')) if side.exists() else {}

    ep = [int(r['epoch']) for r in rows]
    best, best_why = _pick_best(rows, col, result)
    cands = [int(c['epoch']) for c in result.get('candidates', [])
             if int(c['epoch']) != best]

    font = _korean_font()
    L = LABELS['ko' if font else 'en']
    # 한글 글꼴에 없는 글자(수식 등)는 DejaVu Sans 로 채운다. 마이너스는 글꼴마다 없을 수 있어 '-' 로.
    rc = {'font.family': [font, 'DejaVu Sans'], 'axes.unicode_minus': False} if font else {}
    with plt.rc_context(rc):
        fig = _draw(plt, L, csv_path, rows, col, ep, best, best_why, cands, result)
        fig.savefig(out_path, dpi=130, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)
    return out_path


def _draw(plt, L, csv_path, rows, col, ep, best, best_why, cands, result):
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(10, 9.2), sharex=True,
        gridspec_kw={'height_ratios': [3, 2, 1.1], 'hspace': 0.22})
    fig.patch.set_facecolor(SURFACE)

    # ── 1) 손실 ──
    ax1.plot(ep, col('train_loss'), color=C_TRAIN, lw=2, label=L['train'])
    ax1.plot(ep, col('val_rc_loss'), color=C_RC, lw=2, label=L['val_rc'])
    ax1.plot(ep, col('val_real_loss'), color=C_REAL, lw=2, label=L['val_real'])
    _end_label(ax1, ep, col('train_loss'), L['train_end'], C_TRAIN)
    _end_label(ax1, ep, col('val_rc_loss'), L['val_rc'], C_RC)
    _end_label(ax1, ep, col('val_real_loss'), L['val_real'], C_REAL)
    if 'monitor' in rows[0]:
        ax1.plot(ep, col('monitor'), color=C_REAL, lw=2.2, ls=(0, (5, 3)),
                 label=L['monitor'])
    ax1.set_ylabel(L['loss'], color=INK2)
    ax1.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=4, fontsize=8.5,
               frameon=False, labelcolor=INK2, handlelength=2.6, columnspacing=1.4)

    # ── 2) 검증 충돌 recall / precision (0.5 임계값) ──
    for key, color in (('val_real', C_REAL), ('val_rc', C_RC)):
        ax2.plot(ep, col(f'{key}_recallA'), color=color, lw=2,
                 label=L['recall'].format(L[key]))
        ax2.plot(ep, col(f'{key}_precA'), color=color, lw=1.6, ls=(0, (2, 2)),
                 label=L['precision'].format(L[key]))
    ax2.set_ylim(-0.03, 1.05)
    ax2.set_ylabel(L['collision'], color=INK2, linespacing=1.6)  # 한글 글꼴은 두 줄이 겹쳐서
    ax2.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=4, fontsize=8.5,
               frameon=False, labelcolor=INK2, handlelength=2.6, columnspacing=1.4)

    # ── 3) 학습률 ──
    ax3.step(ep, col('lr'), where='post', color=INK2, lw=1.8)
    ax3.set_yscale('log')
    # 눈금 숫자는 '3e-5' 로 — 기본(3×10⁻⁵ 수식)은 한글 글꼴에 수식용 마이너스가 없어 깨진다
    from matplotlib.ticker import LogFormatter
    ax3.yaxis.set_major_formatter(LogFormatter(labelOnlyBase=False))
    ax3.yaxis.set_minor_formatter(LogFormatter(labelOnlyBase=False))
    ax3.set_ylabel(L['lr'], color=INK2)
    ax3.set_xlabel(L['epoch'], color=INK2)

    # ── best / 후보 표시 ──
    for ax in (ax1, ax2, ax3):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(GRID)
        ax.tick_params(which='both', colors=INK2, labelsize=8.5)
        if best is not None:
            ax.axvline(best, color=INK, lw=1.2, alpha=0.8, zorder=1)
        for c in cands:
            ax.axvline(c, color=INK2, lw=1.1, ls=(0, (1, 2)), zorder=1)
    bottom = ax1.get_ylim()[0]
    if best is not None:
        ax1.annotate(L['best'].format(best), (best, bottom), xytext=(3, 4),
                     textcoords='offset points', fontsize=8.5, color=INK)
    for c in cands:
        ax1.annotate(L['cand'].format(c), (c, bottom), xytext=(3, 16),
                     textcoords='offset points', fontsize=8, color=INK2)

    title = csv_path.stem
    if best is not None:
        title += f"   —   {L['best'].format(best)} ({L['why_' + best_why]})"
    if result.get('early_stop'):
        title += f"   {L['early_stop']}"
    fig.suptitle(title, x=0.01, y=0.955, ha='left', fontsize=11, color=INK)
    return fig


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    for p in sys.argv[1:]:
        out = plot_trainlog(p)
        print(f'{p} → {out}' if out else f'{p}: 행 없음, 건너뜀')
