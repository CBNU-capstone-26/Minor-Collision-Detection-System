"""학습 배치 구성을 정합니다. 빈 장면 정상은 돌려 쓰고, 충돌 클립은 배치마다 고르게 넣습니다.

config.TRAIN_S_MOTION_CSV나 TRAIN_SPREAD_A가 켜졌을 때만 train.py가 씁니다(기본은 끔).
학습 쪽(train_idx)에만 적용하고 검증은 건드리지 않습니다.

빈 장면: model/s_motion_check.py가 잰 움직임(clips.csv의 move_pct)이 기준보다 작은 정상 클립입니다.
         매 에포크 그중 KEEP 비율만 쓰되, 고정된 순서에서 에포크마다 다음 구간을 잘라 써서 1/KEEP 에포크마다 전부 한 번씩 봅니다.
         CSV에 없는 정상 클립은 움직임 있음으로 보고 매번 씁니다.
충돌 고르게: 배치 순서를 섞은 뒤 충돌 클립을 배치마다 하나씩 돌아가며 넣고(복제 없음) 나머지 자리를 정상으로 채웁니다.
         충돌이 배치 수 이상이면 모든 배치에 충돌이 들어갑니다.
"""
import csv
import math
import os
from pathlib import Path

import numpy as np
from torch.utils.data import Sampler

import config


def resolve_csv(path):
    """상대경로는 프로젝트 루트 기준입니다."""
    p = Path(path)
    return p if p.is_absolute() else Path(config._ROOT) / p


def load_motion(csv_path):
    """clips.csv를 읽어 {(영상 이름, 차량 번호, 시작 프레임): 움직임 %}를 돌려줍니다(정상 클립만)."""
    motion = {}
    with open(csv_path, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['label'] != '0':
                continue
            try:
                motion[(r['video'], int(r['car_id']), int(r['start_f']))] = float(r['move_pct'])
            except ValueError:
                continue          # nan 등 재지 못한 클립은 움직임 있음으로 둡니다.
    return motion


class BalancedBatchSampler(Sampler):
    """학습 Subset 안의 위치(0부터)로 에포크마다 배치 목록을 만듭니다."""

    def __init__(self, a_pos, s_move_pos, s_empty_pos, batch_size, empty_keep=1.0,
                 spread_a=False, seed=42):
        self.a = np.asarray(a_pos, dtype=np.int64)
        self.s_move = np.asarray(s_move_pos, dtype=np.int64)
        self.bs = int(batch_size)
        self.spread_a = bool(spread_a)
        self.seed = int(seed)
        self.epoch = 0
        empty = np.asarray(s_empty_pos, dtype=np.int64)
        # 빈 장면은 한 번 정한 순서에서 에포크마다 다음 구간을 잘라 씁니다(끝에 닿으면 처음으로 돌아갑니다).
        self.empty_order = np.random.RandomState(seed).permutation(empty)
        self.k_empty = (len(empty) if empty_keep >= 1
                        else int(round(len(empty) * max(0.0, float(empty_keep)))))

    # 에포크당 개수(에포크마다 같습니다)
    def counts(self):
        n_s = len(self.s_move) + self.k_empty
        n = len(self.a) + n_s
        nb = math.ceil(n / self.bs) if n else 0
        if self.spread_a:
            no_a = max(0, nb - len(self.a))
        else:      # 무작위로 섞을 때의 기대값
            no_a = nb * ((n_s / n) ** self.bs) if n else 0
        return {'a': len(self.a), 's_move': len(self.s_move), 's_empty': self.k_empty,
                's_empty_total': len(self.empty_order), 'n': n, 'batches': nb, 'no_a_batches': no_a}

    def _empty_chunk(self, epoch):
        m, k = len(self.empty_order), self.k_empty
        if k >= m:
            return self.empty_order
        if k == 0:
            return self.empty_order[:0]
        start = (epoch * k) % m
        idx = (start + np.arange(k)) % m
        return self.empty_order[idx]

    def batches(self, epoch):
        rng = np.random.RandomState(self.seed + 1000 + epoch)
        a = rng.permutation(self.a)
        s = rng.permutation(np.concatenate([self.s_move, self._empty_chunk(epoch)]))
        n = len(a) + len(s)
        if n == 0:
            return []
        nb = math.ceil(n / self.bs)
        if not self.spread_a:
            allp = rng.permutation(np.concatenate([a, s]))
            return [allp[i * self.bs:(i + 1) * self.bs].tolist() for i in range(nb)]
        sizes = [self.bs] * (nb - 1) + [n - self.bs * (nb - 1)]
        cnt = [0] * nb
        order = rng.permutation(nb)
        k = 0
        for _ in range(len(a)):           # 섞은 배치 순서로 한 바퀴씩 돌며 충돌을 하나씩 넣습니다.
            while cnt[order[k % nb]] >= sizes[order[k % nb]]:
                k += 1
            cnt[order[k % nb]] += 1
            k += 1
        out, ia, is_ = [], 0, 0
        for i in range(nb):
            b = a[ia:ia + cnt[i]].tolist() + s[is_:is_ + sizes[i] - cnt[i]].tolist()
            ia += cnt[i]
            is_ += sizes[i] - cnt[i]
            out.append(rng.permutation(b).tolist())
        return out

    def __iter__(self):
        # 생성기라 실제로 꺼내 쓰기 시작할 때 에포크가 넘어갑니다. 작업자를 유지하는 DataLoader는 처음에 iter()를 두 번 부르고
        # 앞의 것을 버리는데, 그때 에포크가 건너뛰지 않게 하려는 것입니다.
        epoch = self.epoch
        self.epoch += 1
        yield from self.batches(epoch)

    def __len__(self):
        return self.counts()['batches']


def build(train_samples, train_idx, batch_size, csv_path=None, empty_pct=0.25,
          empty_keep=1 / 3, spread_a=False, seed=42, thin_rc=True):
    """train.py용입니다. 학습 쪽 위치를 충돌, 움직임 있는 정상, 빈 장면 정상으로 나눠 샘플러와 요약을 돌려줍니다.
    thin_rc=False(config.TRAIN_S_THIN_RC)면 RC 정상은 솎지 않고 매 에포크 전부 씁니다.
    """
    motion = load_motion(csv_path) if csv_path else {}
    a_pos, s_move, s_empty = [], [], []
    matched = rc_full = ctx = hn = 0
    for pos, i in enumerate(train_idx):
        s = train_samples[i]
        if s['label'] == 1:
            a_pos.append(pos)
            continue
        if s.get('ctx_neg'):          # 사고 영상의 충돌 전후 정상(config.TRAIN_A_CONTEXT_NEG)은 측정표에 없으므로 매번 씁니다.
            s_move.append(pos)
            ctx += 1
            continue
        if s.get('hardneg'):          # 헷갈리는 정상 묶음(config.TRAIN_HARDNEG_CSV)은 측정표의 클립이 아니므로 매번 씁니다.
            s_move.append(pos)
            hn += 1
            continue
        if not thin_rc and 'realdata' not in os.path.normpath(s['mp4_path']).split(os.sep):
            s_move.append(pos)
            rc_full += 1
            continue
        m = motion.get((s['file_name'], int(s.get('car_id', 0)), int(s['start_f'])))
        if m is not None:
            matched += 1
        (s_empty if (m is not None and m < empty_pct) else s_move).append(pos)
    if not csv_path:           # 솎기를 끄면 빈 장면 구분 없이 모두 움직임 있음으로 보고 매번 씁니다.
        s_move, s_empty = s_move + s_empty, []
    sampler = BalancedBatchSampler(a_pos, s_move, s_empty, batch_size,
                                   empty_keep=empty_keep if csv_path else 1.0,
                                   spread_a=spread_a, seed=seed)
    c = sampler.counts()
    n_s_all = len(s_move) + len(s_empty) - rc_full - ctx - hn     # 솎기 대상 정상(RC를 빼면 실제 영상만)
    lines = []
    if ctx:
        lines.append(f"사고 영상의 충돌 전·후 정상 {ctx}개는 솎지 않음(TRAIN_A_CONTEXT_NEG) — 매 에포크 전부 씀")
    if hn:
        lines.append(f"헷갈리는 정상 묶음 {hn}개는 솎지 않음(TRAIN_HARDNEG_CSV) — 매 에포크 전부 씀")
    if rc_full:
        lines.append(f"RC 정상 {rc_full}개는 솎지 않음(TRAIN_S_THIN_RC=0) — 움직임과 상관없이 매 에포크 전부 씀")
    if csv_path:
        lines.append(f"빈 장면 정상 솎기: 움직임 {empty_pct:g}% 미만을 에포크마다 {empty_keep:.0%}만 돌려 씀 — "
                     f"학습 정상 {n_s_all}개 중 빈 장면 {len(s_empty)}개({len(s_empty) / max(n_s_all, 1):.0%}), "
                     f"CSV 짝 맞춤 {matched}/{n_s_all}")
        if matched < n_s_all:
            lines.append(f"  ⚠️ CSV 에 없는 정상 {n_s_all - matched}개는 '움직임 있음'으로 매번 씀 "
                         f"(데이터가 바뀌었으면 s_motion_check 를 다시 돌릴 것)")
    lines.append(f"묶음 구성: 충돌 {'묶음마다 고르게(복제 없음)' if spread_a else '무작위'} — 에포크당 "
                 f"충돌 {c['a']} · 정상 {c['s_move'] + c['s_empty']}(움직임 {c['s_move']} + 빈 장면 {c['s_empty']}) "
                 f"= 1:{(c['s_move'] + c['s_empty']) / max(c['a'], 1):.1f} · 묶음 {c['batches']}개 · "
                 f"충돌 없는 묶음 {c['no_a_batches']:.0f}개({c['no_a_batches'] / max(c['batches'], 1):.0%}"
                 f"{'' if spread_a else ', 무작위 기대값'})")
    return sampler, lines
