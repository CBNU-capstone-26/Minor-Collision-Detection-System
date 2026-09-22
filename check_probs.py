"""학습 중 저장된 best 가중치로 '검증셋 A 확률 분포'를 직접 확인한다.

recall/acc 는 argmax(임계값 0.5) 기준이라, 모델이 A 확률을 0.1 → 0.45 로
올리는 동안 계속 0 으로 찍힌다. "배우고 있는 중"인지 "아무것도 못 배웠는지"
구분이 안 되므로 확률을 직접 본다.

  · A 샘플의 평균 A확률 vs S 샘플의 평균 A확률 → 둘이 벌어지면 판별 중
  · AUC → 임계값과 무관한 판별력 (0.5 = 무작위, 1.0 = 완벽)
  · 임계값별 recall/precision → 0.5 가 아닌 값에서 잡히는지

학습과 동시에 실행 가능(GPU 메모리 여유가 있으면). 분할은 train.py 와 같은
seed 42 를 쓰므로 검증셋이 정확히 일치한다.

사용:
    cd "x3d model" && python3 ../check_probs.py ../weights/hitandrun_x3d_ptY_best.pth
"""
from __future__ import annotations

import os
import sys

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.getcwd())          # 모델 폴더 안에서 실행하는 것을 전제

import config                             # noqa: E402
from dataset import HitAndRunDataset      # noqa: E402
from device_utils import get_device       # noqa: E402
from hitandrun_model import HitAndRun3DCNN  # noqa: E402


def _split_val(dataset, ratio=config.TRAIN_SPLIT_RATIO):
    """train.py 와 동일한 계층적 영상단위 분할 (seed 42) → (val_rc, val_real) 인덱스."""
    groups: dict = {}
    for i, s in enumerate(dataset.samples):
        parts = os.path.normpath(s['mp4_path']).split(os.sep)
        dom = 'real' if 'realdata' in parts else 'rc'
        name = s['file_name']
        seg = name.split('_')
        if dom == 'rc':
            direction = seg[1][0] if len(seg) > 1 and seg[1] else '-'
            scen = seg[1][-1] if len(seg) > 1 and seg[1] else '-'
        else:
            direction = '-'
            scen = seg[1][-1] if len(seg) > 1 and seg[1] else '-'
        key = (dom, direction, scen)
        groups.setdefault(key, {}).setdefault(s.get('video_key', s['mp4_path']), []).append(i)

    gen = torch.Generator().manual_seed(42)
    val_rc, val_real = [], []
    for (dom, _d, _s), vid_map in sorted(groups.items()):
        vkeys = sorted(vid_map)
        order = torch.randperm(len(vkeys), generator=gen).tolist()
        shuffled = [vkeys[i] for i in order]
        cut = int(ratio * len(vkeys))
        for vk in shuffled[cut:]:
            (val_real if dom == 'real' else val_rc).extend(vid_map[vk])
    return val_rc, val_real


def _auc(pos, neg):
    """Mann-Whitney U 로 AUC. pos/neg = A·S 샘플의 A확률 리스트."""
    if not pos or not neg:
        return None
    ranked = sorted([(p, 1) for p in pos] + [(n, 0) for n in neg])
    # 동점은 평균 순위로
    r, i, s_pos = 0.0, 0, 0.0
    while i < len(ranked):
        j = i
        while j + 1 < len(ranked) and ranked[j + 1][0] == ranked[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            if ranked[k][1] == 1:
                s_pos += avg_rank
        i = j + 1
    n1, n0 = len(pos), len(neg)
    return (s_pos - n1 * (n1 + 1) / 2) / (n1 * n0)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    weights = sys.argv[1]
    device = get_device(config.INFER_DEVICE_TYPE)

    ds_eval = HitAndRunDataset(augment=False)      # 검증은 증강 없음
    val_rc, val_real = _split_val(ds_eval)
    print(f"[분할] val_rc {len(val_rc)}클립 / val_real {len(val_real)}클립 (seed 42)")

    model = HitAndRun3DCNN(num_classes=config.MODEL_NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
    model.eval()
    print(f"[가중치] {weights}\n")

    for name, idxs in [("val_rc", val_rc), ("val_real", val_real)]:
        if not idxs:
            continue
        loader = DataLoader(Subset(ds_eval, idxs),
                            batch_size=config.TRAIN_BATCH_SIZE, shuffle=False)
        pA, pS = [], []
        with torch.inference_mode():
            for x, y in loader:
                p = torch.softmax(model(x.to(device)), dim=1)[:, 1].cpu()
                for prob, lab in zip(p.tolist(), y.tolist()):
                    (pA if lab == 1 else pS).append(prob)

        print(f"══ {name}  (A {len(pA)} / S {len(pS)})")
        if pA:
            print(f"   A 샘플의 A확률 : 평균 {sum(pA)/len(pA):.4f}  "
                  f"최소 {min(pA):.4f}  최대 {max(pA):.4f}")
        print(f"   S 샘플의 A확률 : 평균 {sum(pS)/len(pS):.4f}  "
              f"최소 {min(pS):.4f}  최대 {max(pS):.4f}")
        if pA:
            gap = sum(pA)/len(pA) - sum(pS)/len(pS)
            auc = _auc(pA, pS)
            print(f"   판별력         : 평균차 {gap:+.4f}   AUC {auc:.4f}"
                  f"   {'← 판별 중' if auc and auc > 0.6 else '← 거의 무작위'}")
            print(f"   {'임계값':>8} {'recall':>8} {'precision':>10} {'예측A수':>8}")
            for th in (0.5, 0.4, 0.3, 0.2, 0.1):
                tp = sum(1 for p in pA if p >= th)
                fp = sum(1 for p in pS if p >= th)
                rec = tp / len(pA)
                prec = tp / (tp + fp) if (tp + fp) else 0.0
                mark = "  ← 현재 추론 기준" if th == 0.5 else ""
                print(f"   {th:>8.1f} {rec:>8.3f} {prec:>10.3f} {tp+fp:>8}{mark}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
