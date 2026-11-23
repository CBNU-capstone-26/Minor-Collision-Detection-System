"""학습 중 저장된 가중치로 검증셋의 A 확률 분포를 직접 확인합니다.

재현율과 정확도는 판정 기준(0.5)으로 계산하므로, 모델이 A 확률을 0.1에서 0.45로 올리는 동안에도 계속 0으로 나옵니다.
배우는 중인지 아무것도 못 배웠는지 가를 수 없어 확률을 직접 봅니다.
    A 샘플과 S 샘플의 평균 A 확률: 둘이 벌어지면 가르고 있는 것입니다.
    AUC: 기준과 무관한 구분력입니다(0.5는 무작위, 1.0은 완벽).
    기준별 재현율과 정밀도: 0.5가 아닌 기준에서 잡히는지 봅니다.

GPU 메모리에 여유가 있으면 학습과 동시에 돌릴 수 있습니다.
검증 분할은 이 스크립트가 seed 42 방식으로 직접 나눕니다. 지금 학습은 검증 폴더(data/val)를 따로 쓰므로 검증 영상이 다를 수 있습니다.

사용
    cd model && python3 ../check_probs.py ../weights/<가중치>.pth
"""
from __future__ import annotations

import os
import sys

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.getcwd())          # 모델 폴더 안에서 실행하는 것을 전제로 합니다.

import config                             # noqa: E402
from dataset import HitAndRunDataset      # noqa: E402
from device_utils import get_device       # noqa: E402
from hitandrun_model import HitAndRun3DCNN  # noqa: E402


def _split_val(dataset, ratio=config.TRAIN_SPLIT_RATIO):
    """seed 42로 영상 단위 계층 분할을 해 (val_rc, val_real) 인덱스를 돌려줍니다."""
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
    """Mann-Whitney U로 AUC를 계산합니다. pos와 neg는 A, S 샘플의 A 확률 목록입니다."""
    if not pos or not neg:
        return None
    ranked = sorted([(p, 1) for p in pos] + [(n, 0) for n in neg])
    # 동점은 평균 순위로 처리합니다.
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

    ds_eval = HitAndRunDataset(augment=False)      # 검증은 증강하지 않습니다.
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
        a_sample_probs, s_sample_probs = [], []
        with torch.inference_mode():
            for x, y in loader:
                p = torch.softmax(model(x.to(device)), dim=1)[:, 1].cpu()
                for prob, lab in zip(p.tolist(), y.tolist()):
                    (a_sample_probs if lab == 1 else s_sample_probs).append(prob)

        print(f"══ {name}  (A {len(a_sample_probs)} / S {len(s_sample_probs)})")
        if a_sample_probs:
            print(f"   A 샘플의 A확률 : 평균 {sum(a_sample_probs)/len(a_sample_probs):.4f}  "
                  f"최소 {min(a_sample_probs):.4f}  최대 {max(a_sample_probs):.4f}")
        print(f"   S 샘플의 A확률 : 평균 {sum(s_sample_probs)/len(s_sample_probs):.4f}  "
              f"최소 {min(s_sample_probs):.4f}  최대 {max(s_sample_probs):.4f}")
        if a_sample_probs:
            gap = sum(a_sample_probs)/len(a_sample_probs) - sum(s_sample_probs)/len(s_sample_probs)
            auc = _auc(a_sample_probs, s_sample_probs)
            print(f"   판별력         : 평균차 {gap:+.4f}   AUC {auc:.4f}"
                  f"   {'← 판별 중' if auc and auc > 0.6 else '← 거의 무작위'}")
            print(f"   {'임계값':>8} {'recall':>8} {'precision':>10} {'예측A수':>8}")
            for th in (0.5, 0.4, 0.3, 0.2, 0.1):
                tp = sum(1 for p in a_sample_probs if p >= th)
                fp = sum(1 for p in s_sample_probs if p >= th)
                rec = tp / len(a_sample_probs)
                prec = tp / (tp + fp) if (tp + fp) else 0.0
                mark = "  ← 현재 추론 기준" if th == 0.5 else ""
                print(f"   {th:>8.1f} {rec:>8.3f} {prec:>10.3f} {tp+fp:>8}{mark}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
