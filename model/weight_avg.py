"""한 학습 회차의 여러 에포크 가중치를 평균해 하나로 만듭니다. 추론 계산량은 그대로입니다.

같은 회차의 가까운 에포크들은 비슷한 지점에 있어, 평균하면 한 에포크의 우연한 치우침이 줄어 변형(압축, 야간 등)에
덜 민감해질 수 있습니다. 다른 회차끼리는 평균하지 않습니다.
BN 이동평균과 분산도 그대로 평균합니다. 가까운 에포크라 근사로 충분하다고 보며, 결과는 postproc_check와 robust_check로 확인합니다.
정수 버퍼(num_batches_tracked)는 가장 큰 값을 씁니다.

    python -m model.weight_avg --run outputs/trainlogs/<회차> --epochs 25-27,38,39,43

결과는 <회차>/avg/hitandrun_<모델>_<시작시각>_avg25-26-27-38-39-43_<조건>.pth로 저장합니다.
회차 폴더 안에 두므로 postproc_check가 그 회차의 split.json 검증 영상을 그대로 찾습니다.
"""
import argparse
import hashlib
import re
import sys
from pathlib import Path

import torch


def parse_epochs(text):
    """'25-27,38'을 [25, 26, 27, 38]로 바꿉니다(순서대로, 중복 없이)."""
    out = []
    for part in str(text).replace(' ', '').split(','):
        if not part:
            continue
        if '-' in part:
            a, b = (int(x) for x in part.split('-', 1))
            rng = range(a, b + 1) if a <= b else range(a, b - 1, -1)
        else:
            rng = [int(part)]
        out += [e for e in rng if e not in out]
    return out


def average(files):
    """state_dict들의 평균(실수 텐서)을 돌려줍니다. 정수 버퍼는 최댓값을 씁니다."""
    sds = [torch.load(str(f), map_location='cpu', weights_only=True) for f in files]
    keys = list(sds[0].keys())
    for f, sd in zip(files[1:], sds[1:]):
        if list(sd.keys()) != keys:
            raise SystemExit(f"키가 다름 — 같은 구조의 가중치가 아닙니다: {f}")
    out = {}
    for k in keys:
        ts = [sd[k] for sd in sds]
        if ts[0].is_floating_point():
            out[k] = torch.stack([t.double() for t in ts]).mean(0).to(ts[0].dtype)
        else:
            out[k] = torch.stack(ts).max(0).values.clone()
    return out, sds


def out_name(src, epochs):
    """에포크 파일 이름 hitandrun_<모델>_<시작시각>[_n]_<조건>_epNNN.pth를 ..._<시작시각>[_n]_avg<에포크들>_<조건>.pth로 바꿉니다.
    조건 표시(_ptY, _bal, _lqw, _top3 등)는 그대로 둡니다. 평가 폴더 이름, 헤드 풀링, 입력 크기를 이름으로 맞추기 때문입니다.
    """
    base = re.sub(r'_ep\d+$', '', Path(src).stem)
    tag = 'avg' + '-'.join(str(e) for e in epochs)
    m = re.match(r'^(hitandrun_[A-Za-z0-9]+_\d{6}_\d{4}(?:_\d+)?)_(pt[YN].*)$', base)
    return f"{m.group(1)}_{tag}_{m.group(2)}.pth" if m else f"{base}_{tag}.pth"


def main(argv=None):
    ap = argparse.ArgumentParser(description='한 회차의 에포크 가중치 평균')
    ap.add_argument('--run', required=True, help='학습 회차 폴더(outputs/trainlogs/<회차>) — 그 안의 epochs/ 를 쓴다')
    ap.add_argument('--epochs', required=True, help='평균할 에포크(예: 25-27,38,39,43)')
    ap.add_argument('--out-dir', help='저장 폴더(기본 <회차>/avg)')
    args = ap.parse_args(argv)

    run = Path(args.run)
    ep_dir = run / 'epochs'
    epochs = parse_epochs(args.epochs)
    if len(epochs) < 2:
        raise SystemExit('에포크를 2개 이상 주세요')
    files = []
    for e in epochs:
        hit = sorted(ep_dir.glob(f'*_ep{e:03d}.pth'))
        if not hit:
            raise SystemExit(f"에포크 {e} 가중치가 없습니다: {ep_dir}/*_ep{e:03d}.pth")
        files.append(hit[0])

    avg, sds = average(files)
    out_dir = Path(args.out_dir) if args.out_dir else run / 'avg'
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / out_name(files[0], epochs)
    torch.save(avg, str(path))

    # 원래 에포크들과 얼마나 떨어졌는지(실수 텐서 전체의 평균 절대 차이)입니다. 너무 크면 서로 먼 에포크를 섞은 것입니다.
    def _gap(sd):
        num = den = 0.0
        for k, v in avg.items():
            if v.is_floating_point():
                num += float((sd[k].double() - v.double()).abs().sum())
                den += v.numel()
        return num / max(den, 1)
    print(f"[가중치 평균] {run.name} · 에포크 {', '.join(map(str, epochs))} ({len(files)}개)")
    for e, sd in zip(epochs, sds):
        print(f"  ep{e:<3} 평균과의 차이 {_gap(sd):.2e}")
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    print(f"  저장: {path} · md5 {md5}")
    return path


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
