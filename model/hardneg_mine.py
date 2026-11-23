"""학습 영상에서 다른 차가 지나가는 장면처럼 헷갈리는 정상 묶음(하드 네거티브)을 찾습니다.

가중치(기본 서비스 가중치)로 학습 영상의 간격 1 P(A)를 재서, 정상인데 높게 나온 묶음을 고릅니다.
고른 묶음은 학습 옵션 TRAIN_HARDNEG_CSV로 정상 클립에 더할 수 있습니다.

대상은 학습 폴더(config.DATA_DIR)뿐이며, 검증 고정 목록(split_lock.json)에 있는 영상은 학습 폴더에 남아 있어도 뺍니다.
    실제 정상 영상: 학습 정상 클립과 같은 차(0번과 1번)
    실제 사고 영상: 충돌에서 --margin-sec(기본 3초) 이상 떨어진 묶음만(0번 차)
    --rc를 주면 RC 영상도 같은 규칙으로 봅니다(RC 정상은 학습과 같이 0번 차만).
P(A)가 --min-prob(기본 0.3)보다 높은 묶음을 이어진 구간으로 묶고, 구간마다 가장 높은 묶음부터
10프레임 이상 떨어진 것을 최대 --per-seg(기본 3)개 고릅니다. 한 장면이 학습을 덮지 않게 하려는 것입니다.
결과는 outputs/hardneg/<시각>/에 hardneg.csv, report.txt, sheet_NN.jpg로 남습니다.
그림으로 정말 정상인지 확인하고, 레이블이 빠진 충돌이 섞여 있으면 그 줄을 CSV에서 지운 뒤 씁니다.

    python -u -m model.hardneg_mine               # GPU 권장, 서비스 가중치, 실제 영상
    python -u -m model.hardneg_mine --rc          # RC 영상도
    python -u -m model.hardneg_mine --videos 2    # 코드 점검(종류마다 가장 짧은 영상 2개)
"""
import argparse
import csv
import heapq
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

import config
from evaluate import _Tee, _read_label
from postproc_check import _excluded, _short, _window_probs

L = config.CLIP_LENGTH
PIX_T = 15          # 움직인 화소 기준(s_motion_check와 같습니다)
SHEET_ROWS = 10


def _val_lock_stems():
    """검증 고정 목록의 영상 이름입니다. 학습 폴더에 남아 있어도 캐지 않습니다."""
    p = getattr(config, 'TRAIN_SPLIT_LOCK', None)
    try:
        with open(p, encoding='utf-8') as f:
            return {Path(r).stem for r in json.load(f).get('val', [])}
    except (OSError, TypeError, ValueError):
        return set()


def _videos(use_rc, n_check):
    """학습 폴더의 [(mp4, 영역, 종류, [(차 번호, 박스)], 충돌 시작, 충돌 끝)]을 돌려줍니다."""
    lock = _val_lock_stems()
    out = []
    for dom in ('realdata', 'rcdata') if use_rc else ('realdata',):
        for mp4 in sorted((Path(config.DATA_DIR) / dom).glob('*.mp4')):
            if _excluded(mp4.stem) or mp4.stem in lock or not mp4.with_suffix('.txt').exists():
                continue
            bboxes, sf, ef = _read_label(str(mp4.with_suffix('.txt')))
            if not bboxes:
                continue
            tid = config.TARGET_ID if config.TARGET_ID in bboxes else next(iter(bboxes))
            if sf is None:
                cars = [tid] + ([1] if dom == 'realdata' and 1 in bboxes and tid != 1
                                and getattr(config, 'TRAIN_S_REAL_USE_CAR1', False) else [])
                kind = '정상'
            else:
                cars, kind = [tid], '사고먼곳'
            out.append((mp4, 'real' if dom == 'realdata' else 'rc', kind, [(c, bboxes[c]) for c in cars], sf, ef))
    if n_check:
        def _n(v):
            cap = cv2.VideoCapture(str(v[0]))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return n
        picked = []
        for key in {(v[1], v[2]) for v in out}:
            picked += sorted((v for v in out if (v[1], v[2]) == key), key=_n)[:n_check]
        out = sorted(picked, key=lambda v: v[0].name)
    return out


def _read_multi(mp4, boxes):
    """서비스와 같은 크롭(r=config.R_VALUE, 224)을 차마다 만듭니다. 영상은 한 번만 읽고 ({차: [uint8 RGB]}, fps)를 돌려줍니다."""
    from inference import _crop_square_and_pad
    cap = cv2.VideoCapture(str(mp4))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = {c: [] for c, _ in boxes}
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        for c, b in boxes:
            frames[c].append(_crop_square_and_pad(rgb, b, config.R_VALUE, config.RESIZE)[0])
    cap.release()
    return frames, fps


def _motion(frames, w):
    """묶음 안 이웃 프레임의 움직인 화소 비율(%) 중 3번째로 큰 값입니다(s_motion_check와 같은 계산)."""
    gs = [cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (5, 5), 0) for f in frames[w:w + L]]
    v = sorted(float((cv2.absdiff(a, b) > PIX_T).mean()) * 100 for a, b in zip(gs[1:], gs[:-1]))
    return v[-3] if len(v) >= 3 else (v[-1] if v else 0.0)


def _segments(idx, gap=3):
    """정렬된 묶음 번호를 이어진 구간 [(시작, 끝)]으로 묶습니다."""
    segs = []
    for i in idx:
        if segs and i - segs[-1][1] <= gap:
            segs[-1][1] = i
        else:
            segs.append([i, i])
    return segs


def _tile_row(frames, w, text):
    idx = np.linspace(w, w + L - 1, 6).round().astype(int)
    tiles = [cv2.resize(cv2.cvtColor(frames[min(i, len(frames) - 1)], cv2.COLOR_RGB2BGR), (128, 128)) for i in idx]
    row = np.hstack(tiles)
    cv2.putText(row, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3)
    cv2.putText(row, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
    return row


def main():
    ap = argparse.ArgumentParser(description='학습 영상에서 헷갈리는 정상 묶음(하드 네거티브) 캐기')
    ap.add_argument('--weights', default=str(config.SERVICE_WEIGHTS_PATH), help='잴 가중치(기본 서비스 가중치)')
    ap.add_argument('--min-prob', type=float, default=0.3, help='이 P(A) 를 넘는 정상 묶음을 고른다(기본 0.3)')
    ap.add_argument('--margin-sec', type=float, default=3.0, help='사고 영상은 충돌에서 이만큼 떨어진 묶음만(기본 3초)')
    ap.add_argument('--per-seg', type=int, default=3, help='이어진 구간마다 최대 몇 개(기본 3, 10프레임 이상 간격)')
    ap.add_argument('--rc', action='store_true', help='RC 영상도(서버)')
    ap.add_argument('--batch', type=int, default=0, help='추론 묶음(0 = 서비스와 같게 GPU 8 · CPU 1)')
    ap.add_argument('--videos', type=int, default=0, help='코드 점검: 종류마다 가장 짧은 영상 이 수만')
    ap.add_argument('--sheet-max', type=int, default=60, help='그림으로 남길 구간 수(P(A) 높은 순)')
    args = ap.parse_args()

    vids = _videos(args.rc, args.videos)
    out_dir = Path(config._ROOT) / 'outputs' / 'hardneg' / datetime.now().strftime('%y%m%d_%H%M%S')
    out_dir.mkdir(parents=True, exist_ok=True)
    tee = _Tee(out_dir / 'report.txt')
    sys.stdout = tee
    done = False
    try:
        from device_utils import is_cuda_like, predict_batch_size
        from infer_runtime import get_runner
        from main import _load_model
        print(f"[헷갈리는 정상 캐기] {datetime.now():%Y-%m-%d %H:%M} · 가중치 {Path(args.weights).name}")
        print(f"기준 P(A) > {args.min_prob:g} · 사고 영상은 충돌에서 {args.margin_sec:g}초 이상 · 구간마다 최대 {args.per_seg}개"
              f" · 영상 {len(vids)}개(" + ', '.join(f"{d} {k} {sum(v[1] == d and v[2] == k for v in vids)}"
                                              for d in ('real', 'rc') for k in ('정상', '사고먼곳')
                                              if any(v[1] == d and v[2] == k for v in vids)) + ')')
        model, device = _load_model(args.weights)
        model.eval()
        if is_cuda_like(device):
            torch.backends.cudnn.benchmark = True      # 실행기를 고르기 전에 켭니다(infer_runtime 참고).
        runner = get_runner(model)
        batch = args.batch or predict_batch_size(device)
        print(f"실행기 {runner.name} · 장치 {device} · 묶음 {batch}", flush=True)

        rows, sheet = [], []          # sheet는 (최대 P, 순번, 그림 줄)을 담는 최소 힙입니다.
        scanned = {k: 0 for k in ('정상', '사고먼곳')}
        t0 = time.perf_counter()
        for vi, (mp4, dom, kind, boxes, sf, ef) in enumerate(vids, 1):
            frames, fps = _read_multi(mp4, boxes)
            m = int(round(args.margin_sec * fps))
            msg = []
            for car, _ in boxes:
                fr = frames[car]
                if len(fr) < L:
                    continue
                pa = _window_probs(runner, device, fr, batch)
                w = np.arange(len(pa))
                ok = np.ones(len(pa), bool) if sf is None else ((w + L - 1 < sf - m) | (w > ef + m))
                scanned[kind] += int(ok.sum())
                hot = np.nonzero(ok & (pa > args.min_prob))[0]
                n_pick = 0
                for lo, hi in _segments(hot.tolist()):
                    order = sorted(range(lo, hi + 1), key=lambda i: -pa[i])
                    order = [i for i in order if ok[i] and pa[i] > args.min_prob]
                    chosen = []
                    for i in order:
                        if all(abs(i - c) >= 10 for c in chosen):
                            chosen.append(i)
                        if len(chosen) >= args.per_seg:
                            break
                    peak = float(pa[order[0]])
                    for i in chosen:
                        rows.append(dict(video=mp4.stem, rel=os.path.relpath(mp4, config.DATA_DIR).replace('\\', '/'),
                                         domain=dom, kind=kind, car_id=car, start_f=int(i), p=round(float(pa[i]), 4),
                                         seg_lo=lo, seg_hi=hi, seg_peak=round(peak, 4),
                                         motion=round(_motion(fr, int(i)), 3), fps=round(fps, 3)))
                    n_pick += len(chosen)
                    item = (peak, len(rows), _tile_row(fr, order[0], f"{_short(mp4.stem)} car{car} f{order[0]} P{peak:.2f}"))
                    if len(sheet) < args.sheet_max:
                        heapq.heappush(sheet, item)
                    elif peak > sheet[0][0]:
                        heapq.heapreplace(sheet, item)
                msg.append(f"차{car} 최대 {float(pa[ok].max()) if ok.any() else 0:.2f} · 고름 {n_pick}")
            del frames
            print(f"  {vi}/{len(vids)} {dom} {kind} {_short(mp4.stem)} {fps:.0f}fps · " + ' · '.join(msg)
                  + f" · {time.perf_counter() - t0:.0f}초", flush=True)

        with open(out_dir / 'hardneg.csv', 'w', newline='', encoding='utf-8') as f:
            cols = ['video', 'rel', 'domain', 'kind', 'car_id', 'start_f', 'p', 'seg_lo', 'seg_hi', 'seg_peak', 'motion', 'fps']
            wr = csv.DictWriter(f, fieldnames=cols)
            wr.writeheader()
            wr.writerows(rows)
        sheet = sorted(sheet, key=lambda x: -x[0])
        for k in range(0, len(sheet), SHEET_ROWS):
            cv2.imwrite(str(out_dir / f'sheet_{k // SHEET_ROWS + 1:02d}.jpg'), np.vstack([s[2] for s in sheet[k:k + SHEET_ROWS]]))

        print(f"\n[요약] 잰 묶음: 정상 영상 {scanned['정상']} · 사고 영상 먼 곳 {scanned['사고먼곳']} · {time.perf_counter() - t0:.0f}초")
        for dom in ('real', 'rc'):
            for kind in ('정상', '사고먼곳'):
                sel = [r for r in rows if r['domain'] == dom and r['kind'] == kind]
                if sel:
                    ps = np.array([r['p'] for r in sel])
                    print(f"  {dom} {kind}: 고른 묶음 {len(sel)}개 · 영상 {len({r['video'] for r in sel})}개 · 구간 "
                          f"{len({(r['video'], r['car_id'], r['seg_lo']) for r in sel})}개 · P(A) 0.5 넘음 {int((ps > 0.5).sum())}"
                          f" · 0.9 넘음 {int((ps > 0.9).sum())} · 움직임 0.25% 미만(빈 장면) {sum(r['motion'] < 0.25 for r in sel)}")
        print(f"  합계 {len(rows)}개 → {out_dir / 'hardneg.csv'}")
        print(f"  그림 {len(sheet)}구간(P(A) 높은 순) → sheet_*.jpg — 레이블이 빠진 충돌이 보이면 그 영상 줄을 CSV 에서 지운다")
        print(f"  학습에 쓰기: TRAIN_HARDNEG_CSV={os.path.relpath(out_dir / 'hardneg.csv', config._ROOT)} (반복 TRAIN_HARDNEG_REPEAT, 이름 _hn)")
        done = True
        return 0
    finally:
        sys.stdout = tee.stdout
        tee.close()
        print(f'\n[저장] {out_dir}' if done else f'\n[중단] 오류로 멈춤 — 일부만 남음: {out_dir}')


if __name__ == '__main__':
    sys.exit(main())
