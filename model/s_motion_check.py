"""비충돌(S) 학습 클립의 '움직임 양'을 재서, 움직임 없는 S(빈 장면)를 솎았을 때의 A:S 비율을 미리 본다.

    python -u -m model.s_motion_check                       # data/train 전체
    python -u -m model.s_motion_check --workers 8 --montage # 클립 예시 그림까지

학습과 같은 클립 목록을 쓴다(HitAndRunDataset — A 는 pre 방식 창 1개, S 는 30프레임 간격,
실제 영상 S 는 0·1번 차량). 클립마다 학습 입력과 같은 크롭(r=1, 224×224)에서 이웃 프레임
차이를 구해 '움직인 화소 비율'을 잰다. GPU 는 쓰지 않는다(CPU 영상 디코딩, 영상 단위 병렬).

움직임 값(move) = 클립 안 프레임별 '움직인 화소 비율(%)' 중 3번째로 큰 값
    (5×5 블러 후 밝기 차이가 PIX_T 를 넘는 화소. 압축 튐 1~2프레임은 무시된다)

결과: outputs/s_motion/<시각>/clips.csv(클립별 값) · report.txt(요약) · montage_*.jpg(--montage)
"""
import argparse
import csv
import os
import sys
import time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

import config
from dataset import HitAndRunDataset

PIX_T = 15                                  # 이 밝기 차이(0~255)를 넘으면 '움직인 화소'
THRESHOLDS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0)   # '빈 장면' 기준 후보(움직인 화소 %)
KEEP_FRACS = (0.0, 0.2, 1 / 3, 0.5)         # 빈 장면 S 를 이 비율만 남길 때
BATCH = 15                                  # 충돌 없는 묶음 비율 계산용(config.TRAIN_BATCH_SIZE)


class _Crop:
    """HitAndRunDataset._crop_and_pad 를 그대로 쓰기 위한 최소 객체(resize 만 필요)."""
    def __init__(self, resize):
        self.resize = resize

    def __call__(self, frame, bbox):
        return HitAndRunDataset._crop_and_pad(self, frame, list(bbox), config.R_VALUE)


def _gray(crop):
    return cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (5, 5), 0)


def _video_task(args):
    """영상 하나를 처음부터 한 번 읽어, 필요한 박스별로 프레임 차이 시계열을 만들고 클립 값으로 묶는다."""
    mp4, items, clip_len, resize = args       # items: [(sample_idx, start_f, bbox)]
    crop = _Crop(resize)
    boxes = sorted({tuple(b) for _, _, b in items})
    need = max(s for _, s, _ in items) + clip_len
    frac = {b: [] for b in boxes}             # frac[b][k] = 프레임 k → k+1 움직인 화소 비율
    dmean = {b: [] for b in boxes}
    prev = dict.fromkeys(boxes)
    cap = cv2.VideoCapture(mp4)
    t = 0
    while t < need:
        ok, frame = cap.read()
        if not ok:
            break
        for b in boxes:
            g = _gray(crop(frame, b))
            if prev[b] is not None:
                d = cv2.absdiff(g, prev[b])
                frac[b].append(float((d > PIX_T).mean()) * 100.0)
                dmean[b].append(float(d.mean()))
            prev[b] = g
        t += 1
    cap.release()
    out = []
    for idx, s, b in items:
        f = np.asarray(frac[tuple(b)][s:s + clip_len - 1])
        m = np.asarray(dmean[tuple(b)][s:s + clip_len - 1])
        if f.size == 0:
            out.append((idx, float('nan'), float('nan'), float('nan'), 0))
            continue
        third = float(np.sort(f)[-3]) if f.size >= 3 else float(f.max())
        out.append((idx, third, float(f.max()), float(m.mean()), int(f.size)))
    return mp4, t, out


def _scenario(file_name, domain):
    seg = file_name.split('_')
    code = seg[1][-1] if len(seg) > 1 and seg[1] else '?'
    return code if domain == 'rc' else ('A' if code == 'A' else 'S')


def _tile(mp4, s, bbox, clip_len, resize, text):
    """클립 가운데 프레임 | 클립 전체 최대 프레임 차이(히트맵 겹침) — 가로 2칸."""
    crop = _Crop(resize)
    cap = cv2.VideoCapture(mp4)
    cap.set(cv2.CAP_PROP_POS_FRAMES, s)
    frames = []
    for _ in range(clip_len):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(crop(f, bbox))
    cap.release()
    h, w = resize[1], resize[0]
    if len(frames) < 2:
        tile = np.zeros((h, w * 2, 3), np.uint8)
    else:
        gs = [_gray(c) for c in frames]
        dmax = np.max([cv2.absdiff(a, b) for a, b in zip(gs[1:], gs[:-1])], axis=0)
        heat = cv2.applyColorMap(np.clip(dmax.astype(np.int32) * 4, 0, 255).astype(np.uint8),
                                 cv2.COLORMAP_JET)
        mid = frames[len(frames) // 2]
        tile = np.hstack([mid, cv2.addWeighted(mid, 0.45, heat, 0.55, 0)])
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(tile, (0, 0), (tw + 8, th + 10), (0, 0, 0), -1)
    cv2.putText(tile, text, (4, th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def _montage(rows, path, clip_len, resize, cols=3):
    tiles = [_tile(r['mp4'], r['start_f'], r['bbox'], clip_len, resize,
                   f"{r['video'][:22]} f{r['start_f']} c{r['car_id']} {r['move']:.2f}%")
             for r in rows]
    if not tiles:
        return
    blank = np.zeros_like(tiles[0])
    while len(tiles) % cols:
        tiles.append(blank)
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(str(path), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    ap = argparse.ArgumentParser(description='비충돌 학습 클립의 움직임 양 측정')
    ap.add_argument('--data-dir', default=str(config.DATA_DIR), help='학습 데이터 폴더(기본 data/train)')
    ap.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 2),
                    help='동시에 읽을 영상 수(1080p 영상 하나당 약 0.2GB)')
    ap.add_argument('--limit', type=int, default=0, help='영상 N개만 목록 전체에서 고르게 골라 측정(시험용)')
    ap.add_argument('--montage', action='store_true', help='움직임 단계별 S 클립·움직임 적은 A 클립 예시 그림 저장')
    args = ap.parse_args()

    config.TRAIN_A_MODE = 'pre'          # 비율은 '충돌 1건 = 창 1개' 기준(span 이어도 무게 합 1)
    ds = HitAndRunDataset(data_dir=args.data_dir, augment=False)
    clip_len, resize = ds.clip_length, ds.resize
    rows = []
    for i, s in enumerate(ds.samples):
        dom = 'real' if 'realdata' in os.path.normpath(s['mp4_path']).split(os.sep) else 'rc'
        rows.append({'idx': i, 'mp4': s['mp4_path'], 'video': s['file_name'], 'domain': dom,
                     'scenario': _scenario(s['file_name'], dom), 'label': s['label'],
                     'car_id': s['car_id'], 'start_f': s['start_f'], 'fps': s['fps'],
                     'bbox': tuple(s['bbox'])})
    by_video = {}
    for r in rows:
        by_video.setdefault(r['mp4'], []).append((r['idx'], r['start_f'], r['bbox']))
    videos = sorted(by_video)
    if args.limit and args.limit < len(videos):
        pick = np.linspace(0, len(videos) - 1, args.limit).round().astype(int)
        videos = [videos[i] for i in sorted(set(pick))]
        keep = set(videos)
        rows = [r for r in rows if r['mp4'] in keep]
    print(f"[대상] 영상 {len(videos)}개 · 클립 {len(rows)}개 (A {sum(r['label'] for r in rows)}) "
          f"· 동시 처리 {args.workers}개 · 데이터 {args.data_dir}", flush=True)

    out_dir = Path(config._ROOT) / 'outputs' / 's_motion' / datetime.now().strftime('%y%m%d_%H%M')
    out_dir.mkdir(parents=True, exist_ok=True)
    t0, done, frames = time.time(), 0, 0
    tasks = [(v, by_video[v], clip_len, resize) for v in videos]
    res = {}
    with Pool(args.workers) as pool:
        for mp4, nf, out in pool.imap_unordered(_video_task, tasks):
            done += 1
            frames += nf
            for idx, move, move_max, dm, n in out:
                res[idx] = (move, move_max, dm, n)
            if done % 20 == 0 or done == len(tasks):
                el = time.time() - t0
                eta = el / done * (len(tasks) - done)
                print(f"  {done}/{len(tasks)} 영상 · {frames / max(el, 1e-6):.0f} 프레임/초 · "
                      f"경과 {el / 60:.1f}분 · 남은 약 {eta / 60:.1f}분", flush=True)
    for r in rows:
        r['move'], r['move_max'], r['diff_mean'], r['n_diffs'] = res.get(r['idx'], (np.nan,) * 3 + (0,))

    with open(out_dir / 'clips.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['video', 'domain', 'scenario', 'label', 'car_id', 'start_f', 'fps',
                    'move_pct', 'move_max_pct', 'diff_mean', 'n_diffs'])
        for r in rows:
            w.writerow([r['video'], r['domain'], r['scenario'], r['label'], r['car_id'], r['start_f'],
                        f"{r['fps']:.2f}", f"{r['move']:.4f}", f"{r['move_max']:.4f}",
                        f"{r['diff_mean']:.4f}", r['n_diffs']])

    lines = []

    def say(s=''):
        print(s)
        lines.append(s)

    ok_rows = [r for r in rows if r['n_diffs'] > 0]
    bad = len(rows) - len(ok_rows)
    A = [r for r in ok_rows if r['label'] == 1]
    S = [r for r in ok_rows if r['label'] == 0]
    nA, nS = len(A), len(S)

    def no_a(na, ns):
        return (ns / (na + ns)) ** BATCH if na + ns else float('nan')

    say(f"측정 {datetime.now():%Y-%m-%d %H:%M} · 영상 {len(videos)}개 · {time.time() - t0:.0f}초 · "
        f"움직인 화소 기준 밝기차 > {PIX_T} · 움직임 값 = 클립 안 3번째로 큰 프레임 값")
    if bad:
        say(f"⚠️ 읽지 못한 클립 {bad}개(영상 끝·디코딩 실패) — 계산에서 뺌")
    say(f"\n[지금] A {nA} : S {nS} = 1:{nS / max(nA, 1):.1f} · 묶음 {BATCH}개 중 충돌 없는 묶음 약 {no_a(nA, nS) * 100:.0f}%")
    for dom in ('rc', 'real'):
        a = sum(r['domain'] == dom for r in A)
        s = sum(r['domain'] == dom for r in S)
        if a or s:
            say(f"  {dom:<4}: A {a} : S {s} = 1:{s / max(a, 1):.1f}")

    say("\n[움직임 분포] 움직인 화소 % (p = 백분위)")
    say(f"  {'그룹':<12}{'클립':>7}{'p5':>8}{'p10':>8}{'p25':>8}{'p50':>8}{'p75':>8}{'p90':>8}")
    groups = [('A-rc', [r for r in A if r['domain'] == 'rc']), ('A-real', [r for r in A if r['domain'] == 'real'])]
    for sc, name in (('V', '주차'), ('S', '직진'), ('W', '배회')):
        groups.append((f'S-rc-{sc}({name})', [r for r in S if r['domain'] == 'rc' and r['scenario'] == sc]))
    groups.append(('S-real', [r for r in S if r['domain'] == 'real']))
    for name, g in groups:
        if not g:
            continue
        v = np.array([r['move'] for r in g])
        q = np.percentile(v, [5, 10, 25, 50, 75, 90])
        say(f"  {name:<12}{len(g):>7}" + "".join(f"{x:>8.2f}" for x in q))

    say("\n[기준별] 움직임 값이 기준 미만인 S = '빈 장면'. 빈 장면 S 를 일부만 남겼을 때 A:S")
    head = "".join(f"{'남김 ' + (f'{k:.0%}' if k else '0'):>12}" for k in KEEP_FRACS)
    say(f"  {'기준%':>6}{'빈 S':>14}{'빈 S(rc/real)':>16}{head}{'A 중 기준 미만':>16}")
    for th in THRESHOLDS:
        empty = [r for r in S if r['move'] < th]
        e_rc = sum(r['domain'] == 'rc' for r in empty)
        e_real = len(empty) - e_rc
        cells = ""
        for k in KEEP_FRACS:
            ns = nS - len(empty) + round(len(empty) * k)
            cells += f"{f'1:{ns / max(nA, 1):.1f} ({no_a(nA, ns) * 100:.0f}%)':>12}"
        a_low = sum(r['move'] < th for r in A)
        say(f"  {th:>6.2f}{f'{len(empty)} ({len(empty) / max(nS, 1):.0%})':>14}{f'{e_rc}/{e_real}':>16}"
            f"{cells}{f'{a_low} ({a_low / max(nA, 1):.0%})':>16}")
    say("  (괄호 안 % = 묶음 15개 중 충돌 없는 묶음 비율)")
    say("\n읽는 법: 'A 중 기준 미만'이 0에 가까운 기준까지만 쓴다 — 그보다 높이면 미세한 충돌과 비슷한")
    say("움직임의 S 까지 '빈 장면'으로 버리게 된다. 빈 장면 S 를 전부 버리면(남김 0) 서비스의 긴 정지")
    say("구간을 학습하지 못하니 일부는 남긴다.")
    (out_dir / 'report.txt').write_text("\n".join(lines) + "\n", encoding='utf-8')

    if args.montage:
        s_sorted = sorted(S, key=lambda r: r['move'])
        qs = np.linspace(0, len(s_sorted) - 1, 24).round().astype(int) if s_sorted else []
        _montage([s_sorted[i] for i in qs], out_dir / 'montage_S_by_motion.jpg', clip_len, resize)
        _montage(sorted(A, key=lambda r: r['move'])[:12], out_dir / 'montage_A_lowest.jpg', clip_len, resize)
        print("[그림] S 클립 움직임 낮은 순 24개 · 움직임 가장 적은 A 클립 12개 (왼쪽 가운데 프레임 | 오른쪽 움직인 곳)")
    print(f"\n[저장] {out_dir}")


if __name__ == '__main__':
    sys.exit(main())
