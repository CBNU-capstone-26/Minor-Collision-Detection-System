"""판정 기준·후처리·저fps 시간축 점검 — 창마다 P(A)를 한 번 재 두고, 서비스 판정을 여러 설정으로 다시 계산한다.

    python -u -m model.postproc_check --weights <가중치1.pth> [<가중치2.pth> ...]   # GPU 권장(가중치당 약 10분)
    python -u -m model.postproc_check --weights ... --no-val            # 평가 영상만
    python -u -m model.postproc_check --again outputs/postproc_check/<시각>   # 저장한 P(A)로 집계만 다시(모델 안 씀)

바꿔 보는 것 (모든 조합을 summary.csv 에 남긴다)
  · 판정 기준 A_THRESHOLD 0.2~0.7. 거친 탐색 기준은 config 기본식 min(0.3, 0.6×기준)
  · 후처리 단위 — 거친 탐색 간격(10)·최소 길이(35)는 프레임 수라 10fps 영상에선 1초·3.5초가 된다.
    'fps' 는 둘을 30fps 기준 길이로 보고 영상 fps 에 맞춰 줄인다(병합 간격은 이미 초 단위)
  · 최소 길이 35·25·15 (fps 맞춤이면 30fps 기준)
  · 거친 탐색 켬·끔(끄면 모든 창을 본다 — 거친 탐색이 놓치는 것)
  · 저fps(20 미만) 영상 시간축 — none 그대로 · repeat 프레임 반복 · blend 이웃 프레임 섞기. round(30/fps)배로 늘려
    창 하나(30프레임)가 학습 때처럼 약 1초가 되게 한다
영상: 평가 영상(config.EVAL_FOLDER_PATH) + 검증 실제 영상(첫 가중치 회차의 split.json 검증 중 realdata — 학습에 안 쓴 영상).
설정은 검증 영상으로 고르고 평가 영상으로 확인한다(평가 영상으로 고르면 점수가 부풀려짐). 사고는 구간 겹침으로 센다.
서비스 판정(거친 탐색 → 의심 창 주변 촘촘히 → 상태머신 → 병합·최소 길이)은 inference.predict_events_and_clips 의
이벤트 결정 부분을 옮겨 창별 P(A)로 다시 계산한다. 지금 설정 그대로인 경우는 실제 서비스 함수 결과와 영상별로
맞는지 확인한다(서비스 재현 확인 — GPU FP16 은 묶음 구성에 따라 P(A)가 1e-3 쯤 달라 기준 바로 옆 창은 갈릴 수 있다).
결과: outputs/postproc_check/<시각>/ — report.txt(화면과 같음) · summary.csv(설정 조합별 합계) · videos.csv(영상별 P(A)
요약·이벤트) · probs/<가중치>/<eval|val>_<none|repeat|blend>/<영상>.npz(창별 P(A) — 다시 집계·화면 확인용)
"""
import argparse
import contextlib
import csv
import io
import json
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import config
from device_utils import is_channels_last_3d_supported, is_cuda_like, predict_batch_size
from evaluate import _Tee, _read_label

L = config.CLIP_LENGTH
LOW_FPS = 20.0
RESAMPLES = ('none', 'repeat', 'blend')
THRESHOLDS = (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7)
MIN_SPANS = (35, 25, 15)
SETS = (('eval', '평가'), ('val', '검증'))


# ── 영상 목록 ──────────────────────────────────────────────────────────────

def _label(mp4):
    bboxes, sf, ef = _read_label(str(mp4.with_suffix('.txt')))
    if not bboxes:
        return None
    return bboxes.get(config.TARGET_ID, next(iter(bboxes.values()))), sf, ef


def _folder_videos(folder):
    out = []
    for mp4 in sorted(Path(folder).glob('*.mp4')):
        lab = _label(mp4) if mp4.with_suffix('.txt').exists() else None
        if lab:
            out.append((mp4, *lab))
    return out


def _split_path(weights, arg):
    """검증 영상 목록 — 지정한 split.json, 가중치 회차 폴더의 것, 없으면 가장 최근 학습의 것(분할은 seed 고정)."""
    if arg:
        return Path(arg)
    p = Path(weights).resolve().parent.parent / 'split.json'
    if p.exists():
        return p
    cands = sorted((Path(config._ROOT) / 'outputs' / 'trainlogs').glob('*/split.json'))
    return cands[-1] if cands else None


def _val_videos(split):
    with open(split, encoding='utf-8') as f:
        rels = json.load(f).get('val', [])
    out = []
    for rel in rels:
        rel = rel.replace('\\', '/')
        if not rel.startswith('realdata/'):
            continue
        mp4 = Path(config.DATA_DIR) / rel
        lab = _label(mp4) if mp4.exists() and mp4.with_suffix('.txt').exists() else None
        if lab:
            out.append((mp4, *lab))
    return out


# ── 창별 P(A) ─────────────────────────────────────────────────────────────

def _read_crops(mp4, bbox):
    """서비스와 같은 크롭(0번 차량 박스, r=config.R_VALUE, 224) — uint8 RGB 목록."""
    from inference import _crop_square_and_pad
    cap = cv2.VideoCapture(str(mp4))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        frames.append(_crop_square_and_pad(rgb, bbox, config.R_VALUE, config.RESIZE)[0])
    cap.release()
    return frames, fps, size


def _resample(frames, factor, mode):
    """시간축을 factor 배로 — repeat: 프레임 반복, blend: 다음 프레임과 섞어 사이를 채움."""
    if factor <= 1 or mode == 'none':
        return frames
    out = []
    for i, f in enumerate(frames):
        if mode == 'repeat':
            out += [f] * factor
            continue
        cur = f.astype(np.float32)
        nxt = frames[min(i + 1, len(frames) - 1)].astype(np.float32)
        for k in range(factor):
            a = k / factor
            out.append(np.clip((1 - a) * cur + a * nxt + 0.5, 0, 255).astype(np.uint8))
    return out


def _window_probs(runner, device, frames, batch):
    """간격 1 모든 창의 P(A). 서비스 _clip 과 같은 정규화. 영상을 장치에 한 번 올려 창마다 잘라 쓴다."""
    frames = list(frames)
    while len(frames) < L:                       # 짧은 영상은 마지막 프레임 반복(서비스와 같음)
        frames.append(frames[-1])
    u8 = torch.from_numpy(np.stack(frames)).to(device)          # T,H,W,3
    mean = torch.tensor(config.NORM_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor(config.NORM_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1, 1)
    cl = is_channels_last_3d_supported(device) and bool(config.USE_CHANNELS_LAST)
    nw = len(frames) - L + 1
    pa = np.zeros(nw, dtype=np.float32)
    with torch.inference_mode():
        for s in range(0, nw, batch):
            e = min(nw, s + batch)
            x = torch.stack([u8[w:w + L] for w in range(s, e)]).permute(0, 4, 1, 2, 3).float() / 255.0
            x = ((x - mean) / std).contiguous()
            if cl:
                x = x.contiguous(memory_format=torch.channels_last_3d)
            pa[s:e] = F.softmax(runner(x).float(), dim=1)[:, 1].cpu().numpy()
    del u8
    return pa


def _measure(weights, sets, out_dir, batch_arg):
    """가중치 하나로 영상 묶음마다 none·repeat·blend P(A)를 재서 npz 로 남긴다. 반환: (모델, 실행기 이름)."""
    from infer_runtime import get_runner
    from main import _load_model
    model, device = _load_model(weights)
    model.eval()
    if is_cuda_like(device):
        torch.backends.cudnn.benchmark = True      # 실행기를 고르기 전에 켠다(CLAUDE.md 참고)
    runner = get_runner(model)
    batch = batch_arg or predict_batch_size(device)
    wtag = _wtag(weights)
    print(f'\n[P(A) 재기] {wtag} · 장치 {device} · 실행기 {runner.name} · 묶음 {batch}', flush=True)
    t0 = time.perf_counter()
    for set_key, vids in sets.items():
        for i, (mp4, bbox, sf, ef) in enumerate(vids):
            crops, fps, size = _read_crops(mp4, bbox)
            if not crops:
                print(f'  [건너뜀] {mp4.name} — 프레임을 못 읽음')
                continue
            factor = int(round(30.0 / fps)) if fps < LOW_FPS else 1
            msg = []
            for mode in RESAMPLES:
                if mode != 'none' and factor <= 1:
                    continue
                d = out_dir / 'probs' / wtag / f'{set_key}_{mode}'
                d.mkdir(parents=True, exist_ok=True)
                pa = _window_probs(runner, device, _resample(crops, factor if mode != 'none' else 1, mode), batch)
                meta = dict(file=mp4.name, set=set_key, resample=mode, fps=fps,
                            factor=factor if mode != 'none' else 1, n_orig=len(crops),
                            width=size[0], height=size[1], bbox=[int(v) for v in bbox],
                            gt_sf=sf, gt_ef=ef, weights=str(weights))
                np.savez(d / f'{mp4.stem}.npz', pa=pa, meta=json.dumps(meta))
                msg.append(f'{mode} 최대 {pa.max():.2f}')
            print(f'  {set_key} {i + 1}/{len(vids)} {mp4.name} {fps:.1f}fps · ' + ' · '.join(msg)
                  + f' · {time.perf_counter() - t0:.0f}초', flush=True)
    return model, runner.name


def _wtag(weights):
    return Path(weights).stem.replace('hitandrun_x3d_', '').replace('hitandrun_', '')


def _load_probs(wdir, set_key, mode):
    """{영상이름: (pa, meta)}. repeat·blend 폴더엔 저fps 영상만 있어 나머지는 none 으로 채운다."""
    out = {}
    for m in ([mode, 'none'] if mode != 'none' else ['none']):
        for p in sorted((wdir / f'{set_key}_{m}').glob('*.npz')):
            if p.stem not in out:
                z = np.load(p)
                out[p.stem] = (z['pa'], json.loads(str(z['meta'])))
    return dict(sorted(out.items()))


# ── 서비스 판정 다시 계산 ────────────────────────────────────────────────────

def _events(pa, meta, t, *, coarse_prob=None, fps_scale=False, min_span=None, use_coarse=None):
    """inference.predict_events_and_clips 의 이벤트 결정(창 간격 1). 반환: [(시작, 끝)] 원본 프레임, 평가한 창 수."""
    coarse_step = config.PREDICT_COARSE_STEP
    coarse_pad = config.PREDICT_COARSE_PAD
    min_span = config.PREDICT_MIN_EVENT_SPAN_FRAMES if min_span is None else min_span
    use_coarse = config.PREDICT_USE_COARSE_SCAN if use_coarse is None else use_coarse
    if coarse_prob is None:
        coarse_prob = min(0.3, 0.6 * t)
    f = meta['factor']
    fps = meta['fps'] * f
    if fps_scale:
        s = fps / 30.0
        coarse_step = max(1, int(round(coarse_step * s)))
        min_span = max(1, int(round(min_span * s)))
    nw = len(pa)
    last = nw - 1
    n_real = meta['n_orig'] * f
    grid = max(1, int(round(coarse_step))) if use_coarse else 1
    if grid > 1:
        ev = np.zeros(nw, dtype=bool)
        ev[::grid] = True
        ev[last] = True                          # 영상 끝 창이 격자에 없으면 거친 창으로 붙인다
        hot = np.nonzero(ev & (pa > coarse_prob))[0]
        if len(hot):                             # 의심 창 주변 ±pad 를 모두 본다
            cover = np.zeros(nw + 1, dtype=np.int32)
            np.add.at(cover, np.maximum(0, hot - coarse_pad), 1)
            np.add.at(cover, np.minimum(nw, hot + coarse_pad + 1), -1)
            ev |= np.cumsum(cover[:nw]) > 0
    else:
        ev = np.ones(nw, dtype=bool)
    pred = ev & (pa > t)                         # 평가 안 한 창은 비사고
    d = np.diff(np.concatenate(([0], pred.astype(np.int8), [0])))
    starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
    events = [[int(s), int(e) + L - 2 if e < nw else n_real - 1] for s, e in zip(starts, ends)]
    gap = int(round(fps * config.PREDICT_EVENT_MERGE_GAP_SEC))
    if events:
        merged = [events[0]]
        for e in events[1:]:
            if e[0] - merged[-1][1] <= gap:
                merged[-1][1] = max(merged[-1][1], e[1])
            else:
                merged.append(e)
        events = [e for e in merged if e[1] - e[0] >= min_span]
    return [(s // f, e // f) for s, e in events], int(ev.sum())


def _score(data, t, **kw):
    """사고는 구간 겹침으로 잡았는지, 오탐 이벤트 = 정상 영상의 이벤트 + 사고 영상에서 충돌과 안 겹친 이벤트."""
    rows = []
    for name, (pa, m) in data.items():
        evs, _ = _events(pa, m, t, **kw)
        if m['gt_sf'] is not None:
            ov = [e for e in evs if e[0] <= m['gt_ef'] and e[1] >= m['gt_sf']]
            caught, false_ev = bool(ov), len(evs) - len(ov)
        else:
            caught, false_ev = None, len(evs)
        rows.append(dict(name=name, gt='A' if m['gt_sf'] is not None else 'S', events=evs,
                         caught=caught, false_ev=false_ev, base_a=bool((pa > t).any())))
    A = [r for r in rows if r['gt'] == 'A']
    S = [r for r in rows if r['gt'] == 'S']
    miss = [r['name'] for r in A if not r['caught']]
    fa = [r['name'] for r in S if r['events']]
    return dict(n=len(rows), n_A=len(A), n_S=len(S), miss=len(miss), fa=len(fa),
                false_ev=sum(r['false_ev'] for r in rows),
                acc=(len(A) - len(miss) + len(S) - len(fa)) / max(1, len(rows)),
                base_miss=sum(not r['base_a'] for r in A), base_fa=sum(r['base_a'] for r in S),
                miss_list=miss, fa_list=fa), rows


def _short(name):
    """260828_TA_0001 → TA_0001."""
    return name.split('_', 1)[1] if '_' in name else name


# ── 서비스 재현 확인 ────────────────────────────────────────────────────────

def _service_check(model, wtag, sets, probs_root):
    """지금 설정 그대로의 다시 계산이 실제 서비스 함수(predict_events_and_clips)와 같은지 영상별로."""
    from inference import predict_events_and_clips
    tmp = Path(tempfile.mkdtemp(prefix='postproc_check_'))
    print(f'\n[서비스 재현 확인] {wtag} — 지금 설정(기준 {config.A_THRESHOLD:g}, 거친 탐색 {config.PREDICT_COARSE_PROB:g}, '
          f'최소 길이 {config.PREDICT_MIN_EVENT_SPAN_FRAMES})으로 실제 서비스 함수와 비교')
    bad = total = 0
    for set_key, vids in sets.items():
        data = _load_probs(probs_root / wtag, set_key, 'none')
        for mp4, bbox, _, _ in vids:
            if mp4.stem not in data:
                continue
            pa, meta = data[mp4.stem]
            mine, _ = _events(pa, meta, config.A_THRESHOLD, coarse_prob=config.PREDICT_COARSE_PROB)
            with contextlib.redirect_stdout(io.StringIO()):
                evs = predict_events_and_clips(model, video_path=str(mp4), bbox=tuple(bbox), output_dir=tmp,
                                               render_clips=False)
            svc = [(e['event_start_frame'], e['event_end_frame']) for e in evs]
            total += 1
            if svc != mine:
                bad += 1
                near = float(np.abs(pa - config.A_THRESHOLD).min())
                print(f'  다름 {set_key} {mp4.name}: 서비스 {svc} · 다시 계산 {mine} '
                      f'(기준과 가장 가까운 창 P(A) 차이 {near:.4f})')
    print(f'  → {total}개 중 {total - bad}개 같음' + ('' if not bad else
          ' — 다른 영상은 기준 바로 옆 창이 FP16 묶음 차이로 갈린 것인지 위 P(A) 차이로 확인'))
    return bad, total


# ── 집계·보고 ───────────────────────────────────────────────────────────────

def _grid():
    for mode in RESAMPLES:
        for post in ('frame', 'fps'):
            for ms in MIN_SPANS:
                for coarse in (True, False):
                    for t in THRESHOLDS:
                        yield mode, post, ms, coarse, t


def _cfg_name(mode, post, ms, coarse, t):
    parts = [f'기준 {t:g}']
    if post == 'fps':
        parts.append('fps맞춤')
    if ms != config.PREDICT_MIN_EVENT_SPAN_FRAMES:
        parts.append(f'최소 {ms}')
    if not coarse:
        parts.append('거친탐색 끔')
    if mode != 'none':
        parts.append(f'시간축 {mode}')
    return ' + '.join(parts)


def _fmt(tot):
    return (f"놓침 {tot['miss']:>2}/{tot['n_A']} · 오탐영상 {tot['fa']:>2}/{tot['n_S']} · 오탐이벤트 {tot['false_ev']:>2} · "
            f"정확도 {tot['acc'] * 100:5.1f}% (①놓침 {tot['base_miss']}·오탐 {tot['base_fa']})")


def _lists(tot):
    return f"놓침 {' '.join(_short(x) for x in tot['miss_list']) or '-'} | 오탐 {' '.join(_short(x) for x in tot['fa_list']) or '-'}"


def _aggregate(out_dir, probs_root, suffix=''):
    wdirs = sorted(p for p in probs_root.iterdir() if p.is_dir())
    summary, results, caches = [], {}, {}
    for wdir in wdirs:
        cache = {}
        for set_key, _ in SETS:
            for mode in RESAMPLES:
                cache[(set_key, mode)] = _load_probs(wdir, set_key, mode)
        for mode, post, ms, coarse, t in _grid():
            for set_key, _ in SETS:
                data = cache[(set_key, mode)]
                if not data:
                    continue
                if mode != 'none' and not any(m['factor'] > 1 for _, m in data.values()):
                    continue
                tot, _ = _score(data, t, fps_scale=post == 'fps', min_span=ms, use_coarse=coarse)
                key = (wdir.name, mode, post, ms, coarse, t)
                results.setdefault(key, {})[set_key] = tot
                summary.append(dict(weights=wdir.name, set=set_key, resample=mode, postproc=post, min_span=ms,
                                    coarse=int(coarse), threshold=t, n=tot['n'], n_A=tot['n_A'], n_S=tot['n_S'],
                                    miss=tot['miss'], fa_videos=tot['fa'], false_events=tot['false_ev'],
                                    accuracy=round(tot['acc'], 4), base_miss=tot['base_miss'], base_fa=tot['base_fa'],
                                    miss_list=' '.join(_short(x) for x in tot['miss_list']),
                                    fa_list=' '.join(_short(x) for x in tot['fa_list'])))
        _videos_csv(out_dir / f'videos{suffix}.csv', wdir, cache, write_header=wdir == wdirs[0])
        caches[wdir.name] = cache
    if summary:
        with open(out_dir / f'summary{suffix}.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)
    _report(results, [w.name for w in wdirs])
    for wtag, cache in caches.items():
        _wrong_now(wtag, cache)


def _videos_csv(path, wdir, cache, write_header):
    """영상별 — P(A) 최대·위치, 충돌 구간 창/바깥 창 최대, 지금 설정·fps 맞춤 후처리의 이벤트."""
    cols = ['weights', 'set', 'resample', 'file', 'gt', 'fps', 'factor', 'width', 'height', 'box_w', 'box_h',
            'gt_start', 'gt_end', 'n_windows', 'p_max', 'p_max_frame', 'p_max_gt', 'p_max_outside',
            'events_now', 'caught_now', 'events_fps', 'caught_fps']
    with open(path, 'w' if write_header else 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(cols)
        for (set_key, mode), data in cache.items():
            for name, (pa, m) in data.items():
                if mode != 'none' and m['resample'] == 'none':
                    continue                         # repeat·blend 묶음에 채워 넣은 원본 영상은 none 줄과 같다
                f_ = m['factor']
                ws = np.arange(len(pa))
                lo, hi = ws // f_, (ws + L - 1) // f_    # 창이 덮는 원본 프레임
                if m['gt_sf'] is not None:
                    in_gt = (lo <= m['gt_ef']) & (hi >= m['gt_sf'])
                    p_gt = f'{pa[in_gt].max():.4f}' if in_gt.any() else ''
                    p_out = f'{pa[~in_gt].max():.4f}' if (~in_gt).any() else ''
                else:
                    p_gt, p_out = '', f'{pa.max():.4f}'
                row = [wdir.name, set_key, mode, m['file'], 'A' if m['gt_sf'] is not None else 'S',
                       round(m['fps'], 2), f_, m['width'], m['height'], m['bbox'][2] - m['bbox'][0],
                       m['bbox'][3] - m['bbox'][1], m['gt_sf'], m['gt_ef'], len(pa), f'{pa.max():.4f}',
                       int(pa.argmax()) // f_, p_gt, p_out]
                for fps_scale in (False, True):
                    evs, _ = _events(pa, m, config.A_THRESHOLD, coarse_prob=config.PREDICT_COARSE_PROB,
                                     fps_scale=fps_scale)
                    caught = ('' if m['gt_sf'] is None else
                              int(any(e[0] <= m['gt_ef'] and e[1] >= m['gt_sf'] for e in evs)))
                    row += [' '.join(f'{s}-{e}' for s, e in evs), caught]
                w.writerow(row)


def _wrong_now(wtag, cache):
    """지금 설정에서 틀린 영상 — 모델이 못 본 것(P(A) 낮음)인지 후처리가 지운 것인지 가르려고 P(A) 최대를 함께."""
    print(f'\n[{wtag}] 지금 설정(기준 {config.A_THRESHOLD:g})에서 틀린 영상 — fps · 박스 · P(A) 최대(사고는 충돌 구간 창 / 그 밖 창)')
    for set_key, set_name in SETS:
        data = cache.get((set_key, 'none')) or {}
        if not data:
            continue
        _, rows = _score(data, config.A_THRESHOLD, coarse_prob=config.PREDICT_COARSE_PROB)
        wrong = [r for r in rows if (r['gt'] == 'A' and not r['caught']) or (r['gt'] == 'S' and r['events'])]
        for r in wrong:
            pa, m = data[r['name']]
            box = f"{m['bbox'][2] - m['bbox'][0]}x{m['bbox'][3] - m['bbox'][1]}"
            if r['gt'] == 'A':
                ws = np.arange(len(pa))
                in_gt = (ws <= m['gt_ef']) & (ws + L - 1 >= m['gt_sf'])
                p_gt = pa[in_gt].max() if in_gt.any() else float('nan')
                p_out = pa[~in_gt].max() if (~in_gt).any() else float('nan')
                why = '모델이 못 봄' if p_gt <= config.A_THRESHOLD else '모델은 반응 — 후처리·거친 탐색에서 빠짐'
                print(f'  {set_name} 놓침 {_short(r["name"]):<9} {m["fps"]:5.1f}fps 박스 {box:<8} 충돌 구간 최대 {p_gt:.2f} '
                      f'· 그 밖 {p_out:.2f} · 이벤트 {r["events"] or "-"} — {why}')
            else:
                print(f'  {set_name} 오탐 {_short(r["name"]):<9} {m["fps"]:5.1f}fps 박스 {box:<8} 최대 {pa.max():.2f}'
                      f'(창 {int(pa.argmax())}) · 이벤트 {r["events"]}')


def _report(results, wtags):
    now = (config.PREDICT_MIN_EVENT_SPAN_FRAMES, True)
    for wtag in wtags:
        print(f'\n{"=" * 100}\n[{wtag}]\n{"=" * 100}')
        for set_key, set_name in SETS:
            rows = [(k, v[set_key]) for k, v in results.items() if k[0] == wtag and set_key in v]
            if not rows:
                continue
            n = rows[0][1]
            print(f'\n── {set_name} 영상 {n["n"]}개 (사고 {n["n_A"]} · 정상 {n["n_S"]}) ──')
            show = []
            for t in (0.3, 0.4, 0.5, 0.6, 0.7):
                show.append(('none', 'frame', *now, t))
            for t in (0.4, 0.5, 0.6):
                show.append(('none', 'fps', *now, t))
            for ms in (25, 15):
                show.append(('none', 'fps', ms, True, 0.5))
            show.append(('none', 'frame', now[0], False, 0.5))
            show.append(('none', 'fps', now[0], False, 0.5))
            for mode in ('repeat', 'blend'):
                for t in (0.4, 0.5):
                    show.append((mode, 'frame', *now, t))
                show.append((mode, 'fps', *now, 0.5))
            for cfg in show:
                tot = results.get((wtag, *cfg), {}).get(set_key)
                if tot:
                    print(f'  {_cfg_name(*cfg):<38} {_fmt(tot)}  {_lists(tot)}')
        # 검증 영상으로 고른 설정 → 평가 영상에서 확인
        both = [(k, v) for k, v in results.items() if k[0] == wtag and 'val' in v and 'eval' in v]
        if not both:
            continue
        cur = results.get((wtag, 'none', 'frame', *now, config.A_THRESHOLD), {})
        print(f'\n── 검증 영상으로 고른 설정 상위 8 (놓침 → 오탐영상 → 오탐이벤트 순, 같으면 지금 설정에 가까운 쪽) → 평가 영상 결과 ──')
        if cur:
            print(f'  {"지금 설정":<38} 검증 {_fmt(cur["val"])}\n  {"":<38} 평가 {_fmt(cur["eval"])}')

        def _rank(kv):
            k, v = kv
            t = v['val']
            change = (k[1] != 'none') + (k[2] != 'frame') + (k[3] != now[0]) + (not k[4]) + abs(k[5] - config.A_THRESHOLD)
            return (t['miss'], t['fa'], t['false_ev'], change)
        for k, v in sorted(both, key=_rank)[:8]:
            print(f'  {_cfg_name(*k[1:]):<38} 검증 {_fmt(v["val"])}\n  {"":<38} 평가 {_fmt(v["eval"])}  {_lists(v["eval"])}')
    print('\n· 정확도 = (구간이 겹친 사고 영상 + 이벤트 없는 정상 영상) / 전체. ①은 창 하나라도 기준을 넘으면 사고로 본 판정.')
    print('· 모든 조합은 summary.csv, 영상별 P(A)·이벤트는 videos.csv, 창별 P(A)는 probs/ 에 있다.')


# ── 실행 ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='판정 기준·후처리·저fps 시간축 점검')
    ap.add_argument('--weights', nargs='+', help='점검할 가중치(여러 개 가능)')
    ap.add_argument('--folder', default=str(config.EVAL_FOLDER_PATH), help='평가 영상 폴더(mp4 + txt)')
    ap.add_argument('--split', help='검증 영상 목록 split.json(기본: 첫 가중치 회차 폴더의 것)')
    ap.add_argument('--no-val', action='store_true', help='검증 실제 영상은 빼고 평가 영상만')
    ap.add_argument('--no-service-check', action='store_true', help='실제 서비스 함수와의 재현 확인을 건너뜀')
    ap.add_argument('--batch', type=int, default=0, help='추론 묶음(0 = 서비스와 같게 GPU 8 · CPU 1)')
    ap.add_argument('--again', help='이전 결과 폴더 — 저장한 P(A)로 집계만 다시(모델 안 씀)')
    ap.add_argument('--videos', type=int, default=0, help='묶음마다 사고·정상 절반씩 이 수만(0 = 전부, 코드 점검용)')
    args = ap.parse_args()
    if not args.weights and not args.again:
        ap.error('--weights 또는 --again 이 필요합니다')
    if int(config.PREDICT_WINDOW_STRIDE) != 1:
        print(f'⚠️ PREDICT_WINDOW_STRIDE={config.PREDICT_WINDOW_STRIDE} — 이 점검은 간격 1 기준으로 다시 계산한다')

    if args.again:
        out_dir = Path(args.again)
        suffix = '_again_' + datetime.now().strftime('%y%m%d_%H%M%S')
    else:
        out_dir = Path(config._ROOT) / 'outputs' / 'postproc_check' / datetime.now().strftime('%y%m%d_%H%M%S')
        suffix = ''
    out_dir.mkdir(parents=True, exist_ok=True)
    tee = _Tee(out_dir / f'report{suffix}.txt')
    sys.stdout = tee
    try:
        return _run(args, out_dir, suffix)
    finally:
        sys.stdout = tee.stdout
        tee.close()
        print(f'\n[저장] {out_dir}')


def _run(args, out_dir, suffix):
    print(f'[판정 기준·후처리 점검] {datetime.now():%Y-%m-%d %H:%M}')
    print(f'지금 설정: 판정 기준 {config.A_THRESHOLD:g} · 거친 탐색 간격 {config.PREDICT_COARSE_STEP}·기준 '
          f'{config.PREDICT_COARSE_PROB:g}·주변 ±{config.PREDICT_COARSE_PAD} · 병합 {config.PREDICT_EVENT_MERGE_GAP_SEC:g}초 · '
          f'최소 길이 {config.PREDICT_MIN_EVENT_SPAN_FRAMES}프레임 · r {config.R_VALUE:g}')
    probs_root = out_dir / 'probs'
    if not args.again:
        sets = {'eval': _folder_videos(args.folder)}
        print(f'평가 영상: {args.folder} — {len(sets["eval"])}개')
        if not args.no_val:
            split = _split_path(args.weights[0], args.split)
            if split is None:
                print('검증 영상: split.json 을 못 찾아 건너뜀(--split 으로 지정)')
            else:
                sets['val'] = _val_videos(split)
                print(f'검증 영상: {split} 의 검증 중 실제 영상 {len(sets["val"])}개 (학습에 안 쓴 영상)')
        if args.videos:
            for k, vids in sets.items():
                sets[k] = ([v for v in vids if v[2] is not None][:(args.videos + 1) // 2]
                           + [v for v in vids if v[2] is None][:args.videos // 2])
            print(f'코드 점검: 묶음마다 {args.videos}개만')
        print('가중치: ' + ' · '.join(str(w) for w in args.weights))
        t0 = time.perf_counter()
        for weights in args.weights:
            model, _ = _measure(weights, sets, out_dir, args.batch)
            if not args.no_service_check:
                _service_check(model, _wtag(weights), sets, probs_root)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(f'\nP(A) 재기 끝 — {time.perf_counter() - t0:.0f}초')
    _aggregate(out_dir, probs_root, suffix)
    return 0


if __name__ == '__main__':
    sys.exit(main())
