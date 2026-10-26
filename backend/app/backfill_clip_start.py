"""사고 클립이 원본의 몇 초부터인지(crash_events.clip_start_sec)가 비어 있는 구간을 채운다 — 한 번 돌리는 정리용.

재생 화면은 이 값으로 빨간 구간에 CAM 클립을 원본 시각에 맞춰 겹친다. 이 값을 저장하기 시작한 2026-10-27 전 분석은
비어 있는데, 그때 클립 앞뒤 여유가 시기마다 달라(15·30프레임) 구간 시각으로 역산하면 0.5초까지 어긋난다(18번 영상 실측).
그래서 클립 프레임을 원본 프레임과 직접 맞춰 본다: 클립이 구간을 담는 범위의 시작 위치 후보마다 화면 차이를 재고
가장 작은 곳을 고른다. 합성 없는 원본 클립(_raw)이 있으면 그것으로 화면 전체를, 없으면 표시 클립에서 박스·히트맵이
그려진 bbox 부분을 빼고 비교한다(클립 아래 S/A 띠는 늘 뺀다). 2등과 차이가 작으면 확실하지 않아 건너뛴다 —
움직임이 bbox 안에만 있는 영상은 bbox 를 빼면 위치를 가를 수 없다.

    cd backend && ../venv311/bin/python -m app.backfill_clip_start            # 찾기만(바꾸지 않음)
    cd backend && ../venv311/bin/python -m app.backfill_clip_start --apply    # 비어 있는 값만 채운다
"""
import sys

import cv2
import numpy as np

from app import db_models
from app.db_connection import SessionLocal
from app.routers.videos import raw_clip_path
from app.settings import settings

SMALL = (160, 90)      # 비교용 축소 크기
MIN_RATIO = 1.3        # 2등(3프레임 이상 떨어진 위치) 차이 ÷ 1등 차이 — 이보다 작으면 확실하지 않아 건너뛴다


def _gray(frame, mask):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return cv2.resize(g * mask, SMALL, interpolation=cv2.INTER_AREA)


def find_clip_start(video_path, clip_path, width, height, bbox, first, last, mask_bbox=True):
    """클립 첫 프레임 = 원본의 몇 번째 프레임인지 → (프레임, 2등÷1등, fps). 못 찾으면 (None, 이유, fps).
    mask_bbox: bbox 부분을 빼고 비교(박스·히트맵이 그려진 표시 클립). 원본 클립(_raw)은 False."""
    cap = cv2.VideoCapture(str(clip_path))
    clip = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        clip.append(f[:height])                    # 아래 S/A 띠 빼기
    cap.release()
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0        # 분석(inference.py)과 같은 fps
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = len(clip)
    if n == 0:
        cap.release()
        return None, '클립을 못 읽음', fps
    # 클립은 구간(first~last 프레임)을 담는다 → 시작 후보는 last-n+1 ~ first, 영상 안
    lo, hi = max(0, last - n + 1), min(first, max(0, total - n))
    if lo > hi:
        lo, hi = max(0, min(lo, hi) - 30), max(lo, hi) + 30
    mask = np.ones((height, width), np.float32)
    if mask_bbox:
        x1, y1, x2, y2 = bbox
        m = max(6, int(0.05 * max(x2 - x1, y2 - y1)))
        mask[max(0, y1 - m):y2 + m, max(0, x1 - m):x2 + m] = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    orig = []
    for _ in range(hi - lo + n):
        ok, f = cap.read()
        if not ok:
            break
        orig.append(_gray(f, mask))
    cap.release()
    small = [_gray(f, mask) for f in clip]
    ks = range(0, n, 2)
    errs = {}
    for s in range(lo, hi + 1):
        if s - lo + n > len(orig):
            break
        errs[s] = float(np.mean([np.mean((small[k] - orig[s - lo + k]) ** 2) for k in ks]))
    if not errs:
        return None, '원본에서 클립 범위를 못 읽음', fps
    best = min(errs, key=errs.get)
    # 2등은 3프레임 이상 떨어진 위치에서 — 바로 옆 프레임끼리 비긴 건 같은 프레임이 반복되는 영상이라(37번 영상:
    # 연속 프레임의 49% 가 같음) 어느 쪽이든 화면이 같다. 멀리까지 비슷하면(정지 화면) 정말 가를 수 없다
    far = [e for s, e in errs.items() if abs(s - best) >= 3]
    if not far:
        return best, float('inf'), fps
    ratio = min(far) / max(errs[best], 1e-6)
    return (best, ratio, fps) if ratio >= MIN_RATIO else (None, f'확실하지 않음(2등÷1등 {ratio:.2f})', fps)


def main(apply=False):
    db = SessionLocal()
    filled = skipped = 0
    try:
        events = (db.query(db_models.CrashEvent)
                  .filter(db_models.CrashEvent.clip_start_sec.is_(None),
                          db_models.CrashEvent.clip_path.isnot(None))
                  .order_by(db_models.CrashEvent.id).all())
        print(f"시작 시각이 비어 있는 사고 구간 {len(events)}개 ({'채움' if apply else '찾기만 — 채우려면 --apply'})")
        for ev in events:
            video, task = db.get(db_models.Video, ev.video_id), ev.task
            clip_path = settings.abs_path(ev.clip_path)
            if not clip_path.exists():
                print(f"  구간 {ev.id}(영상 {ev.video_id}): 건너뜀 — 클립 파일 없음")
                skipped += 1
                continue
            raw = raw_clip_path(ev)
            # 원본 클립이 있으면 그것으로 화면 전체를(충돌 움직임이 있는 bbox 안까지), 없으면 표시 클립에서 bbox 를 빼고
            attempts = ([(raw, False, '원본 클립')] if raw is not None and raw.exists() else []) + [(clip_path, True, '표시 클립')]
            for path, mask_bbox, label in attempts:
                start, ratio, fps = find_clip_start(
                    settings.abs_path(video.video_path), path, video.width, video.height,
                    (task.bbox_xmin, task.bbox_ymin, task.bbox_xmax, task.bbox_ymax),
                    ev.frame_number, ev.end_frame_number if ev.end_frame_number is not None else ev.frame_number,
                    mask_bbox=mask_bbox)
                if start is not None:
                    break
            if start is None:
                print(f"  구간 {ev.id}(영상 {ev.video_id}): 건너뜀 — {ratio}")
                skipped += 1
                continue
            print(f"  구간 {ev.id}(영상 {ev.video_id}): 클립 시작 {start}프레임 = {start / fps:.3f}초 "
                  f"({label}로 비교, 2등÷1등 {ratio:.1f})")
            if apply:
                ev.clip_start_sec = start / fps
                db.commit()
            filled += 1
    finally:
        db.close()
    print(f"{'채움' if apply else '찾음'} {filled}개 · 건너뜀 {skipped}개")


if __name__ == '__main__':
    main(apply='--apply' in sys.argv[1:])
