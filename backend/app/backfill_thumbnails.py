"""Prepare previews for existing videos: python -B -m app.backfill_thumbnails."""
from app.db_connection import SessionLocal
from app.db_models import Video
from app.object_storage import get_storage
from app.settings import settings
from app.thumbnails import ensure_thumbnail


def main():
    with SessionLocal() as db:
        videos = db.query(Video.id, Video.video_path).order_by(Video.id).all()
    storage = get_storage()
    failures = 0
    for video in videos:
        try:
            path = ensure_thumbnail(storage, video.video_path, settings.THUMBNAIL_DIR)
            print(f"video={video.id} preview_bytes={path.stat().st_size}", flush=True)
        except Exception as exc:
            failures += 1
            print(f"video={video.id} failed={type(exc).__name__}", flush=True)
    print(f"prepared={len(videos) - failures} failed={failures}", flush=True)
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
