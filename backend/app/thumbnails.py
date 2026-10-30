"""Small JPEG previews cached on disk behind the authenticated media API."""
import logging
import os
import tempfile
import threading
from pathlib import Path

from PIL import Image

from app.object_storage import materialize

PREVIEW_VERSION = "preview-v1"
PREVIEW_SIZE = (640, 360)
_locks = [threading.Lock() for _ in range(64)]
logger = logging.getLogger(__name__)


def thumbnail_key(video_key: str) -> str:
    return f"thumbnails/{PREVIEW_VERSION}/{Path(video_key).stem}.jpg"


def cached_thumbnail_path(video_key: str, cache_root: Path) -> Path:
    return Path(cache_root) / PREVIEW_VERSION / f"{Path(video_key).stem}.jpg"


def _save_image(image, destination):
    image = image.convert("RGB")
    image.thumbnail(PREVIEW_SIZE, Image.Resampling.LANCZOS)
    image.save(destination, "JPEG", quality=80, optimize=True, progressive=True)


def _from_video(source, destination):
    import cv2

    cap = cv2.VideoCapture(str(source))
    try:
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        raise ValueError("Cannot read the first video frame")
    with Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) as image:
        _save_image(image, destination)


def ensure_thumbnail(storage, video_key: str, cache_root: Path,
                     *, source_video: Path | None = None) -> Path:
    """Reuse a local preview; upgrade legacy JPEGs before reading a whole video."""
    target = cached_thumbnail_path(video_key, cache_root)
    lock = _locks[hash(str(target)) % len(_locks)]
    with lock:
        if target.is_file():
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        key = thumbnail_key(video_key)
        legacy_key = f"thumbnails/{Path(video_key).stem}.jpg"
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".jpg", delete=False) as tmp:
            temporary = Path(tmp.name)
        try:
            generated = True
            if source_video is not None:
                _from_video(source_video, temporary)
            elif storage.exists(key):
                storage.download_to_path(key, temporary)
                generated = False
            elif storage.exists(legacy_key):
                with materialize(storage, legacy_key) as source:
                    with Image.open(source) as image:
                        _save_image(image, temporary)
            else:
                if not storage.exists(video_key):
                    raise FileNotFoundError(video_key)
                with materialize(storage, video_key, suffix=Path(video_key).suffix) as source:
                    _from_video(source, temporary)
            if generated:
                try:
                    storage.upload_path(temporary, key, content_type="image/jpeg",
                                        cache_control="private, max-age=86400")
                except Exception as exc:
                    # The locally generated preview can still be served if R2 is unavailable.
                    logger.warning("Could not persist thumbnail: %s", type(exc).__name__)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    return target


def delete_thumbnail_cache(video_key: str, cache_root: Path):
    target = cached_thumbnail_path(video_key, cache_root)
    with _locks[hash(str(target)) % len(_locks)]:
        target.unlink(missing_ok=True)
