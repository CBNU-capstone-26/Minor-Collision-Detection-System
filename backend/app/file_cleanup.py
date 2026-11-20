"""Durable, retryable file deletion after database changes have committed."""
import logging

from app import db_models
from app.object_storage import get_storage
from app.settings import settings
from app.thumbnails import delete_thumbnail_cache

logger = logging.getLogger(__name__)


def queue_cleanup(db, object_key, guard_key, guard_kind, cleanup_kind='object'):
    db.add(db_models.FileCleanup(object_key=object_key, guard_key=guard_key,
                                 guard_kind=guard_kind, cleanup_kind=cleanup_kind))


def process_pending_cleanup(db, storage=None, *, limit=20):
    """Remove unreferenced files; keep failed items for a later retry."""
    pending = (db.query(db_models.FileCleanup)
               .order_by(db_models.FileCleanup.attempts, db_models.FileCleanup.id)
               .with_for_update(skip_locked=True).limit(limit).all())
    for item in pending:
        if item.guard_kind == 'video':
            referenced = db.query(db_models.Video.id).filter_by(video_path=item.guard_key).first()
        elif item.guard_kind == 'clip':
            referenced = db.query(db_models.CrashEvent.id).filter_by(clip_path=item.guard_key).first()
        else:
            logger.error('Unknown file cleanup reference type')
            continue
        if referenced:
            db.delete(item)
            continue
        try:
            if item.cleanup_kind == 'object':
                if storage is None:
                    storage = get_storage()
                storage.delete(item.object_key)
            elif item.cleanup_kind == 'cache':
                delete_thumbnail_cache(item.object_key, settings.THUMBNAIL_DIR)
            else:
                logger.error('Unknown file cleanup type')
                continue
        except Exception as exc:
            item.attempts += 1
            logger.warning('File cleanup failed; will retry: %s', type(exc).__name__)
            continue
        db.delete(item)
    db.commit()


def try_pending_cleanup(db, storage=None):
    try:
        process_pending_cleanup(db, storage)
    except Exception as exc:
        db.rollback()
        logger.warning('File cleanup queue unavailable; will retry: %s', type(exc).__name__)
