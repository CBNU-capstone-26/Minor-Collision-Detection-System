"""영상 보관 기간을 관리합니다.

업로드한 지 VIDEO_RETENTION_DAYS(기본 90일)가 지난 영상과 거기에 딸린 사고 클립, 원본 클립, 썸네일,
분석 기록, 업로드 원본을 모두 지웁니다. 웹 서버가 켜질 때 한 번 돌고, 그 뒤로는 CHECK_INTERVAL_SEC마다 돕니다.
분석이 대기 중이거나 진행 중인 영상은 건너뛰고, 분석이 끝난 뒤 다음 정리 때 지웁니다.
"""
from datetime import datetime, timedelta

from app import db_models
from app.db_connection import SessionLocal
from app.routers.analysis import ACTIVE_STATUSES
from app.routers.videos import purge_video
from app.settings import settings

CHECK_INTERVAL_SEC = 60 * 60   # 1시간마다 정리합니다.


def expired_videos(db):
    """업로드 시각 기준으로 보관 기간이 지난 영상을 돌려줍니다. 보관 기간을 끄면 빈 목록입니다."""
    if settings.VIDEO_RETENTION_DAYS <= 0:
        return []
    cutoff = datetime.utcnow() - timedelta(days=settings.VIDEO_RETENTION_DAYS)
    return (db.query(db_models.Video)
            .filter(db_models.Video.created_at < cutoff)
            .order_by(db_models.Video.id).all())


def delete_expired_videos() -> list[int]:
    """기간이 지난 영상을 지우고 지운 영상 번호를 돌려줍니다.
    영상마다 따로 commit하므로 중간에 실패해도 앞에서 지운 영상은 그대로 지워집니다.
    """
    db = SessionLocal()
    deleted = []
    try:
        for video in expired_videos(db):
            if any(t.status in ACTIVE_STATUSES for t in video.analysis_tasks):
                continue   # 분석이 끝나면 다음 정리 때 지웁니다.
            video_id = video.id
            purge_video(video, db)
            db.commit()
            deleted.append(video_id)
    finally:
        db.close()
    return deleted
