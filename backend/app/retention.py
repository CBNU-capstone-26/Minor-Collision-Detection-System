"""영상 보관 기간 — 업로드한 지 VIDEO_RETENTION_DAYS(기본 90일)가 지난 영상과 딸린 것
(사고 클립·원본 클립·썸네일·분석 기록·업로드 원본)을 모두 지운다.

웹 서버(main.py)가 켜질 때 한 번, 그 뒤 CHECK_INTERVAL_SEC 마다 돈다.
분석이 대기·진행 중인 영상은 건너뛰고 끝난 뒤 다음 정리 때 지운다.
"""
from datetime import datetime, timedelta

from app import db_models
from app.db_connection import SessionLocal
from app.routers.analysis import ACTIVE_STATUSES
from app.routers.videos import purge_video
from app.settings import settings

CHECK_INTERVAL_SEC = 60 * 60   # 1시간마다


def expired_videos(db):
    """보관 기간이 지난 영상(업로드 시각 기준). 보관 기간을 끄면 빈 목록."""
    if settings.VIDEO_RETENTION_DAYS <= 0:
        return []
    cutoff = datetime.utcnow() - timedelta(days=settings.VIDEO_RETENTION_DAYS)
    return (db.query(db_models.Video)
            .filter(db_models.Video.created_at < cutoff)
            .order_by(db_models.Video.id).all())


def delete_expired_videos() -> list[int]:
    """기간이 지난 영상을 지우고 지운 영상 번호를 돌려준다. 한 개씩 commit — 중간에 실패해도 앞의 것은 지워진다."""
    db = SessionLocal()
    deleted = []
    try:
        for video in expired_videos(db):
            if any(t.status in ACTIVE_STATUSES for t in video.analysis_tasks):
                continue   # 분석이 끝나면 다음 정리 때
            if any(t.status in ACTIVE_STATUSES for t in video.vehicle_detection_tasks):
                continue   # 차량 탐지가 끝나면 다음 정리 때
            video_id = video.id
            purge_video(video, db)
            db.commit()
            deleted.append(video_id)
    finally:
        db.close()
    return deleted
