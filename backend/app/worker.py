"""Celery 인스턴스 — Redis 브로커/백엔드.

워커 실행: cd backend && celery -A app.worker worker --loglevel=info
(워커 환경에는 torch/opencv 등 ML 의존성이 설치돼 있어야 한다.)
"""
from celery import Celery

from app.settings import settings

celery_app = Celery(
    "hitandrun",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

celery_app.conf.update(
    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
)

# 태스크 등록 (import 시 데코레이터가 celery_app에 바인딩)
from app import prediction_job  # noqa: E402,F401
