"""Celery 인스턴스 — Redis 브로커/백엔드.

워커 실행: cd backend && celery -A app.worker worker --loglevel=info
(워커 환경에는 torch/opencv 등 ML 의존성이 설치돼 있어야 한다.)
"""
from celery import Celery
from celery.signals import worker_ready

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


@worker_ready.connect
def _fail_interrupted_tasks(**_):
    """워커가 (재)시작될 때, 이전 워커에서 처리 중(PROCESSING)이던 작업을 실패로 정리한다.

    처리 중이던 메시지는 시작 시점에 이미 확인(ack)돼 다시 배달되지 않으므로, 그대로
    두면 DB 에 PROCESSING 이 영원히 남고 화면은 끝나길 계속 기다린다. 실패로 바꿔 두면
    화면이 '다시 실행'을 안내한다.
    ⚠️ 워커가 하나라는 전제다. 워커를 여러 대 띄우면 다른 워커가 실제로 처리 중인
       작업까지 실패로 바꾸므로 이 정리 방식을 바꿔야 한다.
    """
    from app.db_connection import SessionLocal
    from app import db_models
    db = SessionLocal()
    try:
        n = (db.query(db_models.AnalysisTask)
             .filter(db_models.AnalysisTask.status == "PROCESSING")
             .update({"status": "FAILURE",
                      "error_message": "분석 서버가 재시작되어 분석이 중단되었습니다. 다시 실행해 주세요."},
                     synchronize_session=False))
        db.commit()
        if n:
            print(f"[worker] 재시작으로 중단된 분석 {n}건을 실패로 정리")
    finally:
        db.close()
