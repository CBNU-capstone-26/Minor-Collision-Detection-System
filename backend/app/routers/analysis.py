"""분석 요청 / 태스크 상태 / 이벤트·클립 조회 라우터."""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.db_connection import get_db
from app import db_models, api_schemas
from app.auth_guard import get_current_user
from app.settings import settings
from app.routers.videos import to_event_out, _get_owned_video

router = APIRouter(prefix="/api", tags=["analysis"])


@router.post("/videos/{video_id}/analyze",
             response_model=api_schemas.AnalyzeResponse)
def analyze_video(
    video_id: int,
    req: api_schemas.AnalyzeRequest,
    db: Session = Depends(get_db),
    user: db_models.User = Depends(get_current_user),
):
    video = _get_owned_video(video_id, db, user)

    task = db_models.AnalysisTask(
        video_id=video.id,
        bbox_xmin=req.bbox_xmin,
        bbox_ymin=req.bbox_ymin,
        bbox_xmax=req.bbox_xmax,
        bbox_ymax=req.bbox_ymax,
        status="PENDING",
    )
    db.add(task)
    db.commit()
    db.refresh(task)

    # Celery 큐에 적재 (torch 미설치 웹 프로세스에서도 .delay는 동작)
    from app.prediction_job import run_prediction_task
    async_result = run_prediction_task.delay(task.id)
    task.celery_task_id = async_result.id
    db.commit()

    return api_schemas.AnalyzeResponse(
        task_id=task.id,
        celery_task_id=task.celery_task_id,
        status=task.status,
    )


@router.get("/tasks/{task_id}", response_model=api_schemas.TaskStatusOut)
def get_task_status(
    task_id: int,
    db: Session = Depends(get_db),
    user: db_models.User = Depends(get_current_user),
):
    task = db.get(db_models.AnalysisTask, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다.")
    events = db.query(db_models.CrashEvent).filter(
        db_models.CrashEvent.task_id == task.id).all()
    return api_schemas.TaskStatusOut(
        task_id=task.id,
        status=task.status,
        error_message=task.error_message,
        events=[to_event_out(e) for e in events],
    )


@router.get("/videos/{video_id}/events",
            response_model=list[api_schemas.EventOut])
def list_events(
    video_id: int,
    db: Session = Depends(get_db),
    user: db_models.User = Depends(get_current_user),
):
    _get_owned_video(video_id, db, user)
    events = db.query(db_models.CrashEvent).filter(
        db_models.CrashEvent.video_id == video_id).order_by(
        db_models.CrashEvent.timestamp_sec).all()
    return [to_event_out(e) for e in events]


@router.get("/events/{event_id}/clip")
def get_event_clip(event_id: int, db: Session = Depends(get_db)):
    # <video src> 태그용 — 인증 미적용 (MVP)
    event = db.get(db_models.CrashEvent, event_id)
    if event is None or not event.cam_heatmap_path:
        raise HTTPException(status_code=404, detail="클립을 찾을 수 없습니다.")
    path = settings.abs_path(event.cam_heatmap_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="클립 파일이 없습니다.")
    return FileResponse(str(path), media_type="video/mp4")
