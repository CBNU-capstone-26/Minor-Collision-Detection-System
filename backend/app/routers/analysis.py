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


# 대기·진행 중으로 보는 상태
ACTIVE_STATUSES = ("PENDING", "PROCESSING")


def _get_owned_task(task_id: int, db: Session, user: db_models.User):
    """다른 사용자의 작업은 조회·취소할 수 없다(없는 것처럼 404)."""
    task = db.get(db_models.AnalysisTask, task_id)
    if task is None or task.video is None or task.video.user_id != user.id:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다.")
    return task


def _active_task_for_video(video_id: int, db: Session):
    return (db.query(db_models.AnalysisTask)
            .filter(db_models.AnalysisTask.video_id == video_id,
                    db_models.AnalysisTask.status.in_(ACTIVE_STATUSES))
            .order_by(db_models.AnalysisTask.id).first())


def _progress(task: db_models.AnalysisTask):
    """처리 중이면 Celery 상태(Redis)에서 (진행도 %, 남은 시간 초, 최소추정 여부)를 읽어온다."""
    if task.status != "PROCESSING" or not task.celery_task_id:
        return None, None, False
    from app.worker import celery_app
    res = celery_app.AsyncResult(task.celery_task_id)
    if res.state == "PROGRESS" and isinstance(res.info, dict):
        return res.info.get("percent"), res.info.get("eta"), bool(res.info.get("eta_min"))
    return None, None, False


def _enqueue(video_id: int, bbox: dict, db: Session) -> api_schemas.AnalyzeResponse:
    """분석 작업을 만들어 Celery 큐에 넣는다. 같은 영상에 대기·진행 중인 분석이
    있으면 새로 만들지 않고 그 작업을 돌려준다(같은 영상 중복 분석 방지 —
    워커가 하나라 중복 요청은 앞 작업이 끝날 때까지 기다리기만 한다)."""
    existing = _active_task_for_video(video_id, db)
    if existing is not None:
        return api_schemas.AnalyzeResponse(
            task_id=existing.id, celery_task_id=existing.celery_task_id,
            status=existing.status, already_running=True)

    task = db_models.AnalysisTask(video_id=video_id, status="PENDING", **bbox)
    db.add(task)
    db.commit()
    db.refresh(task)

    # Celery 큐에 적재 (torch 미설치 웹 프로세스에서도 .delay는 동작)
    from app.prediction_job import run_prediction_task
    async_result = run_prediction_task.delay(task.id)
    task.celery_task_id = async_result.id
    db.commit()

    return api_schemas.AnalyzeResponse(
        task_id=task.id, celery_task_id=task.celery_task_id, status=task.status)


@router.post("/videos/{video_id}/analyze",
             response_model=api_schemas.AnalyzeResponse)
def analyze_video(
    video_id: int,
    req: api_schemas.AnalyzeRequest,
    db: Session = Depends(get_db),
    user: db_models.User = Depends(get_current_user),
):
    video = _get_owned_video(video_id, db, user)
    return _enqueue(video.id, req.model_dump(), db)


# ⚠️ /tasks/{task_id} 보다 먼저 등록해야 한다('active' 가 task_id 로 잡히지 않게)
@router.get("/tasks/active", response_model=list[api_schemas.ActiveTaskOut])
def list_active_tasks(
    db: Session = Depends(get_db),
    user: db_models.User = Depends(get_current_user),
):
    """로그인한 사용자의 대기·진행 중 분석. 화면을 새로 열 때 이걸로 진행 상황을 복원한다."""
    tasks = (db.query(db_models.AnalysisTask)
             .join(db_models.Video, db_models.AnalysisTask.video_id == db_models.Video.id)
             .filter(db_models.Video.user_id == user.id,
                     db_models.AnalysisTask.status.in_(ACTIVE_STATUSES))
             .order_by(db_models.AnalysisTask.id).all())
    out = []
    for t in tasks:
        progress, eta_sec, eta_min = _progress(t)
        out.append(api_schemas.ActiveTaskOut(
            task_id=t.id, video_id=t.video_id, video_name=t.video.video_name,
            recording_date=t.video.recording_date,
            camera_location=t.video.camera_location or "주차장",
            status=t.status, progress=progress, eta_sec=eta_sec, eta_is_min=eta_min,
            created_at=t.created_at))
    return out


@router.get("/tasks/{task_id}", response_model=api_schemas.TaskStatusOut)
def get_task_status(
    task_id: int,
    db: Session = Depends(get_db),
    user: db_models.User = Depends(get_current_user),
):
    task = _get_owned_task(task_id, db, user)
    events = db.query(db_models.CrashEvent).filter(
        db_models.CrashEvent.task_id == task.id).all()
    progress, eta_sec, eta_min = _progress(task)
    return api_schemas.TaskStatusOut(
        task_id=task.id,
        status=task.status,
        progress=progress,
        eta_sec=eta_sec,
        eta_is_min=eta_min,
        error_message=task.error_message,
        events=[to_event_out(e) for e in events],
    )


@router.post("/tasks/{task_id}/cancel", response_model=api_schemas.TaskStatusOut)
def cancel_task(
    task_id: int,
    db: Session = Depends(get_db),
    user: db_models.User = Depends(get_current_user),
):
    task = _get_owned_task(task_id, db, user)
    if task.status in ACTIVE_STATUSES:
        task.status = "CANCELLED"
        task.error_message = "사용자가 작업을 취소함"
        if task.celery_task_id:
            try:
                from app.worker import celery_app
                celery_app.control.revoke(task.celery_task_id, terminate=True)
            except Exception:
                pass
        db.commit()
    events = db.query(db_models.CrashEvent).filter(
        db_models.CrashEvent.task_id == task.id).all()
    return api_schemas.TaskStatusOut(
        task_id=task.id,
        status=task.status,
        progress=None,
        error_message=task.error_message,
        events=[to_event_out(e) for e in events],
    )


@router.post("/tasks/{task_id}/retry", response_model=api_schemas.AnalyzeResponse)
def retry_task(
    task_id: int,
    db: Session = Depends(get_db),
    user: db_models.User = Depends(get_current_user),
):
    """실패·취소된 분석을 같은 영상·같은 차량 박스로 다시 실행한다."""
    task = _get_owned_task(task_id, db, user)
    if task.status in ACTIVE_STATUSES:
        raise HTTPException(status_code=409, detail="이미 대기·진행 중인 작업입니다.")
    bbox = dict(bbox_xmin=task.bbox_xmin, bbox_ymin=task.bbox_ymin,
                bbox_xmax=task.bbox_xmax, bbox_ymax=task.bbox_ymax)
    return _enqueue(task.video_id, bbox, db)



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
    media_type = "video/webm" if str(path).endswith(".webm") else "video/mp4"
    return FileResponse(str(path), media_type=media_type)
