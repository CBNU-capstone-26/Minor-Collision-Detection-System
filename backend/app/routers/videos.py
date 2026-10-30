"""영상 업로드 / 목록 / 상세 / 스트리밍 라우터."""
import json
import logging
import math
import shutil
import tempfile
from datetime import date, datetime, timedelta, timezone
from uuid import uuid4
from pathlib import Path

from fastapi import (
    APIRouter, Depends, File, Form, HTTPException, UploadFile, Query, Request, Response,
)
from fastapi.responses import FileResponse, RedirectResponse
from sqlalchemy.orm import Session, selectinload

from app.db_connection import get_db
from app import db_models, api_schemas
from app.auth_guard import get_current_user, get_media_user
from app.settings import settings
from app.object_storage import LocalObjectStorage, get_storage, materialize
from app.thumbnails import ensure_thumbnail, thumbnail_key, delete_thumbnail_cache

router = APIRouter(prefix="/api/videos", tags=["videos"])


# ---------- 직렬화 헬퍼 ----------
def raw_clip_path(ev: db_models.CrashEvent):
    """합성 없는 원본 사고 클립 경로 — 분석 때 표시 클립 옆에 '<이름>_raw.mp4' 로 만든다.
    (DB 컬럼을 늘리지 않으려고 이름 규칙으로 찾는다. 예전 분석 결과엔 없을 수 있다)"""
    if not ev.clip_path:
        return None
    p = settings.abs_path(ev.clip_path)
    return p.with_name(f"{p.stem}_raw{p.suffix}")


def raw_clip_key(ev: db_models.CrashEvent):
    if not ev.clip_path:
        return None
    return f"{Path(ev.clip_path).with_suffix('')}_raw.mp4"


def _unlink_event_clips(events, db: Session):
    """다른 이벤트가 공유하지 않는 R2/로컬 클립만 지운다."""
    ids = {ev.id for ev in events}
    storage = get_storage()
    for path in {ev.clip_path for ev in events if ev.clip_path}:
        still_used = db.query(db_models.CrashEvent).filter(
            db_models.CrashEvent.clip_path == path,
            db_models.CrashEvent.id.notin_(ids),
        ).count()
        if not still_used:
            storage.delete(path)
            storage.delete(f"{Path(path).with_suffix('')}_raw.mp4")


def to_event_out(ev: db_models.CrashEvent) -> api_schemas.EventOut:
    return api_schemas.EventOut(
        id=ev.id,
        timestamp_sec=ev.timestamp_sec,
        frame_number=ev.frame_number,
        end_timestamp_sec=ev.end_timestamp_sec,
        end_frame_number=ev.end_frame_number,
        crash_prob=ev.crash_prob,
        has_clip=bool(ev.clip_path),
        # 목록을 만드는 동안 원격 객체 저장소에 HEAD 요청을 보내면, R2 지연 하나가
        # 영상 목록 전체를 멈춘다. 원본 클립은 분석 결과의 이름 규칙으로 생성되므로
        # 여기서는 DB 메타데이터만으로 노출 여부를 판단한다.
        has_raw_clip=bool(raw_clip_key(ev)),
        analyzed_at=_utc(ev.created_at),
        analysis_no=analysis_no(ev),
        clip_start_sec=ev.clip_start_sec,
    )


def _utc(dt):
    return dt.replace(tzinfo=timezone.utc) if dt else None


def analysis_no(ev: db_models.CrashEvent):
    ids = sorted(task.id for task in ev.video.analysis_tasks)
    return ids.index(ev.task_id) + 1 if ev.task_id in ids else None


def analysis_history(v: db_models.Video) -> list[api_schemas.AnalysisHistoryOut]:
    first_clip_at = {}
    for event in v.crash_events:
        if event.created_at and (event.task_id not in first_clip_at or event.created_at < first_clip_at[event.task_id]):
            first_clip_at[event.task_id] = event.created_at
    return [api_schemas.AnalysisHistoryOut(
        analysis_no=index + 1,
        task_id=task.id,
        status=task.status,
        error_message=task.error_message,
        requested_at=_utc(task.created_at),
        finished_at=_utc(task.finished_at or first_clip_at.get(task.id)),
        event_count=task.event_count,
    ) for index, task in enumerate(sorted(v.analysis_tasks, key=lambda item: item.id))]


def to_video_out(v: db_models.Video, with_history: bool = False) -> api_schemas.VideoOut:
    duration = v.total_frames / v.fps if v.fps else 0.0
    expires_at = video_expires_at(v)
    expires_in_days = (math.ceil((expires_at - datetime.utcnow()).total_seconds() / 86400)
                       if expires_at else None)
    return api_schemas.VideoOut(
        id=v.id,
        video_name=v.video_name,
        title=v.title,
        recording_date=v.recording_date,
        camera_location=v.camera_location or "주차장",
        width=v.width,
        height=v.height,
        fps=v.fps,
        total_frames=v.total_frames,
        duration_sec=duration,
        detected_vehicles=v.detected_vehicles,
        created_at=_utc(v.created_at),
        expires_at=_utc(expires_at),
        expires_in_days=expires_in_days,
        events=[to_event_out(e) for e in v.crash_events],
        last_task=_last_task_out(v),
        analysis_done=any(t.status == "SUCCESS" for t in v.analysis_tasks),
        analysis_history=analysis_history(v) if with_history else [],
    )


def video_expires_at(v: db_models.Video):
    if settings.VIDEO_RETENTION_DAYS <= 0 or v.created_at is None:
        return None
    return v.created_at + timedelta(days=settings.VIDEO_RETENTION_DAYS)


def video_list_query(db: Session, user_id: int):
    """Build the list query with relationships used by ``to_video_out`` loaded up front."""
    return (
        db.query(db_models.Video)
        .options(
            selectinload(db_models.Video.crash_events),
            selectinload(db_models.Video.analysis_tasks),
        )
        .filter(db_models.Video.user_id == user_id)
        .order_by(db_models.Video.created_at.desc())
    )


def _last_task_out(v: db_models.Video):
    """가장 최근 분석 작업(없으면 None)."""
    if not v.analysis_tasks:
        return None
    t = max(v.analysis_tasks, key=lambda x: x.id)
    return api_schemas.LastTaskOut(task_id=t.id, status=t.status,
                                   error_message=t.error_message, created_at=t.created_at)



def _extract_metadata(path: Path):
    """cv2로 영상 메타데이터(가로/세로/fps/총프레임) 추출."""
    import cv2  # 지연 임포트 (웹 프로세스에 opencv 필요)
    cap = cv2.VideoCapture(str(path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return width, height, fps, total


@router.post("", response_model=api_schemas.VideoOut)
def upload_video(
    file: UploadFile = File(...),
    recording_date: str | None = Form(None),
    title: str | None = Form(None),
    db: Session = Depends(get_db),
    user: db_models.User = Depends(get_current_user),
):
    title = (title or "").strip()
    if len(title) > 100:
        raise HTTPException(status_code=400, detail="제목은 100자까지 입력할 수 있습니다.")

    ext = Path(file.filename).suffix or ".mp4"
    stored_name = f"{uuid4().hex}{ext}"
    storage = get_storage()
    object_key = f"uploads/{stored_name}"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = Path(temp_file.name)
    try:
        width, height, fps, total_frames = _extract_metadata(temp_path)
        if total_frames <= 0:
            raise HTTPException(status_code=400, detail="영상을 읽을 수 없습니다.")
        storage.upload_path(temp_path, object_key, content_type=file.content_type or "video/mp4")
        try:
            ensure_thumbnail(storage, object_key, settings.THUMBNAIL_DIR, source_video=temp_path)
        except Exception as exc:
            logging.getLogger(__name__).warning("Thumbnail preparation failed: %s", type(exc).__name__)
    finally:
        temp_path.unlink(missing_ok=True)

    rec_date: date | None = None
    if recording_date:
        try:
            rec_date = date.fromisoformat(recording_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="녹화일자 형식이 잘못되었습니다.")

    video = db_models.Video(
        user_id=user.id,
        video_name=file.filename,
        title=title or None,
        video_path=object_key,
        recording_date=rec_date,
        width=width,
        height=height,
        fps=fps,
        total_frames=total_frames,
    )
    db.add(video)
    db.commit()
    db.refresh(video)
    return to_video_out(video)


@router.get("", response_model=list[api_schemas.VideoOut])
def list_videos(
    days: int | None = Query(None),
    db: Session = Depends(get_db),
    user: db_models.User = Depends(get_current_user),
):
    q = video_list_query(db, user.id)
    if days and days < 9999:
        cutoff = date.today() - timedelta(days=days)
        q = q.filter(db_models.Video.recording_date >= cutoff)
    videos = q.all()
    return [to_video_out(v) for v in videos]


def _get_owned_video(video_id: int, db: Session, user: db_models.User):
    video = db.get(db_models.Video, video_id)
    if video is None or video.user_id != user.id:
        raise HTTPException(status_code=404, detail="영상을 찾을 수 없습니다.")
    return video


@router.get("/{video_id}", response_model=api_schemas.VideoOut)
def get_video(
    video_id: int,
    db: Session = Depends(get_db),
    user: db_models.User = Depends(get_current_user),
):
    return to_video_out(_get_owned_video(video_id, db, user), with_history=True)


@router.delete("/{video_id}")
def delete_video(
    video_id: int,
    db: Session = Depends(get_db),
    user: db_models.User = Depends(get_current_user),
):
    """영상 삭제 — 관련 파일(원본·썸네일·CAM 클립)과 DB(이벤트·태스크·영상) 모두 제거."""
    video = _get_owned_video(video_id, db, user)

    purge_video(video, db)
    db.commit()
    return {"deleted": video_id}


@router.delete("/{video_id}/analysis-history")
def clear_video_analysis_history(
    video_id: int,
    db: Session = Depends(get_db),
    user: db_models.User = Depends(get_current_user),
):
    """해당 영상의 분석 이력·사고 이벤트·클립을 지운다. 원본 영상은 보존한다."""
    video = _get_owned_video(video_id, db, user)
    tasks = db.query(db_models.AnalysisTask).filter(
        db_models.AnalysisTask.video_id == video.id
    ).all()
    if any(task.status in ("PENDING", "PROCESSING") for task in tasks):
        raise HTTPException(status_code=409, detail="분석이 진행 중인 동안에는 이력을 초기화할 수 없습니다.")

    events = db.query(db_models.CrashEvent).filter(
        db_models.CrashEvent.video_id == video.id
    ).all()
    _unlink_event_clips(events, db)
    for event in events:
        db.delete(event)
    for task in tasks:
        db.delete(task)
    db.commit()
    return {"deleted_history_count": len(tasks), "deleted_event_count": len(events)}


def purge_video(video: db_models.Video, db: Session) -> None:
    """영상·클립·이력·원본을 제거한다. 보관 기간 정리와 수동 삭제가 함께 쓴다."""
    # 1) 관련 CAM 클립 파일 삭제 + crash_events 행 삭제
    events = db.query(db_models.CrashEvent).filter(
        db_models.CrashEvent.video_id == video.id).all()
    _unlink_event_clips(events, db)
    for ev in events:
        db.delete(ev)

    # 2) analysis_tasks 행 삭제
    for task in db.query(db_models.AnalysisTask).filter(
            db_models.AnalysisTask.video_id == video.id).all():
        db.delete(task)

    # Vehicle-detection history has a required FK to videos. Remove these rows
    # explicitly so both Supabase/Postgres and MySQL can delete the video.
    detection_tasks = db.query(db_models.VehicleDetectionTask).filter(
        db_models.VehicleDetectionTask.video_id == video.id).all()
    for task in detection_tasks:
        if task.celery_task_id and task.status in VEHICLE_DETECTION_ACTIVE_STATUSES:
            try:
                from app.worker import celery_app
                celery_app.control.revoke(task.celery_task_id, terminate=True)
            except Exception:
                pass
        db.delete(task)

    # 3) 원본 영상 + 썸네일 파일 삭제
    storage = get_storage()
    storage.delete(video.video_path)
    storage.delete(f"thumbnails/{Path(video.video_path).stem}.jpg")
    storage.delete(thumbnail_key(video.video_path))
    delete_thumbnail_cache(video.video_path, settings.THUMBNAIL_DIR)

    # 4) 영상 행 삭제
    db.delete(video)


@router.delete("/{video_id}/events/{event_id}")
def delete_crash_event(
    video_id: int,
    event_id: int,
    db: Session = Depends(get_db),
    user: db_models.User = Depends(get_current_user),
):
    """특정 사고 감지 이벤트 1건 삭제."""
    video = _get_owned_video(video_id, db, user)
    event = db.query(db_models.CrashEvent).filter(
        db_models.CrashEvent.id == event_id,
        db_models.CrashEvent.video_id == video.id,
    ).first()
    if event is None:
        raise HTTPException(status_code=404, detail="이벤트를 찾을 수 없습니다.")

    _unlink_event_clips([event], db)

    db.delete(event)
    db.commit()
    return {"deleted_event_id": event_id}


@router.delete("/{video_id}/events")
def clear_all_crash_events(
    video_id: int,
    db: Session = Depends(get_db),
    user: db_models.User = Depends(get_current_user),
):
    """해당 영상의 모든 사고 감지 이벤트 삭제."""
    video = _get_owned_video(video_id, db, user)
    events = db.query(db_models.CrashEvent).filter(
        db_models.CrashEvent.video_id == video.id
    ).all()
    count = len(events)
    _unlink_event_clips(events, db)
    for ev in events:
        db.delete(ev)
    db.commit()
    return {"deleted_count": count}



@router.get("/{video_id}/stream")
def stream_video(video_id: int, db: Session = Depends(get_db), user: db_models.User = Depends(get_media_user)):
    video = _get_owned_video(video_id, db, user)
    storage = get_storage()
    if not storage.exists(video.video_path):
        raise HTTPException(status_code=404, detail="영상 파일이 없습니다.")
    if not isinstance(storage, LocalObjectStorage):
        return RedirectResponse(storage.presigned_url(
            video.video_path, settings.S3_PRESIGNED_URL_EXPIRES))
    path = storage.local_path(video.video_path)
    # FileResponse는 HTTP Range 요청(영상 탐색)을 지원한다.
    return FileResponse(str(path), media_type="video/mp4")


@router.get("/{video_id}/thumbnail")
def video_thumbnail(video_id: int, request: Request, db: Session = Depends(get_db), user: db_models.User = Depends(get_media_user)):
    video = _get_owned_video(video_id, db, user)
    try:
        path = ensure_thumbnail(get_storage(), video.video_path, settings.THUMBNAIL_DIR)
    except (FileNotFoundError, ValueError):
        raise HTTPException(status_code=404, detail="썸네일을 생성할 수 없습니다.")
    response = FileResponse(str(path), media_type="image/jpeg", stat_result=path.stat(),
                            headers={"Cache-Control": "private, no-cache",
                                     "Vary": "Cookie, Authorization"})
    # Always authorize before revalidating a browser's private cached image.
    tags = {value.strip().removeprefix("W/")
            for value in request.headers.get("if-none-match", "").split(",")}
    if response.headers["etag"] in tags or "*" in tags:
        return Response(status_code=304, headers={
            name: response.headers[name]
            for name in ("etag", "last-modified", "cache-control", "vary")
        })
    return response


@router.post("/{video_id}/detect-vehicles-sync", response_model=api_schemas.VehicleDetectionResponse)
def detect_vehicles_in_video(
    video_id: int,
    timestamp_sec: float | None = Query(0.0),
    db: Session = Depends(get_db),
    user: db_models.User = Depends(get_current_user),
):
    """YOLO 모델로 지정 시각 프레임의 차량을 탐지하여 BBOX JSON을 DB에 저장하고 반환합니다."""
    import json

    video = _get_owned_video(video_id, db, user)
    storage = get_storage()
    if not storage.exists(video.video_path):
        raise HTTPException(status_code=404, detail="영상 파일이 없습니다.")

    # 차량 탐지 모듈은 지연 임포트 — ultralytics/torch를 웹 프로세스 시작 시 로드하지 않도록.
    try:
        from app.vehicle_detector import get_detector, read_source_frame
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"차량 탐지 모듈 로드 실패: {err}")

    frame_index = int((timestamp_sec or 0.0) * (video.fps or 30.0))
    frame_index = max(0, min(frame_index, max(0, video.total_frames - 1)))

    try:
        with materialize(
            storage, video.video_path, suffix=Path(video.video_path).suffix
        ) as src_path:
            frame = read_source_frame(src_path, frame_index=frame_index)
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"프레임 추출 실패: {err}")

    # YOLO11x(detection) + 동적 imgsz(상한 1536) + 낮은 conf + 야간 전처리
    # → 원거리/야간/부분가림 회수율 강화. detection 모델이라 마스크 없이
    #   YOLO detection bbox를 그대로 사용한다. 동적 imgsz는 프레임 해상도에
    #   맞춰 입력 크기를 정해 고해상도 정보손실/저해상도 낭비를 함께 줄인다.
    # 탐지기는 get_detector로 프로세스당 1회만 로드·재사용(매 요청 재로드 오버헤드 제거).
    model_path = str(settings.BASE_DIR / "yolo11x.pt")
    if not Path(model_path).exists():
        model_path = "yolo11x.pt"  # 파일 없으면 ultralytics가 자동 다운로드

    try:
        detector = get_detector(
            model_path=model_path, conf=0.15, imgsz=1536,
            enhance_night=True, dynamic_imgsz=True, imgsz_min=640)
        detections = detector.detect_frame(frame)
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"YOLO 차량 탐지 중 오류: {err}")

    detected_list = []
    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = det.bbox
        detected_list.append({
            "id": idx,
            "class_name": det.class_name,
            "confidence": round(det.confidence, 4),
            "bbox": [x1, y1, x2, y2],
        })

    # [디버그] 서비스가 실제로 탐지한 결과를 이미지로 저장 (outputs/) — 눈으로 확인용.
    #   탐지에 쓴 원본 프레임에 bbox(#번호·클래스·신뢰도)를 그려 저장한다.
    if settings.DETECT_DEBUG_IMAGES:
      try:
        import cv2
        dbg = frame.copy()
        for item in detected_list:
            x1, y1, x2, y2 = item["bbox"]
            cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'#{item["id"] + 1} {item["class_name"]} {round(item["confidence"] * 100)}%'
            cv2.putText(dbg, label, (x1, max(y1 - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        out_dir = settings.BASE_DIR / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        base = Path(video.video_name).stem
        out_img = out_dir / f"detect_{base}_f{frame_index}.jpg"
        cv2.imwrite(str(out_img), dbg)
        print(f"[detect] 결과 이미지 저장: {out_img}  (탐지 {len(detected_list)}대, 프레임 {frame.shape[1]}x{frame.shape[0]})")
      except Exception as _dbg_err:  # 디버그 저장 실패는 탐지 응답에 영향 없음
          print(f"[detect] 결과 이미지 저장 실패(무시): {_dbg_err}")

    json_str = json.dumps(detected_list, ensure_ascii=False)
    video.detected_vehicles = json_str
    db.commit()

    return api_schemas.VehicleDetectionResponse(
        video_id=video.id,
        total_detected=len(detected_list),
        detected_vehicles=[
            api_schemas.DetectedVehicleBox(**item) for item in detected_list
        ],
    )


VEHICLE_DETECTION_ACTIVE_STATUSES = ("PENDING", "PROCESSING")


def _vehicle_detection_progress(task: db_models.VehicleDetectionTask) -> dict:
    if task.status != "PROCESSING" or not task.celery_task_id:
        return {"progress": None, "stage": None}
    from app.worker import celery_app
    result = celery_app.AsyncResult(task.celery_task_id)
    if result.state == "PROGRESS" and isinstance(result.info, dict):
        return {"progress": result.info.get("percent"), "stage": result.info.get("stage")}
    return {"progress": None, "stage": None}


def _vehicle_detection_rows(task: db_models.VehicleDetectionTask) -> list[dict]:
    if not task.result_json:
        return []
    try:
        rows = json.loads(task.result_json)
    except json.JSONDecodeError:
        return []
    return rows if isinstance(rows, list) else []


def _enqueue_vehicle_detection(video: db_models.Video, timestamp_sec: float, db: Session):
    existing = (db.query(db_models.VehicleDetectionTask)
                .filter(db_models.VehicleDetectionTask.video_id == video.id,
                        db_models.VehicleDetectionTask.status.in_(VEHICLE_DETECTION_ACTIVE_STATUSES))
                .order_by(db_models.VehicleDetectionTask.id).first())
    if existing is not None:
        return api_schemas.VehicleDetectionTaskOut(
            task_id=existing.id,
            celery_task_id=existing.celery_task_id,
            status=existing.status,
            already_running=True,
        )

    task = db_models.VehicleDetectionTask(
        video_id=video.id,
        timestamp_sec=max(0.0, timestamp_sec),
        status="PENDING",
    )
    db.add(task)
    db.commit()
    db.refresh(task)

    from app.vehicle_job import run_vehicle_detection_task
    try:
        async_result = run_vehicle_detection_task.delay(task.id)
    except Exception as exc:
        task.status = "FAILURE"
        task.error_message = f"탐지 작업을 큐에 넣지 못했습니다: {exc}"[:2000]
        task.finished_at = datetime.utcnow()
        db.commit()
        raise HTTPException(status_code=503, detail="탐지 작업을 시작할 수 없습니다. 잠시 후 다시 시도해 주세요.") from exc
    task.celery_task_id = async_result.id
    db.commit()
    return api_schemas.VehicleDetectionTaskOut(
        task_id=task.id,
        celery_task_id=task.celery_task_id,
        status=task.status,
    )


@router.post("/{video_id}/detect-vehicles", response_model=api_schemas.VehicleDetectionTaskOut)
def enqueue_vehicle_detection(
    video_id: int,
    timestamp_sec: float | None = Query(0.0),
    db: Session = Depends(get_db),
    user: db_models.User = Depends(get_current_user),
):
    """GPU 워커에 차량 탐지를 맡기고, UI가 추적할 작업 ID를 반환한다."""
    video = _get_owned_video(video_id, db, user)
    return _enqueue_vehicle_detection(video, timestamp_sec or 0.0, db)


@router.get("/{video_id}/vehicle-detection-tasks/{task_id}",
            response_model=api_schemas.VehicleDetectionTaskStatusOut)
def get_vehicle_detection_task(
    video_id: int,
    task_id: int,
    db: Session = Depends(get_db),
    user: db_models.User = Depends(get_current_user),
):
    video = _get_owned_video(video_id, db, user)
    task = db.get(db_models.VehicleDetectionTask, task_id)
    if task is None or task.video_id != video.id:
        raise HTTPException(status_code=404, detail="차량 탐지 작업을 찾을 수 없습니다.")
    return api_schemas.VehicleDetectionTaskStatusOut(
        task_id=task.id,
        status=task.status,
        error_message=task.error_message,
        detected_vehicles=[api_schemas.DetectedVehicleBox(**row)
                           for row in _vehicle_detection_rows(task)],
        **_vehicle_detection_progress(task),
    )
