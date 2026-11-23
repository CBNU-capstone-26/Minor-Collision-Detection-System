"""영상 업로드, 목록, 상세, 스트리밍 라우터입니다."""
import math
import shutil
from datetime import date, datetime, timedelta, timezone
from uuid import uuid4
from pathlib import Path

from fastapi import (
    APIRouter, Depends, File, Form, HTTPException, UploadFile, Query,
)
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.db_connection import get_db
from app import db_models, api_schemas
from app.auth_guard import get_current_user, get_media_user
from app.settings import settings

router = APIRouter(prefix="/api/videos", tags=["videos"])


# ---------- 직렬화 헬퍼 ----------
def raw_clip_path(ev: db_models.CrashEvent):
    """합성 없는 원본 사고 클립 경로입니다. 분석할 때 표시 클립 옆에 '<이름>_raw.mp4'로 만듭니다.
    DB 컬럼을 늘리지 않으려고 이름 규칙으로 찾으며, 예전 분석 결과에는 없을 수 있습니다.
    """
    if not ev.clip_path:
        return None
    p = settings.abs_path(ev.clip_path)
    return p.with_name(f"{p.stem}_raw{p.suffix}")


def _unlink_event_clips(events, db: Session):
    """지우려는 이벤트들의 표시 클립과 원본 클립 파일을 지웁니다.
    예전 분석은 여러 이벤트가 한 파일을 함께 쓸 수 있으므로, 남는 다른 이벤트가 아직 쓰는 파일은 지우지 않습니다.
    """
    ids = {ev.id for ev in events}
    for path in {ev.clip_path for ev in events if ev.clip_path}:
        still_used = db.query(db_models.CrashEvent).filter(
            db_models.CrashEvent.clip_path == path,
            db_models.CrashEvent.id.notin_(ids)).count()
        if still_used:
            continue
        p = settings.abs_path(path)
        p.unlink(missing_ok=True)
        p.with_name(f"{p.stem}_raw{p.suffix}").unlink(missing_ok=True)


def analysis_no(ev: db_models.CrashEvent):
    """이 클립을 만든 분석이 그 영상의 몇 번째 분석인지(1부터) 돌려줍니다.

    실패하거나 취소된 분석도 한 번으로 셉니다. 한 번의 분석에서 나온 클립들은 같은 번호입니다.
    분석 작업 기록은 클립을 지워도 남고 영상을 지울 때만 지워지므로, 이전 클립을 지워도 번호가 바뀌지 않습니다.
    '다시 실행'도 새 분석 작업이라 한 번으로 셉니다.
    """
    ids = sorted(t.id for t in ev.video.analysis_tasks)
    return ids.index(ev.task_id) + 1 if ev.task_id in ids else None


def to_event_out(ev: db_models.CrashEvent) -> api_schemas.EventOut:
    return api_schemas.EventOut(
        id=ev.id,
        timestamp_sec=ev.timestamp_sec,
        frame_number=ev.frame_number,
        end_timestamp_sec=ev.end_timestamp_sec,
        end_frame_number=ev.end_frame_number,
        crash_prob=ev.crash_prob,
        has_clip=bool(ev.clip_path),
        has_raw_clip=bool(ev.clip_path) and raw_clip_path(ev).exists(),
        analyzed_at=_utc(ev.created_at),
        analysis_no=analysis_no(ev),
        clip_start_sec=ev.clip_start_sec,
    )


def _utc(dt):
    """DB에는 UTC를 시간대 없이 저장하므로 UTC라고 붙여 보냅니다. 브라우저가 자기 시간대로 바꿔 보여 줍니다."""
    return dt.replace(tzinfo=timezone.utc) if dt else None


def analysis_history(v: db_models.Video) -> list[api_schemas.AnalysisHistoryOut]:
    """영상의 분석 이력을 요청 순서대로 돌려줍니다. 번호는 analysis_no와 같은 기준입니다."""
    first_clip_at = {}   # 이력 기능 이전에 성공한 분석은 끝난 시각 대신 클립이 만들어진 시각을 씁니다.
    for ev in v.crash_events:
        if ev.created_at and (ev.task_id not in first_clip_at or ev.created_at < first_clip_at[ev.task_id]):
            first_clip_at[ev.task_id] = ev.created_at
    out = []
    for i, t in enumerate(sorted(v.analysis_tasks, key=lambda x: x.id)):
        out.append(api_schemas.AnalysisHistoryOut(
            analysis_no=i + 1,
            task_id=t.id,
            status=t.status,
            error_message=t.error_message,
            requested_at=_utc(t.created_at),
            finished_at=_utc(t.finished_at or first_clip_at.get(t.id)),
            event_count=t.event_count,
        ))
    return out


def video_expires_at(v: db_models.Video):
    """보관 기간이 끝나는 시각(UTC, 시간대 없음, DB와 같은 형식)입니다. 보관 기간을 끄면 None입니다."""
    if settings.VIDEO_RETENTION_DAYS <= 0 or v.created_at is None:
        return None
    return v.created_at + timedelta(days=settings.VIDEO_RETENTION_DAYS)


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
        created_at=_utc(v.created_at),   # 업로드 시각을 UTC라고 붙여 보내 브라우저 시간대로 표시합니다.
        expires_at=_utc(expires_at),
        expires_in_days=expires_in_days,
        events=[to_event_out(e) for e in v.crash_events],
        last_task=_last_task_out(v),
        analysis_done=any(t.status == "SUCCESS" for t in v.analysis_tasks),
        analysis_history=analysis_history(v) if with_history else [],
    )


def _last_task_out(v: db_models.Video):
    """가장 최근 분석 작업을 돌려줍니다(없으면 None)."""
    if not v.analysis_tasks:
        return None
    t = max(v.analysis_tasks, key=lambda x: x.id)
    return api_schemas.LastTaskOut(task_id=t.id, status=t.status,
                                   error_message=t.error_message, created_at=t.created_at)



def _extract_metadata(path: Path):
    """cv2로 영상 정보(가로, 세로, fps, 총 프레임 수)를 읽습니다."""
    import cv2  # 웹 프로세스에서 필요할 때만 opencv를 불러옵니다.
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
    title: str | None = Form(None),   # 선택 항목입니다. 비우면 화면이 '녹화일 녹화본 #N'으로 보여 줍니다.
    db: Session = Depends(get_db),
    user: db_models.User = Depends(get_current_user),
):
    title = (title or "").strip()   # 파일을 저장하기 전에 먼저 확인합니다.
    if len(title) > 100:
        raise HTTPException(status_code=400, detail="제목은 100자까지 입력할 수 있습니다.")

    ext = Path(file.filename).suffix or ".mp4"
    stored_name = f"{uuid4().hex}{ext}"
    dest = settings.UPLOAD_DIR / stored_name
    with open(dest, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    width, height, fps, total_frames = _extract_metadata(dest)
    if total_frames <= 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="영상을 읽을 수 없습니다.")

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
        video_path=settings.rel_path(dest),
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
    q = db.query(db_models.Video).filter(db_models.Video.user_id == user.id)
    if days and days < 9999:
        cutoff = date.today() - timedelta(days=days)
        q = q.filter(db_models.Video.recording_date >= cutoff)
    videos = q.order_by(db_models.Video.created_at.desc()).all()
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
    """영상을 삭제합니다. 관련 파일(원본, 썸네일, CAM 클립)과 DB(이벤트, 태스크, 영상)를 모두 지웁니다."""
    video = _get_owned_video(video_id, db, user)
    purge_video(video, db)
    db.commit()
    return {"deleted": video_id}


def purge_video(video: db_models.Video, db: Session) -> None:
    """영상과 딸린 것을 모두 지웁니다. 사용자가 지울 때와 보관 기간이 지났을 때(app/retention.py) 함께 씁니다.
    commit은 부르는 쪽에서 합니다.
    """
    # 1) CAM 클립과 원본 클립 파일, crash_events 행을 지웁니다.
    events = db.query(db_models.CrashEvent).filter(
        db_models.CrashEvent.video_id == video.id).all()
    _unlink_event_clips(events, db)
    for ev in events:
        db.delete(ev)

    # 2) analysis_tasks 행을 지웁니다.
    for task in db.query(db_models.AnalysisTask).filter(
            db_models.AnalysisTask.video_id == video.id).all():
        db.delete(task)

    # 3) 원본 영상과 썸네일 파일을 지웁니다.
    settings.abs_path(video.video_path).unlink(missing_ok=True)
    (settings.THUMBNAIL_DIR / f"{Path(video.video_path).stem}.jpg").unlink(missing_ok=True)

    # 4) 영상 행을 지웁니다.
    db.delete(video)


@router.delete("/{video_id}/events/{event_id}")
def delete_crash_event(
    video_id: int,
    event_id: int,
    db: Session = Depends(get_db),
    user: db_models.User = Depends(get_current_user),
):
    """사고 감지 이벤트 하나를 삭제합니다."""
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
    """해당 영상의 모든 사고 감지 이벤트를 삭제합니다."""
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
def stream_video(
    video_id: int,
    db: Session = Depends(get_db),
    user: db_models.User = Depends(get_media_user),
):
    # <video src>는 헤더를 실을 수 없어서 로그인 때 받은 영상 쿠키로 확인합니다(auth_guard).
    video = _get_owned_video(video_id, db, user)
    path = settings.abs_path(video.video_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="영상 파일이 없습니다.")
    # FileResponse는 HTTP Range 요청(영상 탐색)을 지원합니다.
    return FileResponse(str(path), media_type="video/mp4")


@router.get("/{video_id}/thumbnail")
def get_video_thumbnail(
    video_id: int,
    db: Session = Depends(get_db),
    user: db_models.User = Depends(get_media_user),
):
    # <img src>용이라 영상 쿠키로 확인합니다. 영상 첫 프레임을 jpg로 만들어 저장해 둡니다.
    video = _get_owned_video(video_id, db, user)

    thumb = settings.THUMBNAIL_DIR / f"{Path(video.video_path).stem}.jpg"
    if not thumb.exists():
        src = settings.abs_path(video.video_path)
        if not src.exists():
            raise HTTPException(status_code=404, detail="영상 파일이 없습니다.")
        import cv2  # 필요할 때만 불러옵니다.
        cap = cv2.VideoCapture(str(src))
        ok, frame = cap.read()  # 첫 프레임(BGR)
        cap.release()
        if not ok:
            raise HTTPException(status_code=404, detail="썸네일을 생성할 수 없습니다.")
        cv2.imwrite(str(thumb), frame)

    return FileResponse(str(thumb), media_type="image/jpeg")


@router.post("/{video_id}/detect-vehicles", response_model=api_schemas.VehicleDetectionResponse)
def detect_vehicles_in_video(
    video_id: int,
    timestamp_sec: float | None = Query(0.0),
    db: Session = Depends(get_db),
    user: db_models.User = Depends(get_current_user),
):
    """YOLO 모델로 지정 시각 프레임의 차량을 탐지하여 BBOX JSON을 DB에 저장하고 반환합니다."""
    import json

    video = _get_owned_video(video_id, db, user)
    src_path = settings.abs_path(video.video_path)
    if not src_path.exists():
        raise HTTPException(status_code=404, detail="영상 파일이 없습니다.")

    # 차량 탐지 모듈은 필요할 때 불러와, 웹 프로세스가 시작할 때 ultralytics와 torch를 올리지 않게 합니다.
    try:
        from app.vehicle_detector import get_detector, read_source_frame
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"차량 탐지 모듈 로드 실패: {err}")

    frame_index = int((timestamp_sec or 0.0) * (video.fps or 30.0))
    frame_index = max(0, min(frame_index, max(0, video.total_frames - 1)))

    try:
        frame = read_source_frame(src_path, frame_index=frame_index)
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"프레임 추출 실패: {err}")

    # YOLO11x(detection), 동적 imgsz(상한 1536), 낮은 conf, 야간 전처리를 써서 멀리 있거나 어둡거나 일부 가려진 차량도 잡습니다.
    # detection 모델이라 마스크 없이 YOLO 박스를 그대로 씁니다.
    # 탐지기는 get_detector로 프로세스마다 한 번만 불러와 재사용합니다.
    model_path = str(settings.BASE_DIR / "yolo11x.pt")
    if not Path(model_path).exists():
        model_path = "yolo11x.pt"  # 파일이 없으면 ultralytics가 자동으로 내려받습니다.

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

    # 서비스가 실제로 찾은 차량을 확인용 그림으로 outputs/에 저장합니다.
    # 탐지에 쓴 원본 프레임에 박스(번호, 클래스, 신뢰도)를 그립니다.
    # 영상을 지워도 그림은 남으므로 DETECT_DEBUG_IMAGES=1일 때만 저장합니다(settings).
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
        except Exception as _dbg_err:  # 확인용 그림 저장이 실패해도 탐지 응답에는 영향이 없습니다.
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

