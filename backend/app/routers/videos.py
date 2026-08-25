"""영상 업로드 / 목록 / 상세 / 스트리밍 라우터."""
import shutil
from datetime import date, datetime, timedelta
from uuid import uuid4
from pathlib import Path

from fastapi import (
    APIRouter, Depends, File, Form, HTTPException, UploadFile, Query,
)
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.db_connection import get_db
from app import db_models, api_schemas
from app.auth_guard import get_current_user
from app.settings import settings

router = APIRouter(prefix="/api/videos", tags=["videos"])


# ---------- 직렬화 헬퍼 ----------
def to_event_out(ev: db_models.CrashEvent) -> api_schemas.EventOut:
    return api_schemas.EventOut(
        id=ev.id,
        timestamp_sec=ev.timestamp_sec,
        frame_number=ev.frame_number,
        end_timestamp_sec=ev.end_timestamp_sec,
        end_frame_number=ev.end_frame_number,
        crash_prob=ev.crash_prob,
        has_clip=bool(ev.cam_heatmap_path),
    )


def to_video_out(v: db_models.Video) -> api_schemas.VideoOut:
    duration = v.total_frames / v.fps if v.fps else 0.0
    return api_schemas.VideoOut(
        id=v.id,
        video_name=v.video_name,
        recording_date=v.recording_date,
        camera_location=v.camera_location or "주차장",
        recording_start_time=v.recording_start_time or "20:30",
        width=v.width,
        height=v.height,
        fps=v.fps,
        total_frames=v.total_frames,
        duration_sec=duration,
        detected_vehicles=v.detected_vehicles,
        created_at=v.created_at,
        events=[to_event_out(e) for e in v.crash_events],
    )



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
    db: Session = Depends(get_db),
    user: db_models.User = Depends(get_current_user),
):
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
    return to_video_out(_get_owned_video(video_id, db, user))


@router.delete("/{video_id}")
def delete_video(
    video_id: int,
    db: Session = Depends(get_db),
    user: db_models.User = Depends(get_current_user),
):
    """영상 삭제 — 관련 파일(원본·썸네일·CAM 클립)과 DB(이벤트·태스크·영상) 모두 제거."""
    video = _get_owned_video(video_id, db, user)

    # 1) 관련 CAM 클립 파일 삭제 + crash_events 행 삭제
    events = db.query(db_models.CrashEvent).filter(
        db_models.CrashEvent.video_id == video_id).all()
    for ev in events:
        if ev.cam_heatmap_path:
            settings.abs_path(ev.cam_heatmap_path).unlink(missing_ok=True)
        db.delete(ev)

    # 2) analysis_tasks 행 삭제
    for task in db.query(db_models.AnalysisTask).filter(
            db_models.AnalysisTask.video_id == video_id).all():
        db.delete(task)

    # 3) 원본 영상 + 썸네일 파일 삭제
    settings.abs_path(video.video_path).unlink(missing_ok=True)
    (settings.THUMBNAIL_DIR / f"{Path(video.video_path).stem}.jpg").unlink(missing_ok=True)

    # 4) 영상 행 삭제
    db.delete(video)
    db.commit()
    return {"deleted": video_id}


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

    if event.cam_heatmap_path:
        settings.abs_path(event.cam_heatmap_path).unlink(missing_ok=True)

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
    for ev in events:
        if ev.cam_heatmap_path:
            settings.abs_path(ev.cam_heatmap_path).unlink(missing_ok=True)
        db.delete(ev)
    db.commit()
    return {"deleted_count": count}



@router.get("/{video_id}/stream")
def stream_video(video_id: int, db: Session = Depends(get_db)):
    # <video src> 태그는 커스텀 헤더를 못 보내므로 인증 미적용 (MVP)
    video = db.get(db_models.Video, video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="영상을 찾을 수 없습니다.")
    path = settings.abs_path(video.video_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="영상 파일이 없습니다.")
    # FileResponse는 HTTP Range 요청(영상 탐색)을 지원한다.
    return FileResponse(str(path), media_type="video/mp4")


@router.get("/{video_id}/thumbnail")
def video_thumbnail(video_id: int, db: Session = Depends(get_db)):
    # <img src> 태그용 — 인증 미적용 (MVP). 영상 첫 프레임을 jpg로 캐시 생성.
    video = db.get(db_models.Video, video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="영상을 찾을 수 없습니다.")

    thumb = settings.THUMBNAIL_DIR / f"{Path(video.video_path).stem}.jpg"
    if not thumb.exists():
        src = settings.abs_path(video.video_path)
        if not src.exists():
            raise HTTPException(status_code=404, detail="영상 파일이 없습니다.")
        import cv2  # 지연 임포트
        cap = cv2.VideoCapture(str(src))
        ok, frame = cap.read()  # 첫 프레임 (BGR)
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
    import sys

    video = _get_owned_video(video_id, db, user)
    src_path = settings.abs_path(video.video_path)
    if not src_path.exists():
        raise HTTPException(status_code=404, detail="영상 파일이 없습니다.")

    annotation_dir = str(settings.BASE_DIR / "annotation")
    if annotation_dir not in sys.path:
        sys.path.insert(0, annotation_dir)

    try:
        from auto_bbox_yolo_seg import VehicleBBoxDetector, read_source_frame
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"차량 탐지 모듈 로드 실패: {err}")

    frame_index = int((timestamp_sec or 0.0) * (video.fps or 30.0))
    frame_index = max(0, min(frame_index, max(0, video.total_frames - 1)))

    try:
        frame = read_source_frame(src_path, frame_index=frame_index)
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"프레임 추출 실패: {err}")

    model_path = str(settings.BASE_DIR / "yolov8n.pt")
    if not Path(model_path).exists():
        model_path = "yolov8n.pt"

    try:
        detector = VehicleBBoxDetector(model_path=model_path, conf=0.25)
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

