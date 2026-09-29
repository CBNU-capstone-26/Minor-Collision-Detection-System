"""영상 업로드 / 목록 / 상세 / 스트리밍 라우터."""
import shutil
import tempfile
from datetime import date, datetime, timedelta
from uuid import uuid4
from pathlib import Path

from fastapi import (
    APIRouter, Depends, File, Form, HTTPException, UploadFile, Query,
)
from fastapi.responses import FileResponse, RedirectResponse
from sqlalchemy.orm import Session

from app.db_connection import get_db
from app import db_models, api_schemas
from app.auth_guard import get_current_user
from app.settings import settings
from app.object_storage import LocalObjectStorage, get_storage, materialize

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
    storage = get_storage()
    object_key = f"uploads/{stored_name}"
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
        video_path=object_key,
        recording_date=rec_date,
        width=width,
        height=height,
        fps=fps,
        total_frames=total_frames,
    )
    if not isinstance(storage, LocalObjectStorage):
        storage.upload_path(dest, object_key, content_type=file.content_type or "video/mp4")
        dest.unlink(missing_ok=True)

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
            get_storage().delete(ev.cam_heatmap_path)
        db.delete(ev)

    # 2) analysis_tasks 행 삭제
    for task in db.query(db_models.AnalysisTask).filter(
            db_models.AnalysisTask.video_id == video_id).all():
        db.delete(task)

    # 3) 원본 영상 + 썸네일 파일 삭제
    storage = get_storage()
    storage.delete(video.video_path)
    storage.delete(f"thumbnails/{Path(video.video_path).stem}.jpg")

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
        get_storage().delete(event.cam_heatmap_path)

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
            get_storage().delete(ev.cam_heatmap_path)
        db.delete(ev)
    db.commit()
    return {"deleted_count": count}



@router.get("/{video_id}/stream")
def stream_video(video_id: int, db: Session = Depends(get_db)):
    # <video src> 태그는 커스텀 헤더를 못 보내므로 인증 미적용 (MVP)
    video = db.get(db_models.Video, video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="영상을 찾을 수 없습니다.")
    storage = get_storage()
    if not storage.exists(video.video_path):
        raise HTTPException(status_code=404, detail="영상 파일이 없습니다.")
    if not isinstance(storage, LocalObjectStorage):
        return RedirectResponse(storage.presigned_url(video.video_path))
    path = storage.local_path(video.video_path)
    # FileResponse는 HTTP Range 요청(영상 탐색)을 지원한다.
    return FileResponse(str(path), media_type="video/mp4")


@router.get("/{video_id}/thumbnail")
def video_thumbnail(video_id: int, db: Session = Depends(get_db)):
    # <img src> 태그용 — 인증 미적용 (MVP). 영상 첫 프레임을 jpg로 캐시 생성.
    video = db.get(db_models.Video, video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="영상을 찾을 수 없습니다.")

    storage = get_storage()
    thumb_key = f"thumbnails/{Path(video.video_path).stem}.jpg"
    if storage.exists(thumb_key):
        if not isinstance(storage, LocalObjectStorage):
            return RedirectResponse(storage.presigned_url(thumb_key))
        return FileResponse(str(storage.local_path(thumb_key)), media_type="image/jpeg")

    if not storage.exists(video.video_path):
        raise HTTPException(status_code=404, detail="영상 파일이 없습니다.")
    with materialize(storage, video.video_path, suffix=Path(video.video_path).suffix) as src:
        import cv2  # 지연 임포트
        cap = cv2.VideoCapture(str(src))
        ok, frame = cap.read()  # 첫 프레임 (BGR)
        cap.release()
        if not ok:
            raise HTTPException(status_code=404, detail="썸네일을 생성할 수 없습니다.")
        with tempfile.NamedTemporaryFile(suffix=".jpg") as temp_thumb:
            import cv2
            if not cv2.imwrite(temp_thumb.name, frame):
                raise HTTPException(status_code=404, detail="썸네일을 생성할 수 없습니다.")
            storage.upload_path(temp_thumb.name, thumb_key, content_type="image/jpeg")

    if not isinstance(storage, LocalObjectStorage):
        return RedirectResponse(storage.presigned_url(thumb_key))
    return FileResponse(str(storage.local_path(thumb_key)), media_type="image/jpeg")


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
    storage = get_storage()
    if not storage.exists(video.video_path):
        raise HTTPException(status_code=404, detail="영상 파일이 없습니다.")

    # 차량 탐지 모듈은 지연 임포트 — ultralytics/torch를 웹 프로세스 시작 시 로드하지 않도록.
    try:
        from app.vehicle_detector import (
            DEFAULT_RTDETR_MODEL,
            DEFAULT_YOLO_MODEL,
            DEFAULT_YOLO_SEG_MODEL,
            get_hybrid_detector,
            read_source_frame,
        )
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"차량 탐지 모듈 로드 실패: {err}")

    frame_index = int((timestamp_sec or 0.0) * (video.fps or 30.0))
    frame_index = max(0, min(frame_index, max(0, video.total_frames - 1)))

    try:
        with materialize(storage, video.video_path, suffix=Path(video.video_path).suffix) as src_path:
            frame = read_source_frame(src_path, frame_index=frame_index)
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"프레임 추출 실패: {err}")

    try:
        def model_path(name: str) -> str:
            local_path = settings.BASE_DIR / name
            return str(local_path) if local_path.exists() else name

        detector = get_hybrid_detector(
            rtdetr_path=model_path(DEFAULT_RTDETR_MODEL),
            yolo_path=model_path(DEFAULT_YOLO_MODEL),
            yolo_seg_path=model_path(DEFAULT_YOLO_SEG_MODEL),
        )
        detection_result = detector.detect_frame(frame)
        detections = detection_result.detections
        detector_mode = detection_result.mode
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"하이브리드 차량 탐지 중 오류: {err}")

    detected_list = []
    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = det.bbox
        detected_list.append({
            "id": idx,
            "class_name": det.class_name,
            "confidence": round(det.confidence, 4),
            "bbox": [x1, y1, x2, y2],
            "source": det.source,
        })

    # [디버그] 서비스가 실제로 탐지한 결과를 이미지로 저장 (outputs/) — 눈으로 확인용.
    #   탐지에 쓴 원본 프레임에 bbox(#번호·클래스·신뢰도)를 그려 저장한다.
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
        detector_mode=detector_mode,
    )
