"""Celery 차량 탐지 작업.

웹 프로세스는 작업 등록만 하고, 프레임 다운로드·YOLO 추론·결과 저장은 ML 워커가 맡는다.
"""
import json
from datetime import datetime
from pathlib import Path

from app import db_models
from app.db_connection import SessionLocal
from app.object_storage import get_storage, materialize
from app.settings import settings
from app.worker import celery_app


def _detection_rows(detections) -> list[dict]:
    return [
        {
            "id": index,
            "class_name": detection.class_name,
            "confidence": round(detection.confidence, 4),
            "bbox": list(detection.bbox),
        }
        for index, detection in enumerate(detections)
    ]


@celery_app.task(bind=True)
def run_vehicle_detection_task(self, detection_task_id: int):
    """원격 저장소의 영상 한 프레임에서 차량을 찾아 DB에 결과를 저장한다."""
    db = SessionLocal()
    task = None
    try:
        task = db.get(db_models.VehicleDetectionTask, detection_task_id)
        if task is None:
            return {"error": f"vehicle detection task {detection_task_id} not found"}
        if task.status != "PENDING":
            return {"task_id": task.id, "skipped": task.status}

        task.status = "PROCESSING"
        db.commit()
        self.update_state(state="PROGRESS", meta={"percent": 5, "stage": "loading"})

        video = db.get(db_models.Video, task.video_id)
        if video is None:
            raise RuntimeError("대상 영상을 찾을 수 없습니다.")

        storage = get_storage()
        if not storage.exists(video.video_path):
            raise RuntimeError("영상 파일이 없습니다.")

        from app.vehicle_detector import get_detector, read_source_frame

        frame_index = int(task.timestamp_sec * (video.fps or 30.0))
        frame_index = max(0, min(frame_index, max(0, video.total_frames - 1)))
        with materialize(storage, video.video_path, suffix=Path(video.video_path).suffix) as source_path:
            frame = read_source_frame(source_path, frame_index=frame_index)

        self.update_state(state="PROGRESS", meta={"percent": 40, "stage": "detecting"})
        model_path = str(settings.BASE_DIR / "yolo11x.pt")
        detector = get_detector(
            model_path=model_path if Path(model_path).exists() else "yolo11x.pt",
            conf=0.15,
            imgsz=1536,
            enhance_night=True,
            dynamic_imgsz=True,
            imgsz_min=640,
        )
        rows = _detection_rows(detector.detect_frame(frame))

        result_json = json.dumps(rows, ensure_ascii=False)
        task.result_json = result_json
        task.status = "SUCCESS"
        task.finished_at = datetime.utcnow()
        # 기존 UI·분석 경로와의 호환성을 위해 최신 탐지 결과도 영상 행에 유지한다.
        video.detected_vehicles = result_json
        db.commit()
        return {"task_id": task.id, "video_id": video.id, "detections": rows}
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        if task is not None:
            task = db.get(db_models.VehicleDetectionTask, task.id)
            if task is not None:
                task.status = "FAILURE"
                task.error_message = str(exc)[:2000]
                task.finished_at = datetime.utcnow()
                db.commit()
        raise
    finally:
        db.close()
