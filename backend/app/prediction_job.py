"""Celery 태스크 — AI 추론 + 사고구간 CAM 클립 생성.

torch/opencv/모델 임포트는 모두 **태스크 내부에서 지연 로딩**한다.
→ FastAPI 웹 프로세스는 ML 의존성 없이도 이 모듈을 임포트(.delay 호출)할 수 있다.
"""
from app.worker import celery_app
from app.settings import settings
from app.db_connection import SessionLocal
from app import db_models

# 워커 프로세스당 모델 1회 로드 후 재사용
_model = None


def _get_model():
    global _model
    if _model is not None:
        return _model

    import sys
    # model/ 폴더를 path에 추가 — 내부 절대 임포트(import config 등) 지원
    sys.path.insert(0, str(settings.MODEL_DIR))

    import torch
    import config as model_config
    from hitandrun_model import HitAndRun3DCNN
    from device_utils import get_device, is_channels_last_3d_supported

    # 추론 디바이스(config.INFER_DEVICE_TYPE) 사용 — 서비스 워커는 기본 CPU
    device = get_device(model_config.INFER_DEVICE_TYPE)
    model = HitAndRun3DCNN(num_classes=model_config.MODEL_NUM_CLASSES).to(device)
    if is_channels_last_3d_supported(device) and model_config.USE_CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last_3d)
    state_dict = torch.load(
        str(settings.WEIGHTS_PATH), map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    _model = model
    print(f"[worker] 모델 로드 완료 (device={device})")
    return _model


@celery_app.task(bind=True)
def run_prediction_task(self, task_id: int):
    """AnalysisTask를 받아 추론 → 사고구간 클립 생성 → CrashEvent 저장."""
    db = SessionLocal()
    try:
        task = db.get(db_models.AnalysisTask, task_id)
        if task is None:
            return {"error": f"task {task_id} not found"}

        task.status = "PROCESSING"
        db.commit()

        video = db.get(db_models.Video, task.video_id)
        model = _get_model()

        from predict_cam import predict_events_and_clips

        results = predict_events_and_clips(
            model,
            video_path=settings.abs_path(video.video_path),
            bbox=(task.bbox_xmin, task.bbox_ymin, task.bbox_xmax, task.bbox_ymax),
            output_dir=settings.CLIP_DIR,
        )

        for r in results:
            db.add(db_models.CrashEvent(
                task_id=task.id,
                video_id=video.id,
                timestamp_sec=r["start_sec"],
                frame_number=r["start_frame"],
                end_timestamp_sec=r["end_sec"],
                end_frame_number=r["end_frame"],
                crash_prob=r["crash_prob"],
                cam_heatmap_path=settings.rel_path(r["clip_path"]),
            ))

        task.status = "SUCCESS"
        db.commit()
        return {"task_id": task_id, "events": len(results)}

    except Exception as exc:  # noqa: BLE001
        db.rollback()
        task = db.get(db_models.AnalysisTask, task_id)
        if task is not None:
            task.status = "FAILURE"
            task.error_message = str(exc)[:2000]
            db.commit()
        raise
    finally:
        db.close()
