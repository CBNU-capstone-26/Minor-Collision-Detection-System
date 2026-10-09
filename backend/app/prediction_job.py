"""Celery 태스크 — AI 추론 + 사고구간 CAM 클립 생성.

torch/opencv/모델 임포트는 모두 **태스크 내부에서 지연 로딩**한다.
→ FastAPI 웹 프로세스는 ML 의존성 없이도 이 모듈을 임포트(.delay 호출)할 수 있다.
"""
from pathlib import Path

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
    # 선택된 백본 폴더를 path에 추가 — 내부 절대 임포트(import config 등) 지원.
    # 폴더는 settings.MODEL_VARIANT(환경변수)가 고른다.
    sys.path.insert(0, str(settings.MODEL_DIR))

    import torch
    import config as model_config
    from hitandrun_model import HitAndRun3DCNN
    from device_utils import get_device, is_channels_last_3d_supported

    # 추론 디바이스(config.INFER_DEVICE_TYPE="cuda") — GPU 가 없으면 get_device 가
    # 경고를 찍고 CPU 로 자동 전환한다. 실제 디바이스는 아래 '모델 로드 완료' 로그로 확인.
    device = get_device(model_config.INFER_DEVICE_TYPE)
    model = HitAndRun3DCNN(num_classes=model_config.MODEL_NUM_CLASSES).to(device)
    if is_channels_last_3d_supported(device) and model_config.USE_CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last_3d)
    # 배포 가중치 경로는 model/config.py 에서 관리 (SERVICE_WEIGHTS_PATH)
    state_dict = torch.load(
        str(model_config.SERVICE_WEIGHTS_PATH), map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    _model = model
    print(f"[worker] 모델 로드 완료 — {getattr(model_config, 'MODEL_NAME', '?')} "
          f"(device={device}, weights={model_config.SERVICE_WEIGHTS_PATH.name})")
    return _model


@celery_app.task(bind=True)
def run_prediction_task(self, task_id: int):
    """AnalysisTask를 받아 추론 → 사고구간 클립 생성 → CrashEvent 저장."""
    db = SessionLocal()
    try:
        task = db.get(db_models.AnalysisTask, task_id)
        if task is None:
            return {"error": f"task {task_id} not found"}
        # 취소 표시는 워커 메모리(revoke)에만 있어서 워커가 재시작되면 사라지고,
        # 대기 중이던 메시지가 다시 배달될 수 있다 → DB 상태로 한 번 더 거른다.
        if task.status != "PENDING":
            return {"task_id": task_id, "skipped": task.status}

        task.status = "PROCESSING"
        db.commit()

        video = db.get(db_models.Video, task.video_id)
        model = _get_model()

        from predict_cam import predict_events_and_clips

        # 진행률(%)과 남은 시간(초)을 Celery 상태(Redis)에 기록 → /tasks 가 읽어 화면에 전달.
        # 진행률은 0~99 — 100%는 작업 완료(SUCCESS)로만 표시한다. 남은 시간은 분석 함수가
        # 실제로 잰 속도로 계산해 주며, 진행률이 그대로여도 1.5초마다 갱신해 준다.
        import time as _time
        _last = {"pct": -1, "t": 0.0}

        def _on_progress(frac, eta=None, eta_min=False):
            pct = min(99, int(frac * 100))
            now = _time.time()
            if pct != _last["pct"] or now - _last["t"] >= 1.5:
                _last.update(pct=pct, t=now)
                self.update_state(state="PROGRESS", meta={
                    "percent": pct, "eta": None if eta is None else int(round(eta)),
                    "eta_min": bool(eta_min)})

        results = predict_events_and_clips(
            model,
            video_path=settings.abs_path(video.video_path),
            bbox=(task.bbox_xmin, task.bbox_ymin, task.bbox_xmax, task.bbox_ymax),
            output_dir=settings.CLIP_DIR,
            progress_callback=_on_progress,
        )

        # 분석 도중 취소됐는데(revoke 실패 등) 끝까지 돌았다면 결과를 버린다
        db.refresh(task)
        if task.status == "CANCELLED":
            for r in results:
                for key in ("clip_path", "raw_clip_path"):
                    if r.get(key):
                        Path(r[key]).unlink(missing_ok=True)
            return {"task_id": task_id, "skipped": "CANCELLED"}

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
