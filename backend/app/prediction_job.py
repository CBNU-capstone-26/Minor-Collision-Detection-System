"""Celery 태스크입니다. AI 추론을 돌리고 사고 구간 CAM 클립을 만듭니다.

torch, opencv, 모델은 모두 태스크 안에서 처음 쓸 때 불러옵니다.
그래서 FastAPI 웹 프로세스는 ML 의존성 없이도 이 모듈을 불러와 .delay를 호출할 수 있습니다.
"""
from datetime import datetime
from pathlib import Path

from app.worker import celery_app
from app.settings import settings
from app.db_connection import SessionLocal
from app import db_models

# 워커 프로세스마다 모델을 한 번 불러와 재사용합니다.
_model = None


def _get_model():
    global _model
    if _model is not None:
        return _model

    import sys
    # 선택한 백본 폴더를 path에 넣어 그 안의 모듈(import config 등)을 불러올 수 있게 합니다.
    # 폴더는 settings.MODEL_VARIANT가 고릅니다.
    sys.path.insert(0, str(settings.MODEL_DIR))

    import torch
    import config as model_config
    from hitandrun_model import HitAndRun3DCNN
    from device_utils import get_device, is_channels_last_3d_supported

    # 추론 장치는 config.INFER_DEVICE_TYPE을 따릅니다. GPU가 없으면 get_device가 경고를 남기고 CPU로 바꿉니다.
    # 실제 장치는 아래 '모델 로드 완료' 로그로 확인합니다.
    device = get_device(model_config.INFER_DEVICE_TYPE)
    # 서비스 가중치 경로는 model/config.py(SERVICE_WEIGHTS_PATH)에서 관리합니다.
    # 입력 크기가 224가 아닌 가중치(이름에 _s256 등)라면 서버 .env의 INPUT_SIZE를 맞춰야 하며, 다르면 여기서 멈춥니다.
    # 헤드 풀링(_top3 등)은 .env의 HEAD_TEMPORAL_POOL을 비워 두면 가중치 이름을 따라가므로, 모델을 만들기 전에 확인합니다.
    from hitandrun_model import check_weights
    check_weights(model_config.SERVICE_WEIGHTS_PATH)
    model = HitAndRun3DCNN(num_classes=model_config.MODEL_NUM_CLASSES).to(device)
    if is_channels_last_3d_supported(device) and model_config.USE_CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last_3d)
    state_dict = torch.load(
        str(model_config.SERVICE_WEIGHTS_PATH), map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[worker] 모델 로드 완료 — {getattr(model_config, 'MODEL_NAME', '?')} "
          f"(device={device}, weights={model_config.SERVICE_WEIGHTS_PATH.name})")
    # 추론 실행 방식을 모델을 불러올 때 정해 둡니다. 이 컴퓨터에서 처음이면 자가 검사를 하고, 이후에는 저장된 결과를 씁니다.
    # 실패해도 분석은 PyTorch로 돌아야 하므로 여기서 예외를 막습니다.
    try:
        from infer_runtime import get_runner
        get_runner(model)
    except Exception as exc:  # noqa: BLE001
        print(f"[worker] 추론 실행 방식 선택 실패 — PyTorch 로 진행: {exc}")
    _model = model
    return _model


@celery_app.task(bind=True)
def run_prediction_task(self, task_id: int):
    """AnalysisTask를 받아 추론하고, 사고 구간 클립을 만들고, CrashEvent를 저장합니다."""
    db = SessionLocal()
    try:
        task = db.get(db_models.AnalysisTask, task_id)
        if task is None:
            return {"error": f"task {task_id} not found"}
        # 취소 표시는 워커 메모리(revoke)에만 있어서 워커가 다시 시작되면 사라지고,
        # 대기 중이던 메시지가 다시 배달될 수 있습니다. 그래서 DB 상태로 한 번 더 거릅니다.
        if task.status != "PENDING":
            return {"task_id": task_id, "skipped": task.status}

        task.status = "PROCESSING"
        db.commit()
        # 화면에 '모델 불러오기' 단계를 표시합니다(처음이면 추론 실행 방식 자가 검사 포함).
        self.update_state(state="PROGRESS", meta={"percent": 0, "stage": "loading"})

        video = db.get(db_models.Video, task.video_id)
        model = _get_model()

        try:
            from inference import predict_events_and_clips
        except ImportError:
            # 다른 백본 폴더(MODEL_VARIANT)는 파일 이름이 predict_cam.py입니다.
            from predict_cam import predict_events_and_clips

        # 진행률(%)과 남은 시간(초)을 Celery 상태(Redis)에 기록하면 /tasks가 읽어 화면에 전달합니다.
        # 진행률은 0~99까지만 쓰고, 100%는 작업 완료(SUCCESS)로만 표시합니다.
        # 남은 시간은 분석 함수가 실제로 잰 속도로 계산하며, 진행률이 그대로여도 1.5초마다 갱신합니다.
        import time as _time
        _last = {"pct": -1, "t": 0.0}

        def _on_progress(frac, eta=None, eta_min=False, stage=None, count=None, index=None):
            # stage, count, index는 지금 단계와 그 단계의 개수, 순번입니다(model/inference.py의 _report 참고).
            # 진행률은 소수 첫째 자리까지 기록해 긴 영상에서도 막대가 조금씩 움직이게 합니다.
            # 단계가 바뀌면 바로, 진행률만 바뀌면 0.4초에 한 번, 그대로여도 1.5초마다 기록합니다.
            pct = round(min(99.0, frac * 100), 1)
            now = _time.time()
            key = (stage, count, index)
            if (key != _last.get("key")
                    or (pct != _last["pct"] and now - _last["t"] >= 0.4)
                    or now - _last["t"] >= 1.5):
                _last.update(pct=pct, t=now, key=key)
                self.update_state(state="PROGRESS", meta={
                    "percent": pct, "eta": None if eta is None else int(round(eta)),
                    "eta_min": bool(eta_min), "stage": stage, "count": count, "index": index})

        results = predict_events_and_clips(
            model,
            video_path=settings.abs_path(video.video_path),
            bbox=(task.bbox_xmin, task.bbox_ymin, task.bbox_xmax, task.bbox_ymax),
            output_dir=settings.CLIP_DIR,
            progress_callback=_on_progress,
            # 분석마다 클립 이름을 다르게 해, 같은 영상을 다시 분석해도 이전 클립을 덮어쓰지 않게 합니다.
            clip_name=f"{Path(video.video_path).stem}_t{task.id}",
        )

        # 분석 도중 취소됐는데(revoke 실패 등) 끝까지 돌았다면 결과를 버립니다.
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
                clip_path=settings.rel_path(r["clip_path"]),
                clip_start_sec=r.get("clip_start_sec"),
            ))

        task.status = "SUCCESS"
        task.finished_at = datetime.utcnow()
        task.event_count = len(results)
        db.commit()
        return {"task_id": task_id, "events": len(results)}

    except Exception as exc:  # noqa: BLE001
        db.rollback()
        task = db.get(db_models.AnalysisTask, task_id)
        if task is not None:
            task.status = "FAILURE"
            task.error_message = str(exc)[:2000]
            task.finished_at = datetime.utcnow()
            db.commit()
        raise
    finally:
        db.close()
