from __future__ import annotations

from pathlib import Path

from .tracker import Observation


def track_video(video: str | Path, model_name: str, confidence: float = 0.5) -> list[Observation]:
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("YOLO 탐지를 실행하려면 ultralytics와 opencv-python을 설치해야 합니다.") from exc
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video}")
    model = YOLO(model_name)
    observations: list[Observation] = []
    frame = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            result = model.track(image, persist=True, classes=[2], conf=confidence, tracker="bytetrack.yaml", verbose=False)[0]
            if result.boxes is None or result.boxes.id is None:
                frame += 1
                continue
            boxes = result.boxes.xyxy.cpu().tolist()
            ids = result.boxes.id.int().cpu().tolist()
            confidences = result.boxes.conf.cpu().tolist()
            class_ids = result.boxes.cls.int().cpu().tolist()
            for bbox, track_id, confidence_value, class_id in zip(boxes, ids, confidences, class_ids):
                observations.append(Observation(int(track_id), frame, [round(value) for value in bbox], float(confidence_value), int(class_id)))
            frame += 1
    finally:
        capture.release()
    return observations
