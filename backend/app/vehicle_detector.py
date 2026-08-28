"""
YOLO segmentation 기반 차량 bbox 자동 생성 도구.

목적:
  - YOLO-seg로 차량을 찾는다.
  - detection bbox를 그대로 쓰지 않고 segmentation mask의 실제 외곽을 기준으로
    상/하/좌/우 좌표를 다시 계산한다.
  - 프로젝트 학습 txt 포맷인 `car,id,x1,y1,x2,y2`로 저장한다.

설치:
  pip install ultralytics opencv-python

예시:
  python model/auto_bbox_yolo_seg.py \
    --source data/real/real01.mp4 \
    --output-image outputs/real01_auto_bbox.jpg \
    --output-txt data/real/real01.txt \
    --frame-index 0

  # YOLO가 부분 차량을 못 잡았을 때: 기존 bbox를 넣고 좌우만 타이트하게 보정
  python model/auto_bbox_yolo_seg.py \
    --source outputs/sample.jpg \
    --output-image outputs/sample_tight_bbox.jpg \
    --output-txt outputs/sample_tight_bbox.txt \
    --manual-bbox 190,0,1240,965 \
    --shrink-left 0.02 \
    --shrink-right 0.12
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - runtime dependency 안내용
    raise SystemExit(
        "ultralytics가 설치되어 있지 않습니다. "
        "다음 명령으로 설치하세요: pip install ultralytics opencv-python"
    ) from exc


DEFAULT_VEHICLE_CLASSES = ("car", "truck", "bus", "motorcycle")


@dataclass
class VehicleDetection:
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]


class VehicleBBoxDetector:
    """YOLO-seg mask 외곽으로 차량 bbox를 타이트하게 재계산한다."""

    def __init__(
        self,
        model_path: str = "yolov8n-seg.pt",
        conf: float = 0.35,
        vehicle_classes: Iterable[str] = DEFAULT_VEHICLE_CLASSES,
        mask_threshold: float = 0.5,
        min_area: int = 100,
        shrink_x: float = 0.0,
        shrink_left: float | None = None,
        shrink_right: float | None = None,
        nms_iou: float = 0.35,
        contain_threshold: float = 0.85,
        imgsz: int = 640,
        enhance_night: bool = False,
    ):
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz              # 추론 입력 해상도 (클수록 원거리 차량 탐지↑)
        self.enhance_night = enhance_night  # 어두운 프레임 CLAHE/감마 전처리 여부
        self.vehicle_classes = set(vehicle_classes)
        self.mask_threshold = mask_threshold
        self.min_area = min_area
        self.shrink_left = shrink_x if shrink_left is None else shrink_left
        self.shrink_right = shrink_x if shrink_right is None else shrink_right
        self.nms_iou = nms_iou
        self.contain_threshold = contain_threshold

    @staticmethod
    def _clip_box(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        width: int,
        height: int,
    ) -> tuple[int, int, int, int]:
        x1 = max(0, min(int(round(x1)), width - 1))
        y1 = max(0, min(int(round(y1)), height - 1))
        x2 = max(0, min(int(round(x2)), width - 1))
        y2 = max(0, min(int(round(y2)), height - 1))
        return x1, y1, x2, y2

    @staticmethod
    def _shrink_box_horizontal(
        bbox: tuple[int, int, int, int],
        left_ratio: float,
        right_ratio: float,
        frame_width: int,
        frame_height: int,
    ) -> tuple[int, int, int, int]:
        """좌우가 넓게 잡힌 bbox를 차량 중심 방향으로 줄인다."""
        if left_ratio <= 0 and right_ratio <= 0:
            return bbox

        x1, y1, x2, y2 = bbox
        width = x2 - x1
        left_dx = int(round(width * left_ratio))
        right_dx = int(round(width * right_ratio))
        if left_dx <= 0 and right_dx <= 0:
            return bbox
        if x1 + left_dx >= x2 - right_dx:
            return bbox

        return VehicleBBoxDetector._clip_box(
            x1 + left_dx,
            y1,
            x2 - right_dx,
            y2,
            frame_width,
            frame_height,
        )

    @staticmethod
    def _box_area(bbox: tuple[int, int, int, int]) -> int:
        x1, y1, x2, y2 = bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    @classmethod
    def _intersection_area(
        cls,
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int],
    ) -> int:
        ax1, ay1, ax2, ay2 = first
        bx1, by1, bx2, by2 = second
        x1 = max(ax1, bx1)
        y1 = max(ay1, by1)
        x2 = min(ax2, bx2)
        y2 = min(ay2, by2)
        return cls._box_area((x1, y1, x2, y2))

    @classmethod
    def _iou(
        cls,
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int],
    ) -> float:
        intersection = cls._intersection_area(first, second)
        union = cls._box_area(first) + cls._box_area(second) - intersection
        return intersection / union if union > 0 else 0.0

    @classmethod
    def _contained_ratio(
        cls,
        inner: tuple[int, int, int, int],
        outer: tuple[int, int, int, int],
    ) -> float:
        inner_area = cls._box_area(inner)
        if inner_area <= 0:
            return 0.0
        return cls._intersection_area(inner, outer) / inner_area

    def _deduplicate_detections(
        self,
        detections: list[VehicleDetection],
    ) -> list[VehicleDetection]:
        """같은 차량에 여러 bbox가 생기면 confidence가 높은 bbox만 남긴다."""
        kept: list[VehicleDetection] = []
        for det in sorted(detections, key=lambda item: item.confidence, reverse=True):
            duplicate = False
            for kept_det in kept:
                iou = self._iou(det.bbox, kept_det.bbox)
                contain = self._contained_ratio(det.bbox, kept_det.bbox)
                if iou >= self.nms_iou or contain >= self.contain_threshold:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(det)
        return sorted(kept, key=lambda det: (det.bbox[0], det.bbox[1]))

    def _bbox_from_mask(
        self,
        mask: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> tuple[int, int, int, int] | None:
        """Segmentation mask의 실제 양수 픽셀 외곽으로 bbox를 계산한다."""
        binary = (mask > self.mask_threshold).astype(np.uint8)
        if binary.shape[:2] != (frame_height, frame_width):
            binary = cv2.resize(
                binary,
                (frame_width, frame_height),
                interpolation=cv2.INTER_NEAREST,
            )

        # 작은 구멍이나 끊긴 mask를 약하게 정리한다.
        kernel = np.ones((3, 3), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < self.min_area:
            return None

        x, y, w, h = cv2.boundingRect(contour)
        return self._clip_box(x, y, x + w - 1, y + h - 1, frame_width, frame_height)

    @staticmethod
    def _enhance_low_light(frame_bgr: np.ndarray, brightness_thresh: int = 80) -> np.ndarray:
        """어두운(야간) 프레임이면 CLAHE(대비 향상)+감마 보정으로 차량 가시성을 높인다.

        밝은 주간 프레임은 원본을 그대로 반환(과보정 방지). 밝기(그레이 평균)가
        임계값 미만일 때만 보정을 적용한다.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if float(gray.mean()) >= brightness_thresh:
            return frame_bgr
        # CLAHE는 LAB의 L(밝기) 채널에만 적용 → 색 왜곡 없이 대비만 향상
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_ch = clahe.apply(l_ch)
        enhanced = cv2.cvtColor(cv2.merge((l_ch, a_ch, b_ch)), cv2.COLOR_LAB2BGR)
        # 감마 보정(<1) → 어두운 영역을 밝게
        gamma = 0.7
        table = np.array([((i / 255.0) ** gamma) * 255
                          for i in range(256)], dtype=np.uint8)
        return cv2.LUT(enhanced, table)

    def detect_frame(self, frame_bgr: np.ndarray) -> list[VehicleDetection]:
        if self.enhance_night:
            frame_bgr = self._enhance_low_light(frame_bgr)
        height, width = frame_bgr.shape[:2]
        result = self.model(frame_bgr, conf=self.conf,
                             imgsz=self.imgsz, verbose=False)[0]

        if result.boxes is None:
            return []

        masks = result.masks.data.cpu().numpy() if result.masks is not None else None
        detections: list[VehicleDetection] = []

        for idx, box in enumerate(result.boxes):
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            if class_name not in self.vehicle_classes:
                continue

            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            tight_bbox = None
            if masks is not None and idx < len(masks):
                tight_bbox = self._bbox_from_mask(masks[idx], width, height)

            if tight_bbox is None:
                tight_bbox = self._clip_box(x1, y1, x2, y2, width, height)

            # 좌우 여백이 크게 잡히는 CCTV/부분 차량 프레임에서만 선택적으로 사용한다.
            tight_bbox = self._shrink_box_horizontal(
                tight_bbox,
                self.shrink_left,
                self.shrink_right,
                width,
                height,
            )

            bx1, by1, bx2, by2 = tight_bbox
            if bx2 <= bx1 or by2 <= by1:
                continue

            detections.append(
                VehicleDetection(
                    class_name=class_name,
                    confidence=confidence,
                    bbox=tight_bbox,
                )
            )

        detections = self._deduplicate_detections(detections)
        return sorted(detections, key=lambda det: (det.bbox[0], det.bbox[1]))


def read_source_frame(source: Path, frame_index: int = 0) -> np.ndarray:
    if source.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        frame = cv2.imread(str(source))
        if frame is None:
            raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {source}")
        return frame

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise FileNotFoundError(f"영상을 읽을 수 없습니다: {source}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count > 0:
        frame_index = max(0, min(frame_index, frame_count - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"{frame_index}번 프레임을 읽지 못했습니다: {source}")
    return frame
