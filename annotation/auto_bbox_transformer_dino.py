"""
RT-DETR Transformer detector 기반 차량 bbox 자동 생성 도구.

목적:
  - RT-DETR로 차량을 찾는다.
  - segmentation mask/contour 변환 없이 detector가 예측한 bbox를 직접 사용한다.
  - 프로젝트 학습 txt 포맷인 `car,id,x1,y1,x2,y2`로 저장한다.

설치:
  pip install transformers pillow opencv-python torch

예시:
  python annotation/auto_bbox_transformer_dino.py \
    --source data/real/real01.mp4 \
    --output-image outputs/real01_auto_bbox.jpg \
    --output-txt data/real/real01.txt \
    --frame-index 0

  # Transformer detector가 부분 차량을 못 잡았을 때: 기존 bbox를 넣고 좌우만 타이트하게 보정
  python annotation/auto_bbox_transformer_dino.py \
    --source outputs/sample.jpg \
    --output-image outputs/sample_tight_bbox.jpg \
    --output-txt outputs/sample_tight_bbox.txt \
    --manual-bbox 190,0,1240,965 \
    --shrink-left 0.02 \
    --shrink-right 0.12
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


DEFAULT_VEHICLE_CLASSES = (
    "car",
    "parked car",
    "truck",
    "bus",
    "van",
    "suv",
    "sedan",
    "automobile",
)
DEFAULT_DINO_MODEL = "IDEA-Research/grounding-dino-base"
DEFAULT_RTDETR_MODEL = "rtdetr-x.pt"
DEFAULT_YOLO_MODEL = "yolo11x.pt"
DEFAULT_YOLO_SEG_MODEL = "yolov8m-seg.pt"
DEFAULT_DETECTOR_IMAGE_SIZE = 1280
YOLO_VEHICLE_CLASS_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
DEFAULT_TRANSFORMER_MIN_AREA_RATIO = 0.01
DEFAULT_TRANSFORMER_MAX_AREA_RATIO = 0.80
DEFAULT_TRANSFORMER_MIN_WIDTH = 50
DEFAULT_TRANSFORMER_MIN_HEIGHT = 35
DEFAULT_TRANSFORMER_MIN_ASPECT_RATIO = 1.0
DEFAULT_TRANSFORMER_MAX_ASPECT_RATIO = 5.5
DEFAULT_EDGE_MARGIN_RATIO = 0.04
DEFAULT_EDGE_MIN_AREA_RATIO = 0.0025
DEFAULT_EDGE_MIN_WIDTH = 20
DEFAULT_EDGE_MIN_HEIGHT = 20
DEFAULT_EDGE_MIN_ASPECT_RATIO = 0.25
DEFAULT_EDGE_MAX_ASPECT_RATIO = 8.0
DEFAULT_RTDETR_EDGE_CONF = 0.12
DEFAULT_YOLO_EDGE_CONF = 0.18
DEFAULT_MAX_DARK_PIXEL_RATIO = 0.85
DEFAULT_MIN_MEAN_LUMA = 35.0
DEFAULT_SEG_MIN_REFINED_AREA_RATIO = 0.20
DEFAULT_DINO_MIN_AREA_RATIO = DEFAULT_TRANSFORMER_MIN_AREA_RATIO
DEFAULT_DINO_MAX_AREA_RATIO = DEFAULT_TRANSFORMER_MAX_AREA_RATIO
DEFAULT_DINO_MIN_WIDTH = DEFAULT_TRANSFORMER_MIN_WIDTH
DEFAULT_DINO_MIN_HEIGHT = DEFAULT_TRANSFORMER_MIN_HEIGHT
DEFAULT_DINO_MIN_ASPECT_RATIO = DEFAULT_TRANSFORMER_MIN_ASPECT_RATIO
DEFAULT_DINO_MAX_ASPECT_RATIO = DEFAULT_TRANSFORMER_MAX_ASPECT_RATIO


@dataclass
class VehicleDetection:
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]
    source: str = "dino"


class VehicleBBoxDetector:
    """Grounding DINO가 직접 예측한 차량 bbox를 프로젝트 포맷으로 변환한다."""

    def __init__(
        self,
        model_path: str = DEFAULT_DINO_MODEL,
        conf: float = 0.12,
        vehicle_classes: Iterable[str] = DEFAULT_VEHICLE_CLASSES,
        text_threshold: float = 0.10,
        min_area: int = 100,
        min_area_ratio: float = DEFAULT_DINO_MIN_AREA_RATIO,
        max_area_ratio: float = DEFAULT_DINO_MAX_AREA_RATIO,
        min_width: int = DEFAULT_DINO_MIN_WIDTH,
        min_height: int = DEFAULT_DINO_MIN_HEIGHT,
        min_aspect_ratio: float = DEFAULT_DINO_MIN_ASPECT_RATIO,
        max_aspect_ratio: float = DEFAULT_DINO_MAX_ASPECT_RATIO,
        max_dark_pixel_ratio: float = DEFAULT_MAX_DARK_PIXEL_RATIO,
        min_mean_luma: float = DEFAULT_MIN_MEAN_LUMA,
        shrink_x: float = 0.0,
        shrink_left: float | None = None,
        shrink_right: float | None = None,
        nms_iou: float = 0.35,
        contain_threshold: float = 0.85,
        device: str = "auto",
        local_files_only: bool = False,
    ):
        try:
            import torch
            from PIL import Image
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:  # pragma: no cover - runtime dependency 안내용
            raise SystemExit(
                "Grounding DINO 실행 의존성이 설치되어 있지 않습니다. "
                "다음 명령으로 설치하세요: "
                "pip install transformers pillow opencv-python torch"
            ) from exc

        self.torch = torch
        self.Image = Image
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=local_files_only,
        )
        self.device = self._resolve_device(device)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_path,
            local_files_only=local_files_only,
        )
        self.model.to(self.device)
        self.model.eval()
        self.conf = conf
        self.vehicle_classes = tuple(dict.fromkeys(vehicle_classes))
        self.vehicle_class_set = set(self.vehicle_classes)
        self.text_threshold = text_threshold
        self.min_area = min_area
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.min_width = min_width
        self.min_height = min_height
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.max_dark_pixel_ratio = max_dark_pixel_ratio
        self.min_mean_luma = min_mean_luma
        self.shrink_left = shrink_x if shrink_left is None else shrink_left
        self.shrink_right = shrink_x if shrink_right is None else shrink_right
        self.nms_iou = nms_iou
        self.contain_threshold = contain_threshold
        self.text_labels = [[f"a {class_name}" for class_name in self.vehicle_classes]]

    def _resolve_device(self, requested: str) -> str:
        if requested != "auto":
            return requested
        if self.torch.cuda.is_available():
            return "cuda"
        return "cpu"

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

    @staticmethod
    def _box_width_height(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
        x1, y1, x2, y2 = bbox
        return max(0, x2 - x1), max(0, y2 - y1)

    @classmethod
    def _box_area_ratio(
        cls,
        bbox: tuple[int, int, int, int],
        frame_width: int,
        frame_height: int,
    ) -> float:
        frame_area = max(1, frame_width * frame_height)
        return cls._box_area(bbox) / frame_area

    @classmethod
    def _aspect_ratio(cls, bbox: tuple[int, int, int, int]) -> float:
        box_width, box_height = cls._box_width_height(bbox)
        return box_width / box_height if box_height > 0 else 0.0

    @staticmethod
    def _touches_frame_edge(
        bbox: tuple[int, int, int, int],
        frame_width: int,
        frame_height: int,
        margin_ratio: float = DEFAULT_EDGE_MARGIN_RATIO,
    ) -> bool:
        x1, y1, x2, y2 = bbox
        margin_x = max(1, int(frame_width * margin_ratio))
        margin_y = max(1, int(frame_height * margin_ratio))
        return x1 <= margin_x or y1 <= margin_y or x2 >= frame_width - margin_x or y2 >= frame_height - margin_y

    @staticmethod
    def _is_mostly_black_region(
        frame_bgr: np.ndarray,
        bbox: tuple[int, int, int, int],
        max_dark_pixel_ratio: float = DEFAULT_MAX_DARK_PIXEL_RATIO,
        min_mean_luma: float = DEFAULT_MIN_MEAN_LUMA,
    ) -> bool:
        x1, y1, x2, y2 = bbox
        roi = frame_bgr[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if roi.size == 0:
            return True
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        dark_ratio = float((gray <= 20).mean())
        mean_luma = float(gray.mean())
        return dark_ratio >= max_dark_pixel_ratio and mean_luma <= min_mean_luma

    def _passes_dino_vehicle_shape_filter(
        self,
        bbox: tuple[int, int, int, int],
        frame_width: int,
        frame_height: int,
    ) -> bool:
        box_width, box_height = self._box_width_height(bbox)
        if box_width < self.min_width or box_height < self.min_height:
            return False
        if self._box_area_ratio(bbox, frame_width, frame_height) < self.min_area_ratio:
            return False
        if self._box_area_ratio(bbox, frame_width, frame_height) > self.max_area_ratio:
            return False
        aspect_ratio = self._aspect_ratio(bbox)
        return self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio

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

    def _label_to_class(self, label: str) -> str | None:
        normalized = label.strip().lower()
        if normalized.startswith("a "):
            normalized = normalized[2:]
        if normalized.startswith("an "):
            normalized = normalized[3:]
        return normalized if normalized in self.vehicle_class_set else None

    def detect_frame(self, frame_bgr: np.ndarray) -> list[VehicleDetection]:
        height, width = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self.Image.fromarray(frame_rgb)

        inputs = self.processor(
            images=image,
            text=self.text_labels,
            return_tensors="pt",
        ).to(self.device)

        with self.torch.inference_mode():
            outputs = self.model(**inputs)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*`labels`.*`text_labels`.*",
                category=FutureWarning,
            )
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=self.conf,
                text_threshold=self.text_threshold,
                target_sizes=[(height, width)],
            )[0]

        detections: list[VehicleDetection] = []
        labels = results["text_labels"] if "text_labels" in results else results["labels"]
        for score, label, box in zip(
            results["scores"],
            labels,
            results["boxes"],
        ):
            class_name = self._label_to_class(str(label))
            if class_name is None:
                continue

            x1, y1, x2, y2 = box.detach().cpu().tolist()
            bbox = self._clip_box(x1, y1, x2, y2, width, height)
            if self._box_area(bbox) < self.min_area:
                continue

            # 좌우 여백이 크게 잡히는 CCTV/부분 차량 프레임에서만 선택적으로 사용한다.
            bbox = self._shrink_box_horizontal(
                bbox,
                self.shrink_left,
                self.shrink_right,
                width,
                height,
            )

            bx1, by1, bx2, by2 = bbox
            if bx2 <= bx1 or by2 <= by1:
                continue
            if not self._passes_dino_vehicle_shape_filter(bbox, width, height):
                continue
            if self._is_mostly_black_region(
                frame_bgr,
                bbox,
                self.max_dark_pixel_ratio,
                self.min_mean_luma,
            ):
                continue

            detections.append(
                VehicleDetection(
                    class_name=class_name,
                    confidence=float(score.detach().cpu()),
                    bbox=bbox,
                )
            )

        detections = self._deduplicate_detections(detections)
        return sorted(detections, key=lambda det: (det.bbox[0], det.bbox[1]))


class RTDETRVehicleBBoxDetector:
    """RT-DETR가 직접 예측한 차량 bbox를 프로젝트 포맷으로 변환한다."""

    def __init__(
        self,
        model_path: str = DEFAULT_RTDETR_MODEL,
        conf: float = 0.20,
        min_area: int = 100,
        min_area_ratio: float = DEFAULT_TRANSFORMER_MIN_AREA_RATIO,
        max_area_ratio: float = DEFAULT_TRANSFORMER_MAX_AREA_RATIO,
        min_width: int = DEFAULT_TRANSFORMER_MIN_WIDTH,
        min_height: int = DEFAULT_TRANSFORMER_MIN_HEIGHT,
        min_aspect_ratio: float = DEFAULT_TRANSFORMER_MIN_ASPECT_RATIO,
        max_aspect_ratio: float = DEFAULT_TRANSFORMER_MAX_ASPECT_RATIO,
        edge_conf: float = DEFAULT_RTDETR_EDGE_CONF,
        edge_margin_ratio: float = DEFAULT_EDGE_MARGIN_RATIO,
        edge_min_area_ratio: float = DEFAULT_EDGE_MIN_AREA_RATIO,
        edge_min_width: int = DEFAULT_EDGE_MIN_WIDTH,
        edge_min_height: int = DEFAULT_EDGE_MIN_HEIGHT,
        edge_min_aspect_ratio: float = DEFAULT_EDGE_MIN_ASPECT_RATIO,
        edge_max_aspect_ratio: float = DEFAULT_EDGE_MAX_ASPECT_RATIO,
        max_dark_pixel_ratio: float = DEFAULT_MAX_DARK_PIXEL_RATIO,
        min_mean_luma: float = DEFAULT_MIN_MEAN_LUMA,
        nms_iou: float = 0.45,
        contain_threshold: float = 0.85,
        device: str = "auto",
    ):
        try:
            from ultralytics import RTDETR
        except ImportError as exc:  # pragma: no cover - runtime dependency 안내용
            raise SystemExit(
                "RT-DETR 실행 의존성이 설치되어 있지 않습니다. "
                "다음 명령으로 설치하세요: pip install ultralytics"
            ) from exc

        self.model = RTDETR(model_path)
        self.conf = conf
        self.min_area = min_area
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.min_width = min_width
        self.min_height = min_height
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.edge_conf = edge_conf
        self.edge_margin_ratio = edge_margin_ratio
        self.edge_min_area_ratio = edge_min_area_ratio
        self.edge_min_width = edge_min_width
        self.edge_min_height = edge_min_height
        self.edge_min_aspect_ratio = edge_min_aspect_ratio
        self.edge_max_aspect_ratio = edge_max_aspect_ratio
        self.max_dark_pixel_ratio = max_dark_pixel_ratio
        self.min_mean_luma = min_mean_luma
        self.nms_iou = nms_iou
        self.contain_threshold = contain_threshold
        self.device = None if device == "auto" else device

    def detect_frame(self, frame_bgr: np.ndarray) -> list[VehicleDetection]:
        detections = self._detect_frame_with_conf(frame_bgr, self.conf, edge_only=False)
        if self.edge_conf < self.conf:
            detections.extend(self._detect_frame_with_conf(frame_bgr, self.edge_conf, edge_only=True))
        return self._deduplicate_detections(detections)

    def _detect_frame_with_conf(
        self,
        frame_bgr: np.ndarray,
        conf: float,
        edge_only: bool,
    ) -> list[VehicleDetection]:
        height, width = frame_bgr.shape[:2]
        results = self.model.predict(
            source=frame_bgr,
            imgsz=DEFAULT_DETECTOR_IMAGE_SIZE,
            conf=conf,
            classes=sorted(YOLO_VEHICLE_CLASS_IDS),
            device=self.device,
            verbose=False,
        )
        detections: list[VehicleDetection] = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None:
            return detections

        for box in boxes:
            cls_id = int(box.cls.detach().cpu().item())
            class_name = YOLO_VEHICLE_CLASS_IDS.get(cls_id)
            if class_name is None:
                continue
            x1, y1, x2, y2 = box.xyxy[0].detach().cpu().tolist()
            bbox = VehicleBBoxDetector._clip_box(x1, y1, x2, y2, width, height)
            if VehicleBBoxDetector._box_area(bbox) < self.min_area:
                continue
            is_edge_candidate = VehicleBBoxDetector._touches_frame_edge(
                bbox,
                width,
                height,
                self.edge_margin_ratio,
            )
            if edge_only and not is_edge_candidate:
                continue
            if not self._passes_vehicle_shape_filter(bbox, width, height, is_edge_candidate):
                continue
            if VehicleBBoxDetector._is_mostly_black_region(
                frame_bgr,
                bbox,
                self.max_dark_pixel_ratio,
                self.min_mean_luma,
            ):
                continue
            detections.append(
                VehicleDetection(
                    class_name=class_name,
                    confidence=float(box.conf.detach().cpu().item()),
                    bbox=bbox,
                    source="rtdetr",
                )
            )

        return detections

    def _passes_vehicle_shape_filter(
        self,
        bbox: tuple[int, int, int, int],
        frame_width: int,
        frame_height: int,
        is_edge_candidate: bool = False,
    ) -> bool:
        box_width, box_height = VehicleBBoxDetector._box_width_height(bbox)
        min_width = self.edge_min_width if is_edge_candidate else self.min_width
        min_height = self.edge_min_height if is_edge_candidate else self.min_height
        min_area_ratio = self.edge_min_area_ratio if is_edge_candidate else self.min_area_ratio
        min_aspect_ratio = self.edge_min_aspect_ratio if is_edge_candidate else self.min_aspect_ratio
        max_aspect_ratio = self.edge_max_aspect_ratio if is_edge_candidate else self.max_aspect_ratio
        if box_width < min_width or box_height < min_height:
            return False
        area_ratio = VehicleBBoxDetector._box_area_ratio(bbox, frame_width, frame_height)
        if area_ratio < min_area_ratio:
            return False
        if area_ratio > self.max_area_ratio:
            return False
        aspect_ratio = VehicleBBoxDetector._aspect_ratio(bbox)
        return min_aspect_ratio <= aspect_ratio <= max_aspect_ratio

    def _deduplicate_detections(
        self,
        detections: list[VehicleDetection],
    ) -> list[VehicleDetection]:
        kept: list[VehicleDetection] = []
        for det in sorted(detections, key=lambda item: item.confidence, reverse=True):
            duplicate = False
            for kept_det in kept:
                iou = VehicleBBoxDetector._iou(det.bbox, kept_det.bbox)
                contain = VehicleBBoxDetector._contained_ratio(det.bbox, kept_det.bbox)
                if iou >= self.nms_iou or contain >= self.contain_threshold:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(det)
        return sorted(kept, key=lambda det: (det.bbox[0], det.bbox[1]))


class YoloFallbackVehicleBBoxDetector:
    """Transformer detector가 차량을 충분히 못 잡았을 때 보강용으로 쓰는 YOLO detector."""

    def __init__(
        self,
        model_path: str = DEFAULT_YOLO_MODEL,
        conf: float = 0.25,
        min_area: int = 100,
        edge_conf: float = DEFAULT_YOLO_EDGE_CONF,
        edge_margin_ratio: float = DEFAULT_EDGE_MARGIN_RATIO,
        edge_min_area_ratio: float = DEFAULT_EDGE_MIN_AREA_RATIO,
        edge_min_width: int = DEFAULT_EDGE_MIN_WIDTH,
        edge_min_height: int = DEFAULT_EDGE_MIN_HEIGHT,
        edge_min_aspect_ratio: float = DEFAULT_EDGE_MIN_ASPECT_RATIO,
        edge_max_aspect_ratio: float = DEFAULT_EDGE_MAX_ASPECT_RATIO,
        max_dark_pixel_ratio: float = DEFAULT_MAX_DARK_PIXEL_RATIO,
        min_mean_luma: float = DEFAULT_MIN_MEAN_LUMA,
        nms_iou: float = 0.45,
        contain_threshold: float = 0.85,
    ):
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - runtime dependency 안내용
            raise SystemExit(
                "YOLO fallback 실행 의존성이 설치되어 있지 않습니다. "
                "다음 명령으로 설치하세요: pip install ultralytics"
            ) from exc

        self.model = YOLO(model_path)
        self.conf = conf
        self.min_area = min_area
        self.edge_conf = edge_conf
        self.edge_margin_ratio = edge_margin_ratio
        self.edge_min_area_ratio = edge_min_area_ratio
        self.edge_min_width = edge_min_width
        self.edge_min_height = edge_min_height
        self.edge_min_aspect_ratio = edge_min_aspect_ratio
        self.edge_max_aspect_ratio = edge_max_aspect_ratio
        self.max_dark_pixel_ratio = max_dark_pixel_ratio
        self.min_mean_luma = min_mean_luma
        self.nms_iou = nms_iou
        self.contain_threshold = contain_threshold

    def detect_frame(self, frame_bgr: np.ndarray) -> list[VehicleDetection]:
        detections = self._detect_frame_with_conf(frame_bgr, self.conf, edge_only=False)
        if self.edge_conf < self.conf:
            detections.extend(self._detect_frame_with_conf(frame_bgr, self.edge_conf, edge_only=True))
        return self._deduplicate_detections(detections)

    def _detect_frame_with_conf(
        self,
        frame_bgr: np.ndarray,
        conf: float,
        edge_only: bool,
    ) -> list[VehicleDetection]:
        height, width = frame_bgr.shape[:2]
        results = self.model.predict(
            source=frame_bgr,
            imgsz=DEFAULT_DETECTOR_IMAGE_SIZE,
            conf=conf,
            classes=sorted(YOLO_VEHICLE_CLASS_IDS),
            verbose=False,
        )
        detections: list[VehicleDetection] = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None:
            return detections

        for box in boxes:
            cls_id = int(box.cls.detach().cpu().item())
            class_name = YOLO_VEHICLE_CLASS_IDS.get(cls_id)
            if class_name is None:
                continue
            x1, y1, x2, y2 = box.xyxy[0].detach().cpu().tolist()
            bbox = VehicleBBoxDetector._clip_box(x1, y1, x2, y2, width, height)
            if VehicleBBoxDetector._box_area(bbox) < self.min_area:
                continue
            is_edge_candidate = VehicleBBoxDetector._touches_frame_edge(
                bbox,
                width,
                height,
                self.edge_margin_ratio,
            )
            if edge_only and not is_edge_candidate:
                continue
            if not self._passes_edge_shape_filter(bbox, width, height, is_edge_candidate):
                continue
            if VehicleBBoxDetector._is_mostly_black_region(
                frame_bgr,
                bbox,
                self.max_dark_pixel_ratio,
                self.min_mean_luma,
            ):
                continue
            detections.append(
                VehicleDetection(
                    class_name=class_name,
                    confidence=float(box.conf.detach().cpu().item()),
                    bbox=bbox,
                    source="yolo",
                )
            )

        return detections

    def _passes_edge_shape_filter(
        self,
        bbox: tuple[int, int, int, int],
        frame_width: int,
        frame_height: int,
        is_edge_candidate: bool,
    ) -> bool:
        if not is_edge_candidate:
            return True
        box_width, box_height = VehicleBBoxDetector._box_width_height(bbox)
        if box_width < self.edge_min_width or box_height < self.edge_min_height:
            return False
        area_ratio = VehicleBBoxDetector._box_area_ratio(bbox, frame_width, frame_height)
        if area_ratio < self.edge_min_area_ratio:
            return False
        if area_ratio > DEFAULT_TRANSFORMER_MAX_AREA_RATIO:
            return False
        aspect_ratio = VehicleBBoxDetector._aspect_ratio(bbox)
        return self.edge_min_aspect_ratio <= aspect_ratio <= self.edge_max_aspect_ratio

    def _deduplicate_detections(
        self,
        detections: list[VehicleDetection],
    ) -> list[VehicleDetection]:
        kept: list[VehicleDetection] = []
        for det in sorted(detections, key=lambda item: item.confidence, reverse=True):
            duplicate = False
            for kept_det in kept:
                iou = VehicleBBoxDetector._iou(det.bbox, kept_det.bbox)
                contain = VehicleBBoxDetector._contained_ratio(det.bbox, kept_det.bbox)
                if iou >= self.nms_iou or contain >= self.contain_threshold:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(det)
        return sorted(kept, key=lambda det: (det.bbox[0], det.bbox[1]))


class YoloSegBBoxRefiner:
    """탐지 후보 bbox를 YOLOv8-seg mask 외곽 bbox로 타이트하게 보정한다."""

    def __init__(
        self,
        model_path: str = DEFAULT_YOLO_SEG_MODEL,
        conf: float = 0.25,
        mask_threshold: float = 0.5,
        min_area: int = 100,
        min_area_ratio: float = DEFAULT_TRANSFORMER_MIN_AREA_RATIO,
        max_area_ratio: float = DEFAULT_TRANSFORMER_MAX_AREA_RATIO,
        min_width: int = DEFAULT_TRANSFORMER_MIN_WIDTH,
        min_height: int = DEFAULT_TRANSFORMER_MIN_HEIGHT,
        min_aspect_ratio: float = DEFAULT_TRANSFORMER_MIN_ASPECT_RATIO,
        max_aspect_ratio: float = DEFAULT_TRANSFORMER_MAX_ASPECT_RATIO,
        min_refined_area_ratio: float = DEFAULT_SEG_MIN_REFINED_AREA_RATIO,
        match_iou: float = 0.20,
        match_contain_threshold: float = 0.45,
        max_dark_pixel_ratio: float = DEFAULT_MAX_DARK_PIXEL_RATIO,
        min_mean_luma: float = DEFAULT_MIN_MEAN_LUMA,
    ):
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - runtime dependency 안내용
            raise SystemExit(
                "YOLO segmentation 실행 의존성이 설치되어 있지 않습니다. "
                "다음 명령으로 설치하세요: pip install ultralytics"
            ) from exc

        self.model = YOLO(model_path)
        self.conf = conf
        self.mask_threshold = mask_threshold
        self.min_area = min_area
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.min_width = min_width
        self.min_height = min_height
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_refined_area_ratio = min_refined_area_ratio
        self.match_iou = match_iou
        self.match_contain_threshold = match_contain_threshold
        self.max_dark_pixel_ratio = max_dark_pixel_ratio
        self.min_mean_luma = min_mean_luma

    def refine_frame(
        self,
        frame_bgr: np.ndarray,
        detections: list[VehicleDetection],
    ) -> list[VehicleDetection]:
        if not detections:
            return detections

        height, width = frame_bgr.shape[:2]
        seg_detections = self._segment_vehicle_boxes(frame_bgr, width, height)
        if not seg_detections:
            return detections

        refined: list[VehicleDetection] = []
        used_seg_indexes: set[int] = set()
        for det in detections:
            best_idx = None
            best_score = 0.0
            for idx, seg_det in enumerate(seg_detections):
                if idx in used_seg_indexes:
                    continue
                iou = VehicleBBoxDetector._iou(det.bbox, seg_det.bbox)
                seg_inside_candidate = VehicleBBoxDetector._contained_ratio(seg_det.bbox, det.bbox)
                candidate_inside_seg = VehicleBBoxDetector._contained_ratio(det.bbox, seg_det.bbox)
                match_score = max(iou, seg_inside_candidate * 0.8, candidate_inside_seg * 0.8)
                is_match = (
                    iou >= self.match_iou
                    or seg_inside_candidate >= self.match_contain_threshold
                    or candidate_inside_seg >= self.match_contain_threshold
                )
                if is_match and match_score > best_score:
                    best_idx = idx
                    best_score = match_score

            if best_idx is None:
                refined.append(det)
                continue

            seg_det = seg_detections[best_idx]
            candidate_area = max(1, VehicleBBoxDetector._box_area(det.bbox))
            refined_area = VehicleBBoxDetector._box_area(seg_det.bbox)
            if refined_area / candidate_area < self.min_refined_area_ratio:
                refined.append(det)
                continue

            used_seg_indexes.add(best_idx)
            source = f"{det.source}_seg" if det.source else "seg"
            refined.append(
                VehicleDetection(
                    class_name=det.class_name,
                    confidence=max(det.confidence, seg_det.confidence),
                    bbox=seg_det.bbox,
                    source=source,
                )
            )

        return sorted(refined, key=lambda item: (item.bbox[0], item.bbox[1]))

    def _segment_vehicle_boxes(
        self,
        frame_bgr: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> list[VehicleDetection]:
        result = self.model.predict(
            source=frame_bgr,
            imgsz=DEFAULT_DETECTOR_IMAGE_SIZE,
            conf=self.conf,
            classes=sorted(YOLO_VEHICLE_CLASS_IDS),
            verbose=False,
        )[0]
        if result.boxes is None:
            return []

        masks = result.masks.data.cpu().numpy() if result.masks is not None else None
        detections: list[VehicleDetection] = []
        for idx, box in enumerate(result.boxes):
            class_id = int(box.cls.detach().cpu().item())
            class_name = YOLO_VEHICLE_CLASS_IDS.get(class_id)
            if class_name is None:
                continue

            x1, y1, x2, y2 = box.xyxy[0].detach().cpu().tolist()
            bbox = VehicleBBoxDetector._clip_box(x1, y1, x2, y2, frame_width, frame_height)
            if masks is not None and idx < len(masks):
                mask_bbox = self._bbox_from_mask(masks[idx], frame_width, frame_height)
                if mask_bbox is not None:
                    bbox = mask_bbox

            if VehicleBBoxDetector._box_area(bbox) < self.min_area:
                continue
            if not self._passes_vehicle_shape_filter(bbox, frame_width, frame_height):
                continue
            if VehicleBBoxDetector._is_mostly_black_region(
                frame_bgr,
                bbox,
                self.max_dark_pixel_ratio,
                self.min_mean_luma,
            ):
                continue
            detections.append(
                VehicleDetection(
                    class_name=class_name,
                    confidence=float(box.conf.detach().cpu().item()),
                    bbox=bbox,
                    source="seg",
                )
            )

        return detections

    def _passes_vehicle_shape_filter(
        self,
        bbox: tuple[int, int, int, int],
        frame_width: int,
        frame_height: int,
    ) -> bool:
        box_width, box_height = VehicleBBoxDetector._box_width_height(bbox)
        if box_width < self.min_width or box_height < self.min_height:
            return False
        if VehicleBBoxDetector._box_area_ratio(bbox, frame_width, frame_height) < self.min_area_ratio:
            return False
        if VehicleBBoxDetector._box_area_ratio(bbox, frame_width, frame_height) > self.max_area_ratio:
            return False
        aspect_ratio = VehicleBBoxDetector._aspect_ratio(bbox)
        return self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio

    def _bbox_from_mask(
        self,
        mask: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> tuple[int, int, int, int] | None:
        binary = (mask > self.mask_threshold).astype(np.uint8)
        if binary.shape[:2] != (frame_height, frame_width):
            binary = cv2.resize(
                binary,
                (frame_width, frame_height),
                interpolation=cv2.INTER_NEAREST,
            )

        kernel = np.ones((3, 3), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < self.min_area:
            return None

        x, y, w, h = cv2.boundingRect(contour)
        return VehicleBBoxDetector._clip_box(x, y, x + w - 1, y + h - 1, frame_width, frame_height)


def merge_dino_yolo_detections(
    dino_detections: list[VehicleDetection],
    yolo_detections: list[VehicleDetection],
    iou_threshold: float = 0.35,
    contain_threshold: float = 0.85,
) -> list[VehicleDetection]:
    """DINO를 우선 유지하고, 겹치지 않는 YOLO fallback bbox만 추가한다."""
    merged = list(dino_detections)
    for yolo_det in yolo_detections:
        duplicate = False
        for det in merged:
            iou = VehicleBBoxDetector._iou(yolo_det.bbox, det.bbox)
            contain = max(
                VehicleBBoxDetector._contained_ratio(yolo_det.bbox, det.bbox),
                VehicleBBoxDetector._contained_ratio(det.bbox, yolo_det.bbox),
            )
            if iou >= iou_threshold or contain >= contain_threshold:
                duplicate = True
                break
        if not duplicate:
            merged.append(yolo_det)
    return sorted(merged, key=lambda det: (det.bbox[0], det.bbox[1]))


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


def save_training_txt(
    detections: list[VehicleDetection],
    output_txt: Path,
    label_class: str | None = None,
    target_id: int = 0,
    start_frame: int = 0,
) -> None:
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    with output_txt.open("w", encoding="utf-8") as file:
        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = det.bbox
            file.write(f"car,{idx},{x1},{y1},{x2},{y2}\n")
        if label_class is not None:
            file.write(f"{label_class},{target_id},{start_frame}\n")


def draw_detections(
    frame_bgr: np.ndarray,
    detections: list[VehicleDetection],
) -> np.ndarray:
    output = frame_bgr.copy()
    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(
            output,
            str(idx),
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RT-DETR Transformer detector 기반 차량 bbox 자동 생성"
    )
    parser.add_argument("--source", required=True, help="입력 이미지 또는 영상 경로")
    parser.add_argument(
        "--output-txt",
        required=True,
        help="저장할 txt 경로. 형식: car,id,x1,y1,x2,y2",
    )
    parser.add_argument("--output-image", help="bbox 확인용 이미지 저장 경로")
    parser.add_argument(
        "--model",
        default=DEFAULT_RTDETR_MODEL,
        help="RT-DETR 모델 파일 경로. 예: rtdetr-l.pt 또는 rtdetr-x.pt",
    )
    parser.add_argument("--conf", type=float, default=0.20, help="RT-DETR confidence threshold")
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.10,
        help="Grounding DINO 사용 시 text threshold",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help="Transformer 추론 디바이스",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Grounding DINO 사용 시 Hugging Face Hub에 접속하지 않고 로컬 캐시/로컬 모델 경로만 사용",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="영상 입력일 때 bbox를 생성할 프레임 인덱스",
    )
    parser.add_argument(
        "--label-class",
        choices=("A", "S"),
        help="선택 시 txt 마지막 줄에 A/S 이벤트 라벨도 함께 저장",
    )
    parser.add_argument("--target-id", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--min-area", type=int, default=100)
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=DEFAULT_DINO_MIN_AREA_RATIO,
        help="Transformer bbox 최소 프레임 면적 비율. 너무 작은 부품 탐지를 제거함.",
    )
    parser.add_argument(
        "--min-width",
        type=int,
        default=DEFAULT_DINO_MIN_WIDTH,
        help="Transformer bbox 최소 너비(px).",
    )
    parser.add_argument(
        "--min-height",
        type=int,
        default=DEFAULT_DINO_MIN_HEIGHT,
        help="Transformer bbox 최소 높이(px).",
    )
    parser.add_argument(
        "--min-aspect-ratio",
        type=float,
        default=DEFAULT_DINO_MIN_ASPECT_RATIO,
        help="Transformer bbox 최소 가로/세로 비율.",
    )
    parser.add_argument(
        "--max-aspect-ratio",
        type=float,
        default=DEFAULT_DINO_MAX_ASPECT_RATIO,
        help="Transformer bbox 최대 가로/세로 비율.",
    )
    parser.add_argument(
        "--shrink-x",
        type=float,
        default=0.0,
        help="bbox 좌우를 각각 bbox 너비의 해당 비율만큼 줄임. 예: 0.08",
    )
    parser.add_argument(
        "--shrink-left",
        type=float,
        help="bbox 왼쪽만 bbox 너비의 해당 비율만큼 줄임. 예: 0.02",
    )
    parser.add_argument(
        "--shrink-right",
        type=float,
        help="bbox 오른쪽만 bbox 너비의 해당 비율만큼 줄임. 예: 0.12",
    )
    parser.add_argument(
        "--manual-bbox",
        help="Transformer 검출 대신 사용할 bbox. 형식: x1,y1,x2,y2",
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.35,
        help="중복 bbox 제거 IoU 기준. 낮을수록 더 많이 제거함.",
    )
    parser.add_argument(
        "--contain-threshold",
        type=float,
        default=0.85,
        help="작은 bbox가 큰 bbox 안에 이 비율 이상 들어가면 중복으로 제거",
    )
    return parser.parse_args()


def parse_manual_bbox(value: str) -> tuple[int, int, int, int]:
    try:
        parts = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--manual-bbox는 x1,y1,x2,y2 숫자 형식이어야 합니다."
        ) from exc
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "--manual-bbox는 x1,y1,x2,y2 네 개 좌표가 필요합니다."
        )
    return parts[0], parts[1], parts[2], parts[3]


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    output_txt = Path(args.output_txt)

    frame = read_source_frame(source, frame_index=args.frame_index)
    if args.manual_bbox:
        height, width = frame.shape[:2]
        bbox = VehicleBBoxDetector._clip_box(*parse_manual_bbox(args.manual_bbox), width, height)
        bbox = VehicleBBoxDetector._shrink_box_horizontal(
            bbox,
            args.shrink_x if args.shrink_left is None else args.shrink_left,
            args.shrink_x if args.shrink_right is None else args.shrink_right,
            width,
            height,
        )
        detections = [VehicleDetection(class_name="car", confidence=1.0, bbox=bbox)]
    else:
        detector = RTDETRVehicleBBoxDetector(
            model_path=args.model,
            conf=args.conf,
            min_area=args.min_area,
            min_area_ratio=args.min_area_ratio,
            min_width=args.min_width,
            min_height=args.min_height,
            min_aspect_ratio=args.min_aspect_ratio,
            max_aspect_ratio=args.max_aspect_ratio,
            nms_iou=args.nms_iou,
            contain_threshold=args.contain_threshold,
            device=args.device,
        )
        detections = detector.detect_frame(frame)
    save_training_txt(
        detections,
        output_txt,
        label_class=args.label_class,
        target_id=args.target_id,
        start_frame=args.start_frame,
    )

    if args.output_image:
        output_image = Path(args.output_image)
        output_image.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_image), draw_detections(frame, detections))

    print(f"detected vehicles: {len(detections)}")
    print(f"saved txt: {output_txt}")
    if args.output_image:
        print(f"saved image: {args.output_image}")


if __name__ == "__main__":
    main()
