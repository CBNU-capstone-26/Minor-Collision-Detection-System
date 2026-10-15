"""라이브 차량 탐지 모듈 (웹 서비스 /detect-vehicles 에서 사용).

YOLO(권장: segmentation 모델)로 프레임의 차량을 탐지하고, segmentation mask의
실제 외곽으로 상/하/좌/우가 차량 엣지에 맞는 bbox를 계산한다. detection 모델을
쓰면 마스크가 없어 YOLO detection bbox로 폴백한다.

- 라이브 진입점: get_detector(...) 로 탐지기를 프로세스당 1회 로드·재사용.
- 반환: VehicleDetection(class_name, confidence, bbox=(x1,y1,x2,y2)) 목록.
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
        mask_threshold: float = 0.6,
        min_area: int = 100,
        nms_iou: float = 0.35,
        contain_threshold: float = 0.85,
        imgsz: int = 640,
        enhance_night: bool = False,
        dynamic_imgsz: bool = False,
        imgsz_min: int = 640,
    ):
        self.model = YOLO(model_path)
        self.conf = conf
        # imgsz 의미:
        #   dynamic_imgsz=False → 항상 이 값으로 추론(고정).
        #   dynamic_imgsz=True  → 프레임 긴 변에 맞춘 동적 imgsz의 '상한(cap)'.
        #     eff = clamp(round32(long_side), imgsz_min, imgsz)
        #     → 고해상도는 과다 다운스케일(정보손실) 방지, 저해상도는 업스케일 낭비 방지.
        self.imgsz = imgsz              # 추론 입력 해상도(동적 모드에선 상한)
        self.dynamic_imgsz = dynamic_imgsz
        self.imgsz_min = imgsz_min
        self.enhance_night = enhance_night  # 어두운 프레임 CLAHE/감마 전처리 여부
        self.vehicle_classes = set(vehicle_classes)
        self.mask_threshold = mask_threshold  # 마스크 이진화 임계값(↑ = bbox 타이트)
        self.min_area = min_area
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

    def _effective_imgsz(self, width: int, height: int) -> int:
        """동적 모드면 프레임 긴 변을 32배수로 맞춰 [imgsz_min, imgsz]로 clamp."""
        if not self.dynamic_imgsz:
            return self.imgsz
        long_side = max(width, height)
        snapped = int(round(long_side / 32.0)) * 32
        return max(self.imgsz_min, min(snapped, self.imgsz))

    def detect_frame(self, frame_bgr: np.ndarray) -> list[VehicleDetection]:
        if self.enhance_night:
            frame_bgr = self._enhance_low_light(frame_bgr)
        height, width = frame_bgr.shape[:2]
        eff_imgsz = self._effective_imgsz(width, height)
        result = self.model(frame_bgr, conf=self.conf,
                             imgsz=eff_imgsz, verbose=False)[0]

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

            # (엣지 정밀도는 마스크로 결정 — blind 좌우 shrink 휴리스틱은 사용하지 않음)
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


# 프로세스당 탐지기(YOLO 모델)를 1회만 로드해 재사용 — 매 요청 재로드 오버헤드 제거.
_detector_cache: dict = {}


def get_detector(
    model_path: str = "yolo11x.pt",
    conf: float = 0.15,
    imgsz: int = 1536,
    enhance_night: bool = True,
    mask_threshold: float = 0.6,
    dynamic_imgsz: bool = True,
    imgsz_min: int = 640,
) -> "VehicleBBoxDetector":
    """설정별로 VehicleBBoxDetector를 캐시해 반환한다(같은 설정이면 재사용).

    YOLO 가중치 로드(≈수 초)를 요청마다 반복하지 않도록 프로세스 전역에 보관한다.
    기본은 동적 imgsz(상한 1536): 프레임 해상도에 맞춰 imgsz를 정해 고해상도
    정보손실과 저해상도 업스케일 낭비를 동시에 줄인다.
    """
    key = (model_path, conf, imgsz, enhance_night, mask_threshold,
           dynamic_imgsz, imgsz_min)
    detector = _detector_cache.get(key)
    if detector is None:
        detector = VehicleBBoxDetector(
            model_path=model_path, conf=conf, imgsz=imgsz,
            enhance_night=enhance_night, mask_threshold=mask_threshold,
            dynamic_imgsz=dynamic_imgsz, imgsz_min=imgsz_min)
        _detector_cache[key] = detector
    return detector
