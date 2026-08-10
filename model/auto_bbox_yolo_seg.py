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

import argparse
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
    ):
        self.model = YOLO(model_path)
        self.conf = conf
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

    def detect_frame(self, frame_bgr: np.ndarray) -> list[VehicleDetection]:
        height, width = frame_bgr.shape[:2]
        result = self.model(frame_bgr, conf=self.conf, verbose=False)[0]

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
        description="YOLO segmentation mask 기반 차량 bbox 자동 생성"
    )
    parser.add_argument("--source", required=True, help="입력 이미지 또는 영상 경로")
    parser.add_argument(
        "--output-txt",
        required=True,
        help="저장할 txt 경로. 형식: car,id,x1,y1,x2,y2",
    )
    parser.add_argument("--output-image", help="bbox 확인용 이미지 저장 경로")
    parser.add_argument("--model", default="yolov8n-seg.pt", help="YOLO-seg 모델 경로")
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO confidence")
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
        help="YOLO 검출 대신 사용할 bbox. 형식: x1,y1,x2,y2",
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
        detector = VehicleBBoxDetector(
            model_path=args.model,
            conf=args.conf,
            min_area=args.min_area,
            shrink_x=args.shrink_x,
            shrink_left=args.shrink_left,
            shrink_right=args.shrink_right,
            nms_iou=args.nms_iou,
            contain_threshold=args.contain_threshold,
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
