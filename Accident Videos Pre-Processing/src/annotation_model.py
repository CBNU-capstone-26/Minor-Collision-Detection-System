from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

JSON_INDENT = 4


class AnnotationError(ValueError):
    """Raised when an annotation cannot be used for paper-format export."""


@dataclass
class Box:
    vehicle_id: int
    bbox: list[int]
    reference_frame: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Box":
        return cls(int(data["vehicle_id"]), [int(v) for v in data["bbox"]], int(data["reference_frame"]))


@dataclass
class Event:
    event_id: str
    vehicle_id: int
    start_frame: int
    end_frame: int
    status: str = "needs_review"
    note: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Event":
        return cls(
            str(data["event_id"]),
            int(data["vehicle_id"]),
            int(data["start_frame"]),
            int(data["end_frame"]),
            str(data.get("status", "needs_review")),
            str(data.get("note", "")),
        )


@dataclass
class VideoAnnotation:
    video_id: str
    source_video: str
    split: str
    width: int
    height: int
    frame_count: int
    fps: float
    boxes: list[Box] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    status: str = "not_started"
    schema_version: int = 1

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VideoAnnotation":
        return cls(
            video_id=str(data["video_id"]),
            source_video=str(data["source_video"]),
            split=str(data.get("split", "unknown")),
            width=int(data["width"]),
            height=int(data["height"]),
            frame_count=int(data["frame_count"]),
            fps=float(data["fps"]),
            boxes=[Box.from_dict(item) for item in data.get("boxes", [])],
            events=[Event.from_dict(item) for item in data.get("events", [])],
            status=str(data.get("status", "not_started")),
            schema_version=int(data.get("schema_version", 1)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_annotations(path: str | Path) -> dict[str, VideoAnnotation]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise AnnotationError("unsupported annotation schema_version")
    return {item.video_id: item for item in (VideoAnnotation.from_dict(row) for row in payload.get("videos", []))}


def save_annotations(path: str | Path, annotations: dict[str, VideoAnnotation]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "videos": [item.to_dict() for item in annotations.values()]}
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=JSON_INDENT) + "\n", encoding="utf-8")


def validate_annotation(annotation: VideoAnnotation) -> list[str]:
    errors: list[str] = []
    if annotation.width <= 0 or annotation.height <= 0:
        errors.append("video dimensions must be positive")
    if annotation.frame_count <= 0:
        errors.append("frame_count must be positive")
    if annotation.status not in {"not_started", "in_progress", "confirmed", "excluded"}:
        errors.append(f"invalid video status {annotation.status!r}")
    ids: set[int] = set()
    for box in annotation.boxes:
        if box.vehicle_id in ids:
            errors.append(f"duplicate vehicle_id: {box.vehicle_id}")
        ids.add(box.vehicle_id)
        if len(box.bbox) != 4:
            errors.append(f"vehicle {box.vehicle_id} bbox must have four coordinates")
            continue
        x1, y1, x2, y2 = box.bbox
        if not (0 <= x1 < x2 <= annotation.width and 0 <= y1 < y2 <= annotation.height):
            errors.append(f"vehicle {box.vehicle_id} bbox is outside video bounds")
        if not 0 <= box.reference_frame < annotation.frame_count:
            errors.append(f"vehicle {box.vehicle_id} reference_frame is outside video bounds")
    for event in annotation.events:
        if event.vehicle_id not in ids:
            errors.append(f"event {event.event_id} references missing vehicle {event.vehicle_id}")
        if event.start_frame < 0 or event.end_frame >= annotation.frame_count:
            errors.append(f"event {event.event_id} frame range is outside video bounds")
        if event.start_frame > event.end_frame:
            errors.append(f"event {event.event_id} has start_frame after end_frame")
        if event.status not in {"needs_review", "confirmed", "excluded"}:
            errors.append(f"event {event.event_id} has invalid status {event.status!r}")
    event_ids = [event.event_id for event in annotation.events]
    if len(event_ids) != len(set(event_ids)):
        errors.append("duplicate event_id")
    return errors


def paper_txt(annotation: VideoAnnotation, event: Event) -> str:
    errors = validate_annotation(annotation)
    if errors:
        raise AnnotationError("; ".join(errors))
    if event.status != "confirmed":
        raise AnnotationError(f"event {event.event_id} is not confirmed")
    lines = [f"car,{box.vehicle_id},{box.bbox[0]},{box.bbox[1]},{box.bbox[2]},{box.bbox[3]}" for box in annotation.boxes]
    lines.append(f"A,{event.vehicle_id},{event.start_frame},{event.end_frame}")
    return "\n".join(lines) + "\n"


def event_output_stem(video_id: str, event_index: int) -> str:
    return video_id if event_index == 0 else f"{video_id}__a{event_index + 1}"
