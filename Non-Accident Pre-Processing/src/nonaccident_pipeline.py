from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from .annotation_model import AnnotationError, Box, VideoAnnotation, load_annotations, save_annotations, validate_annotation
from .detector import track_video
from .exporter import export_nonaccident
from .tracker import Observation, _area, _overlap_ratio, choose_reference_tracks


def inventory(source_root: Path) -> list[dict]:
    paths = sorted(source_root.rglob("*.mp4")) if source_root.is_dir() else []
    used: dict[str, int] = {}
    rows = []
    for path in paths:
        base = path.stem
        used[base] = used.get(base, 0) + 1
        video_id = base if used[base] == 1 else f"{base}__copy{used[base]}"
        rows.append({"video_id": video_id, "source_video": str(path.resolve()), "split": "normal"})
    return rows


def read_video_info(path: Path) -> tuple[int, int, int, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    capture.release()
    return width, height, count, fps


def representative_boxes(observations: list[Observation], track_ids: list[int], moving_track_id: int | None = None) -> tuple[int | None, dict[int, list[int]]]:
    grouped = {track_id: [item for item in observations if item.track_id == track_id] for track_id in track_ids}
    common_frames = set.intersection(*(set(item.frame for item in items) for items in grouped.values())) if grouped else set()
    if not common_frames:
        return None, {}
    by_frame = {frame: {item.track_id: item for item in observations if item.frame == frame} for frame in common_frames}
    moving_by_frame = {item.frame: item for item in observations if moving_track_id is not None and item.track_id == moving_track_id}
    scored_frames = []
    for frame, frame_items in by_frame.items():
        moving = moving_by_frame.get(frame)
        overlap_penalty = 0.0 if moving is None else sum(_overlap_ratio(moving.bbox, frame_items[track_id].bbox) for track_id in track_ids)
        visibility_score = sum(item.confidence + min(1.0, _area(item.bbox) / 100_000.0) for item in frame_items.values())
        scored_frames.append((visibility_score - 2.0 * overlap_penalty, frame))
    frame = max(scored_frames)[1]
    boxes = {track_id: next(item.bbox for item in grouped[track_id] if item.frame == frame) for track_id in track_ids}
    return frame, boxes


def detect_one(row: dict, model_name: str) -> dict:
    source = Path(row["source_video"])
    width, height, frame_count, fps = read_video_info(source)
    observations = track_video(source, model_name)
    moving_id, reference_ids = choose_reference_tracks(observations, frame_count)
    reference_frame, boxes = representative_boxes(observations, reference_ids, moving_id)
    return {
        **row,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "fps": fps,
        "moving_track_id": moving_id,
        "reference_track_ids": reference_ids,
        "reference_frame": reference_frame,
        "boxes": boxes,
        "observations": [item.__dict__ for item in observations],
        "status": "needs_review",
        "review_note": "",
    }


def draft_to_annotation(draft: dict) -> VideoAnnotation:
    boxes = []
    if draft.get("reference_frame") is not None:
        for index, track_id in enumerate(draft.get("reference_track_ids", [])):
            key = str(track_id) if str(track_id) in draft["boxes"] else track_id
            if key in draft["boxes"]:
                boxes.append(Box(index, list(draft["boxes"][key]), int(draft["reference_frame"])))
    return VideoAnnotation(draft["video_id"], draft["source_video"], "normal", int(draft["width"]), int(draft["height"]), int(draft["frame_count"]), float(draft["fps"]), boxes=boxes, status="needs_review")


def classify_annotation_tag(annotation: VideoAnnotation) -> str:
    """Classify box completeness without claiming human visual approval."""
    errors = validate_annotation_for_tag(annotation)
    if errors:
        return "needs_review"
    return "normal"


def validate_annotation_for_tag(annotation: VideoAnnotation) -> list[str]:
    errors = []
    if len(annotation.boxes) != 2:
        errors.append("requires exactly two boxes")
    if {box.vehicle_id for box in annotation.boxes} != {0, 1}:
        errors.append("requires car 0 and car 1")
    errors.extend(validate_annotation(annotation))
    return errors


def command_inventory(args: argparse.Namespace) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema_version": 1, "videos": inventory(args.source_root)}, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
    print(f"wrote {len(inventory(args.source_root))} videos to {args.output}")


def command_detect(args: argparse.Namespace) -> None:
    payload = json.loads(args.inventory.read_text(encoding="utf-8"))
    drafts = []
    for index, row in enumerate(payload.get("videos", []), 1):
        print(f"[detect {index}/{len(payload.get('videos', []))}] {row['video_id']}: processing", flush=True)
        draft = detect_one(row, args.model)
        drafts.append(draft)
        print(f"[detect {index}/{len(payload.get('videos', []))}] {row['video_id']}: {len(draft['reference_track_ids'])} reference candidate(s)", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema_version": 1, "videos": drafts}, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")


def command_prepare(args: argparse.Namespace) -> None:
    payload = json.loads(args.drafts.read_text(encoding="utf-8"))
    annotations = {}
    for draft in payload.get("videos", []):
        annotation = draft_to_annotation(draft)
        annotation.tag = classify_annotation_tag(annotation)
        annotations[annotation.video_id] = annotation
    save_annotations(args.output, annotations)
    print(f"wrote {len(annotations)} review annotations to {args.output}")


def command_classify(args: argparse.Namespace) -> None:
    annotations = load_annotations(args.annotations)
    counts = {"normal": 0, "needs_review": 0}
    for annotation in annotations.values():
        annotation.tag = classify_annotation_tag(annotation)
        counts[annotation.tag] += 1
    save_annotations(args.output, annotations)
    print(f"classified {len(annotations)} video(s): normal={counts['normal']}, needs_review={counts['needs_review']}")
    print(f"wrote tagged annotations to {args.output}")


def command_export(args: argparse.Namespace) -> None:
    annotations = load_annotations(args.annotations)
    exported = 0
    for annotation in annotations.values():
        if annotation.status != "confirmed":
            continue
        try:
            paths = export_nonaccident(annotation, args.output_root, args.ffmpeg)
        except AnnotationError as exc:
            print(f"skip {annotation.video_id}: {exc}")
            continue
        print(f"exported {annotation.video_id}: {paths[0]} / {paths[1]}")
        exported += 1
    print(f"exported {exported} confirmed video(s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Non-accident paper-compatible annotation pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    inventory_parser = sub.add_parser("inventory"); inventory_parser.add_argument("--source-root", type=Path, required=True); inventory_parser.add_argument("--output", type=Path, required=True); inventory_parser.set_defaults(func=command_inventory)
    detect_parser = sub.add_parser("detect"); detect_parser.add_argument("--inventory", type=Path, required=True); detect_parser.add_argument("--output", type=Path, required=True); detect_parser.add_argument("--model", default="yolo11n.pt"); detect_parser.set_defaults(func=command_detect)
    prepare_parser = sub.add_parser("prepare-review"); prepare_parser.add_argument("--drafts", type=Path, required=True); prepare_parser.add_argument("--output", type=Path, required=True); prepare_parser.set_defaults(func=command_prepare)
    classify_parser = sub.add_parser("classify", help="tag annotations as normal or needs_review based on box validity"); classify_parser.add_argument("--annotations", type=Path, required=True); classify_parser.add_argument("--output", type=Path, required=True); classify_parser.set_defaults(func=command_classify)
    export_parser = sub.add_parser("export"); export_parser.add_argument("--annotations", type=Path, required=True); export_parser.add_argument("--output-root", type=Path, required=True); export_parser.add_argument("--ffmpeg", default="ffmpeg"); export_parser.set_defaults(func=command_export)
    args = parser.parse_args(); args.func(args)


if __name__ == "__main__":
    main()
