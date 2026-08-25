from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from .annotation_model import AnnotationError, Event, VideoAnnotation, paper_txt


def export_event(annotation: VideoAnnotation, event: Event, output_root: str | Path, event_index: int, ffmpeg: str = "ffmpeg") -> tuple[Path, Path]:
    """Write paper TXT and a full-length H.264 review video for one event."""
    errors = paper_txt(annotation, event)  # validates and returns the text
    source = Path(annotation.source_video).expanduser()
    if not source.is_file():
        raise AnnotationError(f"source video does not exist: {source}")
    root = Path(output_root) / annotation.split
    txt_dir = root / "txt"
    video_dir = root / "visualized"
    txt_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    stem = annotation.video_id if event_index == 0 else f"{annotation.video_id}__a{event_index + 1}"
    txt_path = txt_dir / f"{stem}.txt"
    video_path = video_dir / f"{stem}.mp4"
    txt_path.write_text(errors, encoding="utf-8")

    try:
        import cv2
    except ImportError as exc:
        raise AnnotationError("opencv-python is required to render review videos") from exc
    if shutil.which(ffmpeg) is None:
        raise AnnotationError(f"ffmpeg was not found: {ffmpeg}")

    with tempfile.TemporaryDirectory(prefix="paper-review-") as directory:
        intermediate = Path(directory) / "intermediate.mp4"
        capture = cv2.VideoCapture(str(source))
        writer = cv2.VideoWriter(str(intermediate), cv2.VideoWriter_fourcc(*"mp4v"), annotation.fps, (annotation.width, annotation.height))
        if not capture.isOpened() or not writer.isOpened():
            capture.release()
            writer.release()
            raise AnnotationError(f"could not open video for rendering: {source}")
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            for box in annotation.boxes:
                x1, y1, x2, y2 = box.bbox
                color = (75, 184, 242) if box.vehicle_id == event.vehicle_id else (208, 213, 104)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, f"car {box.vehicle_id}", (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
            frame_label = f"frame {frame_index:04d}"
            label_origin = (36, 52)
            (text_width, text_height), baseline = cv2.getTextSize(frame_label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            padding = 3
            cv2.rectangle(
                frame,
                (label_origin[0] - padding, label_origin[1] - text_height - padding),
                (label_origin[0] + text_width + padding, label_origin[1] + baseline + padding),
                (8, 11, 13),
                -1,
            )
            cv2.putText(frame, frame_label, label_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (244, 240, 232), 2, cv2.LINE_AA)
            if event.start_frame <= frame_index <= event.end_frame:
                cv2.putText(frame, f"ACCIDENT {event.start_frame}-{event.end_frame}", (36, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (75, 184, 242), 2, cv2.LINE_AA)
            writer.write(frame)
            frame_index += 1
        capture.release()
        writer.release()
        if frame_index != annotation.frame_count:
            raise AnnotationError(f"rendered {frame_index} frames, expected {annotation.frame_count}")
        command = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(intermediate), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(annotation.fps), str(video_path)]
        subprocess.run(command, check=True)
    return txt_path, video_path
