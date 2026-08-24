from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Iterable

from .annotation_model import AnnotationError, Event, VideoAnnotation, load_annotations
from .exporter import export_event


def confirmed_events(annotations: dict[str, VideoAnnotation]) -> Iterable[tuple[VideoAnnotation, Event, int]]:
    """Yield confirmed events with their original index inside each video."""
    for annotation in annotations.values():
        for event_index, event in enumerate(annotation.events):
            if event.status == "confirmed":
                yield annotation, event, event_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Export TXT and review MP4 files for confirmed events only")
    parser.add_argument("--annotations", type=Path, required=True, help="annotation JSON file")
    parser.add_argument("--output-root", type=Path, required=True, help="output directory")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg executable name or path")
    args = parser.parse_args()

    annotations = load_annotations(args.annotations)
    selected = list(confirmed_events(annotations))
    if not selected:
        print("confirmed event 없음: 생성할 TXT/MP4가 없습니다.")
        return

    exported = 0
    failures: list[str] = []
    for annotation, event, event_index in selected:
        try:
            txt_path, video_path = export_event(annotation, event, args.output_root, event_index, args.ffmpeg)
        except (AnnotationError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
            failures.append(f"{event.event_id}: {exc}")
            continue
        exported += 1
        print(f"exported {event.event_id}: {txt_path} / {video_path}")

    print(f"완료: {exported}/{len(selected)} confirmed event(s)")
    if failures:
        print("실패한 이벤트:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
