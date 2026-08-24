from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Mapping


def natural_video_key(path_or_name: str | Path) -> tuple[object, ...]:
    """Sort names such as 2.mp4, 11.mp4, 100.mp4 in numeric order."""
    name = Path(path_or_name).name.lower()
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name))


def clamp_box(box: list[int] | list[float], video_width: int, video_height: int, min_size: int = 4) -> list[int]:
    """Normalize an xyxy box so every edge stays inside the video."""
    if video_width <= 0 or video_height <= 0:
        raise ValueError("video dimensions must be positive")
    minimum = min(min_size, video_width, video_height)
    left, right = sorted((round(box[0]), round(box[2])))
    top, bottom = sorted((round(box[1]), round(box[3])))
    left = max(0, min(video_width - minimum, left))
    right = max(left + minimum, min(video_width, right))
    if right > video_width:
        right = video_width
        left = max(0, right - minimum)
    top = max(0, min(video_height - minimum, top))
    bottom = max(top + minimum, min(video_height, bottom))
    if bottom > video_height:
        bottom = video_height
        top = max(0, bottom - minimum)
    return [left, top, right, bottom]


def resize_box(box: list[int], handle: str, dx: float, dy: float, video_width: int, video_height: int) -> list[int]:
    """Resize an xyxy box from an edge or corner handle.

    Edge handles change one dimension. Corner handles keep the original aspect ratio.
    The opposite corner remains fixed and the result stays inside the video.
    """
    x1, y1, x2, y2 = box
    min_size = 4
    if handle in {"n", "s"}:
        if handle == "n":
            y1 = round(max(0, min(y2 - min_size, y1 + dy)))
        else:
            y2 = round(min(video_height, max(y1 + min_size, y2 + dy)))
    elif handle in {"w", "e"}:
        if handle == "w":
            x1 = round(max(0, min(x2 - min_size, x1 + dx)))
        else:
            x2 = round(min(video_width, max(x1 + min_size, x2 + dx)))
    elif handle in {"nw", "ne", "se", "sw"}:
        original_width = max(min_size, x2 - x1)
        original_height = max(min_size, y2 - y1)
        ratio = original_width / original_height
        anchor_x = x2 if "w" in handle else x1
        anchor_y = y2 if "n" in handle else y1
        pointer_x = x1 + dx if "w" in handle else x2 + dx
        pointer_y = y1 + dy if "n" in handle else y2 + dy
        width = max(min_size, abs(pointer_x - anchor_x))
        height = max(min_size, abs(pointer_y - anchor_y))
        if width / height > ratio:
            height = width / ratio
        else:
            width = height * ratio
        max_width = anchor_x if "w" in handle else video_width - anchor_x
        max_height = anchor_y if "n" in handle else video_height - anchor_y
        width = min(width, max_width, max_height * ratio)
        height = width / ratio
        width, height = round(max(min_size, width)), round(max(min_size, height))
        if "w" in handle:
            x1 = round(anchor_x - width)
        else:
            x2 = round(anchor_x + width)
        if "n" in handle:
            y1 = round(anchor_y - height)
        else:
            y2 = round(anchor_y + height)
    return clamp_box([x1, y1, x2, y2], video_width, video_height, min_size)


def sort_video_paths(paths: Iterable[Path], mode: str, annotations: Mapping[str, object]) -> list[Path]:
    items = list(paths)
    if mode == "status":
        def status(path: Path) -> int:
            annotation = annotations.get(path.stem)
            if annotation is None:
                return 0
            return {"not_started": 0, "in_progress": 1, "confirmed": 2, "excluded": 3}.get(getattr(annotation, "status", "not_started"), 0)
        return sorted(items, key=lambda path: (status(path), natural_video_key(path)))
    if mode == "name":
        return sorted(items, key=lambda path: path.name.lower())
    return sorted(items, key=natural_video_key)
