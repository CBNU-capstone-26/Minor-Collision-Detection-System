from __future__ import annotations

from dataclasses import dataclass
from math import hypot


@dataclass(frozen=True)
class Observation:
    track_id: int
    frame: int
    bbox: list[int]
    confidence: float
    class_id: int = 2

    @property
    def center(self) -> tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)


def _movement(track: list[Observation]) -> float:
    if len(track) < 2:
        return 0.0
    first, last = track[0].center, track[-1].center
    return hypot(last[0] - first[0], last[1] - first[1])


def _nearest_distance(left: list[Observation], right: list[Observation]) -> float:
    return min(hypot(a.center[0] - b.center[0], a.center[1] - b.center[1]) for a in left for b in right)


def _area(bbox: list[int]) -> float:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def _overlap_ratio(left: list[int], right: list[int]) -> float:
    intersection = [max(left[0], right[0]), max(left[1], right[1]), min(left[2], right[2]), min(left[3], right[3])]
    right_area = _area(right)
    return _area(intersection) / right_area if right_area else 1.0


def choose_reference_tracks(observations: list[Observation], frame_count: int) -> tuple[int | None, list[int]]:
    """Choose the moving vehicle and two nearby low-motion reference vehicles."""
    grouped: dict[int, list[Observation]] = {}
    for observation in observations:
        grouped.setdefault(observation.track_id, []).append(observation)
    for track in grouped.values():
        track.sort(key=lambda item: item.frame)
    if not grouped:
        return None, []

    moving_id = max(grouped, key=lambda track_id: (_movement(grouped[track_id]), len(grouped[track_id])))
    moving = grouped[moving_id]
    moving_distance = _movement(moving)
    stationary_limit = max(5.0, moving_distance * 0.25)
    candidates = []
    min_observations = max(2, min(15, round(frame_count * 0.02)))
    for track_id, track in grouped.items():
        if track_id == moving_id:
            continue
        if len(track) < min_observations:
            continue
        motion = _movement(track)
        if motion > stationary_limit:
            continue
        distance = _nearest_distance(track, moving)
        candidates.append((distance, -len(track), track_id))
    candidates.sort()
    references = [track_id for _, _, track_id in candidates[:2]]
    return moving_id, references if len(references) == 2 else []
