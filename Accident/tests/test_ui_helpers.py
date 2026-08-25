import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.annotation_model import Event, VideoAnnotation
from src.ui_helpers import clamp_box, natural_video_key, resize_box, sort_video_paths


class UiHelperTests(unittest.TestCase):
    def test_video_numbers_sort_numerically_not_lexicographically(self):
        names = ["100 - last.mp4", "2 - second.mp4", "11 - eleventh.mp4"]
        self.assertEqual(sorted(names, key=natural_video_key), ["2 - second.mp4", "11 - eleventh.mp4", "100 - last.mp4"])

    def test_edge_handle_changes_only_one_dimension(self):
        self.assertEqual(resize_box([100, 100, 300, 200], "e", 40, 20, 1000, 800), [100, 100, 340, 200])

    def test_clamp_box_stays_inside_video(self):
        self.assertEqual(clamp_box([-50, 20, 700, 500], 640, 480), [0, 20, 640, 480])

    def test_corner_handle_preserves_original_aspect_ratio(self):
        result = resize_box([100, 100, 300, 200], "se", 40, 60, 1000, 800)
        self.assertEqual(result, [100, 100, 420, 260])

    def test_status_sort_uses_saved_video_status(self):
        paths = [Path("003.mp4"), Path("001.mp4"), Path("002.mp4")]
        annotations = {
            "001": VideoAnnotation("001", "001.mp4", "learning", 10, 10, 1, 30, status="confirmed"),
            "002": VideoAnnotation("002", "002.mp4", "learning", 10, 10, 1, 30, status="in_progress"),
            "003": VideoAnnotation("003", "003.mp4", "learning", 10, 10, 1, 30, status="not_started"),
        }
        self.assertEqual([path.stem for path in sort_video_paths(paths, "status", annotations)], ["003", "002", "001"])


if __name__ == "__main__":
    unittest.main()
