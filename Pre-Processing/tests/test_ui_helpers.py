import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ui_helpers import natural_video_key, resize_box


class UiHelperTests(unittest.TestCase):
    def test_video_numbers_sort_numerically_not_lexicographically(self):
        names = ["100 - last.mp4", "2 - second.mp4", "11 - eleventh.mp4"]
        self.assertEqual(sorted(names, key=natural_video_key), ["2 - second.mp4", "11 - eleventh.mp4", "100 - last.mp4"])

    def test_edge_handle_changes_only_one_dimension(self):
        self.assertEqual(resize_box([100, 100, 300, 200], "e", 40, 20, 1000, 800), [100, 100, 340, 200])

    def test_corner_handle_preserves_original_aspect_ratio(self):
        result = resize_box([100, 100, 300, 200], "se", 40, 60, 1000, 800)
        self.assertEqual(result, [100, 100, 420, 260])


if __name__ == "__main__":
    unittest.main()
