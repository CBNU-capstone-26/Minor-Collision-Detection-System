import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.annotation_model import Box, Event, VideoAnnotation, event_output_stem, load_annotations, paper_txt, save_annotations, validate_annotation
from src.nonaccident_pipeline import classify_annotation_status


class AnnotationModelTests(unittest.TestCase):
    def make_annotation(self) -> VideoAnnotation:
        return VideoAnnotation(
            video_id="demo",
            source_video="demo.mp4",
            split="learning",
            width=1920,
            height=1080,
            frame_count=90,
            fps=30.0,
            boxes=[Box(0, [10, 20, 100, 120], 4), Box(1, [200, 30, 300, 140], 4)],
            events=[Event("demo-a1", 0, 73, 89, "confirmed")],
        )

    def test_paper_txt_preserves_static_boxes_and_event_id(self):
        annotation = self.make_annotation()
        self.assertEqual(
            paper_txt(annotation, annotation.events[0]),
            "car,0,10,20,100,120\ncar,1,200,30,300,140\nA,0,73,89\n",
        )

    def test_invalid_event_vehicle_is_rejected(self):
        annotation = self.make_annotation()
        annotation.events[0].vehicle_id = 9
        self.assertIn("event demo-a1 references missing vehicle 9", validate_annotation(annotation))

    def test_json_round_trip(self):
        annotation = self.make_annotation()
        annotation.status = "in_progress"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "accident_annotations.json"
            save_annotations(path, {annotation.video_id: annotation})
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[2], '    "videos": [')
            restored = load_annotations(path)["demo"]
            self.assertEqual(restored.to_dict(), annotation.to_dict())

    def test_multiple_events_get_distinct_output_names(self):
        self.assertEqual(event_output_stem("demo", 0), "demo")
        self.assertEqual(event_output_stem("demo", 1), "demo__a2")

    def test_two_valid_boxes_are_status_normal(self):
        annotation = self.make_annotation()
        annotation.events = []
        self.assertEqual(classify_annotation_status(annotation), "normal")

    def test_empty_boxes_are_status_needs_review(self):
        annotation = self.make_annotation()
        annotation.boxes = []
        annotation.events = []
        self.assertEqual(classify_annotation_status(annotation), "needs_review")

    def test_serialized_annotation_has_no_tag_field(self):
        annotation = self.make_annotation()
        self.assertNotIn("tag", annotation.to_dict())


if __name__ == "__main__":
    unittest.main()
