import unittest

from src.annotation_model import Event, VideoAnnotation
from src.export_confirmed import confirmed_events


class ConfirmedExportTests(unittest.TestCase):
    def test_only_confirmed_events_are_selected_and_original_indexes_are_preserved(self):
        annotation = VideoAnnotation(
            "demo",
            "/tmp/demo.mp4",
            "learning",
            640,
            480,
            100,
            30.0,
            events=[
                Event("demo-a1", 0, 10, 20, "needs_review"),
                Event("demo-a2", 0, 30, 40, "confirmed"),
                Event("demo-a3", 0, 50, 60, "excluded"),
            ],
        )

        selected = list(confirmed_events({"demo": annotation}))

        self.assertEqual([(item[1].event_id, item[2]) for item in selected], [("demo-a2", 1)])


if __name__ == "__main__":
    unittest.main()
