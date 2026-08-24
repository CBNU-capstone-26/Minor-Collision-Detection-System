import json
import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication

from src.app import AnnotationWindow


class AppStartupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls.qt_app = QApplication.instance() or QApplication([])

    def test_starts_empty_and_loads_annotations_when_folder_is_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "videos"
            source.mkdir()
            video = source / "001.mp4"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (16, 16))
            writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
            writer.release()
            annotations = root / "accident_annotations.json"
            annotations.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "videos": [
                            {
                                "video_id": "001",
                                "source_video": str(video),
                                "split": "normal",
                                "width": 640,
                                "height": 480,
                                "frame_count": 1,
                                "fps": 30,
                                "status": "in_progress",
                                "boxes": [],
                                "events": [],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            window = AnnotationWindow(source, annotations, root / "output")
            self.assertEqual(window.videos, [])
            self.assertEqual(window.annotations, {})

            window.load_folder(source)
            self.assertEqual(window.videos, [video])
            self.assertIn("001", window.annotations)

            new_annotations = root / "new-annotations.json"
            window.create_annotation_file(new_annotations)
            self.assertEqual(window.annotation_path, new_annotations)
            self.assertEqual(window.annotations, {})
            self.assertTrue(new_annotations.is_file())

            window.load_annotation_file(annotations)
            self.assertEqual(window.annotation_path, annotations)
            self.assertIn("001", window.annotations)

            window.mode = "non-accident"
            window.current.status = "normal"
            window.set_status_combo("normal")
            window.save_current()
            self.assertEqual(window.current.status, "normal")
            self.assertEqual(window.load_annotation_file(annotations), None)
            window.close()


if __name__ == "__main__":
    unittest.main()
