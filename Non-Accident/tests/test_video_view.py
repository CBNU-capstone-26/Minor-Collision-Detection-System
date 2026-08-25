import os
import unittest

from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from src.video_view import VideoCanvas


class VideoCanvasZoomTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls.qt_app = QApplication.instance() or QApplication([])

    def test_reset_restores_default_zoom_and_pan(self):
        canvas = VideoCanvas()
        canvas.resize(800, 600)
        canvas.set_frame(QImage(640, 480, QImage.Format.Format_RGB32), 640, 480, [])
        canvas.zoom_in()
        canvas.pan_by(50, 25)
        self.assertGreater(canvas.zoom_factor, 1.0)
        canvas.reset_view()
        self.assertEqual(canvas.zoom_factor, 1.0)
        self.assertEqual(canvas.pan_offset.x(), 0.0)
        self.assertEqual(canvas.pan_offset.y(), 0.0)

    def test_zoom_does_not_change_box_coordinates(self):
        canvas = VideoCanvas()
        canvas.resize(800, 600)
        canvas.set_frame(QImage(640, 480, QImage.Format.Format_RGB32), 640, 480, [(0, [10, 20, 100, 120])])
        canvas.zoom_in()
        self.assertEqual(canvas.boxes, [(0, [10, 20, 100, 120])])

    def test_button_zoom_keeps_video_center_as_anchor(self):
        canvas = VideoCanvas()
        canvas.resize(800, 600)
        canvas.set_frame(QImage(640, 480, QImage.Format.Format_RGB32), 640, 480, [])
        canvas.zoom_in()
        canvas.pan_by(100, 0)
        center_before = canvas._to_widget(320, 240)
        canvas.zoom_in()
        center_after = canvas._to_widget(320, 240)
        self.assertEqual(center_before, center_after)


if __name__ == "__main__":
    unittest.main()
