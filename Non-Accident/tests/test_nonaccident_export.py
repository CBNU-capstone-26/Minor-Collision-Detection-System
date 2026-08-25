import unittest

from src.annotation_model import AnnotationError, Box, VideoAnnotation
from src.exporter import nonaccident_txt


class NonAccidentExportTests(unittest.TestCase):
    def test_nonaccident_txt_contains_only_two_car_records(self):
        annotation = VideoAnnotation(
            "demo", "/tmp/demo.mp4", "normal", 640, 480, 10, 30.0,
            boxes=[Box(0, [10, 20, 100, 120], 0), Box(1, [200, 30, 300, 140], 0)],
            status="confirmed",
        )

        self.assertEqual(nonaccident_txt(annotation), "car,0,10,20,100,120\ncar,1,200,30,300,140\n")

    def test_nonaccident_rejects_noncanonical_vehicle_ids(self):
        annotation = VideoAnnotation(
            "demo", "/tmp/demo.mp4", "normal", 640, 480, 10, 30.0,
            boxes=[Box(4, [10, 20, 100, 120], 0), Box(5, [200, 30, 300, 140], 0)],
            status="confirmed",
        )

        with self.assertRaises(AnnotationError):
            nonaccident_txt(annotation)


if __name__ == "__main__":
    unittest.main()
