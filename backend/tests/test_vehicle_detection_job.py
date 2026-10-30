import importlib
import importlib.util
import unittest
from types import SimpleNamespace

from app.worker import celery_app


class VehicleDetectionJobTests(unittest.TestCase):
    def test_vehicle_detection_job_is_registered_with_celery(self):
        self.assertIn(
            "app.vehicle_job.run_vehicle_detection_task",
            celery_app.tasks,
        )

    def test_detection_rows_are_json_ready(self):
        spec = importlib.util.find_spec("app.vehicle_job")
        self.assertIsNotNone(spec)
        if spec is None:
            return

        vehicle_job = importlib.import_module("app.vehicle_job")
        rows = vehicle_job._detection_rows([
            SimpleNamespace(class_name="car", confidence=0.98765, bbox=(1, 2, 30, 40)),
        ])

        self.assertEqual(rows, [{
            "id": 0,
            "class_name": "car",
            "confidence": 0.9877,
            "bbox": [1, 2, 30, 40],
        }])
