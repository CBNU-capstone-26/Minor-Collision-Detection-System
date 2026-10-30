import unittest

from app import db_models


class VehicleDetectionTaskModelTests(unittest.TestCase):
    def test_vehicle_detection_task_persists_queue_tracking_fields(self):
        task_model = getattr(db_models, "VehicleDetectionTask", None)

        self.assertIsNotNone(task_model)
        self.assertTrue({
            "id", "video_id", "celery_task_id", "timestamp_sec", "status",
            "error_message", "result_json", "created_at", "finished_at",
        }.issubset(task_model.__table__.columns.keys()))
