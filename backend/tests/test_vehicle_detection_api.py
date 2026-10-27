import unittest

from app import api_schemas
from app.routers import videos


class VehicleDetectionApiTests(unittest.TestCase):
    def test_vehicle_detection_request_returns_a_trackable_task(self):
        task_out = getattr(api_schemas, "VehicleDetectionTaskOut", None)
        self.assertIsNotNone(task_out)
        if task_out is None:
            return

        route = next(
            item for item in videos.router.routes
            if item.path == "/api/videos/{video_id}/detect-vehicles"
            and "POST" in item.methods
        )
        self.assertIs(route.response_model, task_out)

    def test_vehicle_detection_status_has_a_dedicated_endpoint(self):
        status_out = getattr(api_schemas, "VehicleDetectionTaskStatusOut", None)
        self.assertIsNotNone(status_out)
        if status_out is None:
            return

        route = next(
            item for item in videos.router.routes
            if item.path == "/api/videos/{video_id}/vehicle-detection-tasks/{task_id}"
            and "GET" in item.methods
        )
        self.assertIs(route.response_model, status_out)
