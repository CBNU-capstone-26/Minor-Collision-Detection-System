import unittest

from app import worker


class GpuQueueRoutingTests(unittest.TestCase):
    def test_prediction_task_can_be_routed_to_gpu_queue(self):
        build_routes = getattr(worker, "build_task_routes", None)

        self.assertTrue(callable(build_routes))
        self.assertEqual(
            build_routes("gpu"),
            {
                "app.prediction_job.run_prediction_task": {"queue": "gpu"},
                "app.vehicle_job.run_vehicle_detection_task": {"queue": "gpu"},
            },
        )
