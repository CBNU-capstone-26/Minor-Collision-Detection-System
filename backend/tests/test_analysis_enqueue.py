"""Queue failures must not leave an analysis task permanently pending."""

import unittest
from unittest.mock import MagicMock, patch

from fastapi import HTTPException

from app.routers.analysis import _enqueue


class AnalysisEnqueueTests(unittest.TestCase):
    def test_queue_failure_marks_task_failed_and_returns_service_unavailable(self):
        db = MagicMock()
        db.query.return_value.filter.return_value.order_by.return_value.with_for_update.return_value.first.return_value = None
        db.refresh.side_effect = lambda task: setattr(task, "id", 42)

        with patch(
            "app.prediction_job.run_prediction_task.delay",
            side_effect=RuntimeError("Redis is unavailable"),
        ):
            with self.assertRaises(HTTPException) as raised:
                _enqueue(
                    video_id=7,
                    bbox={"bbox_xmin": 1, "bbox_ymin": 2, "bbox_xmax": 30, "bbox_ymax": 40},
                    db=db,
                )

        task = db.add.call_args.args[0]
        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(task.status, "FAILURE")
        self.assertIn("Redis is unavailable", task.error_message)
        self.assertIsNotNone(task.finished_at)
        self.assertEqual(db.commit.call_count, 2)


if __name__ == "__main__":
    unittest.main()
