import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.routers.analysis import _progress
from app.routers.videos import _vehicle_detection_progress


class TaskProgressTests(unittest.TestCase):
    def test_redis_failure_does_not_hide_saved_analysis_status(self):
        task = SimpleNamespace(status='PROCESSING', celery_task_id='job')
        with patch('app.worker.celery_app.AsyncResult', side_effect=ConnectionError('unavailable')):
            self.assertIsNone(_progress(task)['progress'])

    def test_redis_failure_does_not_hide_saved_detection_status(self):
        task = SimpleNamespace(status='PROCESSING', celery_task_id='job')
        with patch('app.worker.celery_app.AsyncResult', side_effect=ConnectionError('unavailable')):
            self.assertIsNone(_vehicle_detection_progress(task)['progress'])
