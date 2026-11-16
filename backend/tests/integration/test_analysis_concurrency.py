"""Concurrent requests for one video should create only one active task."""
import os
import sys
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch
from urllib.parse import urlparse

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from app import db_models
from app.db_connection import Base
from app.routers import analysis
from app.routers import videos


@unittest.skipUnless(os.environ.get('SIOT_TEST_POSTGRES_URL'), 'isolated PostgreSQL required')
class ConcurrentAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        url = os.environ['SIOT_TEST_POSTGRES_URL']
        parsed = urlparse(url.replace('postgresql+psycopg://', 'postgresql://', 1))
        if parsed.hostname not in ('127.0.0.1', 'localhost') or not parsed.path.lstrip('/').startswith('siot_test_'):
            raise RuntimeError('refusing to use a non-isolated database')
        cls.engine = create_engine(url)
        Base.metadata.create_all(cls.engine)
        cls.sessions = sessionmaker(bind=cls.engine, expire_on_commit=False)

    @classmethod
    def tearDownClass(cls):
        cls.engine.dispose()

    def test_simultaneous_requests_enqueue_only_one_task(self):
        with self.sessions.begin() as db:
            user = db_models.User(username='concurrency-test', password_hash='test')
            db.add(user)
            db.flush()
            video = db_models.Video(user_id=user.id, video_name='test.mp4', video_path='uploads/test.mp4',
                                    camera_location='test', width=10, height=10, fps=10, total_frames=10)
            db.add(video)
            db.flush()
            video_id = video.id
        first_check = threading.Event()
        second_check = threading.Event()
        lock = threading.Lock()
        counter = [0]
        original = analysis._active_task_for_video

        def synchronize_after_check(video_id, db):
            result = original(video_id, db)
            with lock:
                counter[0] += 1
                number = counter[0]
            if number == 1:
                first_check.set()
                second_check.wait(timeout=1)
            elif number == 2:
                second_check.set()
            return result

        def request():
            with self.sessions() as db:
                # The real endpoint loads the video before entering _enqueue.
                db.get(db_models.Video, video_id)
                return analysis._enqueue(video_id, {'bbox_xmin': 1, 'bbox_ymin': 1,
                                                  'bbox_xmax': 5, 'bbox_ymax': 5}, db)

        with patch.object(analysis, '_active_task_for_video', side_effect=synchronize_after_check), \
             patch('app.prediction_job.run_prediction_task.delay', side_effect=lambda task_id: SimpleNamespace(id=f'fake-job-{task_id}')) as delay, \
             ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(request)
            self.assertTrue(first_check.wait(timeout=5))
            second = executor.submit(request)
            responses = [first.result(timeout=10), second.result(timeout=10)]
        self.assertEqual(responses[0].task_id, responses[1].task_id)
        self.assertEqual(delay.call_count, 1)
        with self.sessions() as db:
            self.assertEqual(db.query(db_models.AnalysisTask).filter_by(video_id=video_id).count(), 1)

    def test_simultaneous_vehicle_detection_requests_enqueue_only_one_task(self):
        with self.sessions.begin() as db:
            user = db_models.User(username='detection-concurrency-test', password_hash='test')
            db.add(user)
            db.flush()
            video = db_models.Video(user_id=user.id, video_name='test.mp4', video_path='uploads/test.mp4',
                                    camera_location='test', width=10, height=10, fps=10, total_frames=10)
            db.add(video)
            db.flush()
            video_id = video.id
        first_check, second_check = threading.Event(), threading.Event()
        lock, counter = threading.Lock(), [0]

        def synchronize_after_query(conn, cursor, statement, parameters, context, executemany):
            if 'FROM vehicle_detection_tasks' not in statement or not statement.lstrip().upper().startswith('SELECT'):
                return
            with lock:
                counter[0] += 1
                number = counter[0]
            if number == 1:
                first_check.set()
                second_check.wait(timeout=1)
            elif number == 2:
                second_check.set()

        def request():
            with self.sessions() as db:
                return videos._enqueue_vehicle_detection(db.get(db_models.Video, video_id), 0, db)

        event.listen(self.engine, 'after_cursor_execute', synchronize_after_query)
        try:
            with patch('app.vehicle_job.run_vehicle_detection_task.delay',
                       side_effect=lambda task_id: SimpleNamespace(id=f'detect-{task_id}')) as delay, \
                 ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(request)
                self.assertTrue(first_check.wait(timeout=5))
                second = executor.submit(request)
                responses = [first.result(timeout=10), second.result(timeout=10)]
            self.assertEqual(responses[0].task_id, responses[1].task_id)
            self.assertEqual(delay.call_count, 1)
        finally:
            event.remove(self.engine, 'after_cursor_execute', synchronize_after_query)

    def test_cancel_during_clip_upload_is_not_overwritten_by_success(self):
        from app import prediction_job

        with self.sessions.begin() as db:
            user = db_models.User(username='cancel-upload-test', password_hash='test')
            db.add(user)
            db.flush()
            video = db_models.Video(user_id=user.id, video_name='test.mp4', video_path='uploads/test.mp4',
                                    camera_location='test', width=10, height=10, fps=10, total_frames=10)
            db.add(video)
            db.flush()
            task = db_models.AnalysisTask(video_id=video.id, status='PENDING', bbox_xmin=1,
                                          bbox_ymin=1, bbox_xmax=5, bbox_ymax=5)
            db.add(task)
            db.flush()
            task_id = task.id
        inference = ModuleType('inference')
        inference.predict_events_and_clips = lambda *args, **kwargs: [{
            'clip_path': '/tmp/siot-cancel-test.mp4', 'start_sec': 0, 'start_frame': 0,
            'end_sec': 1, 'end_frame': 1, 'crash_prob': 0.5,
        }]

        def cancel_while_uploading(*args, **kwargs):
            with self.sessions.begin() as db:
                db.get(db_models.AnalysisTask, task_id).status = 'CANCELLED'

        storage = SimpleNamespace(upload_path=cancel_while_uploading, delete=Mock())
        with patch.object(prediction_job, 'SessionLocal', self.sessions), \
             patch.object(prediction_job, '_get_model', return_value=object()), \
             patch.object(prediction_job, 'get_storage', return_value=storage), \
             patch.object(prediction_job, 'materialize', return_value=nullcontext(Path('/tmp/test.mp4'))), \
             patch.object(prediction_job.run_prediction_task, 'update_state'), \
             patch.dict(sys.modules, {'inference': inference}):
            prediction_job.run_prediction_task.run(task_id)
        with self.sessions() as db:
            self.assertEqual(db.get(db_models.AnalysisTask, task_id).status, 'CANCELLED')
            self.assertEqual(db.query(db_models.CrashEvent).filter_by(task_id=task_id).count(), 0)
        storage.delete.assert_called_once_with('clips/siot-cancel-test.mp4')

    def test_failure_after_cancel_does_not_overwrite_cancelled_status(self):
        from app import prediction_job

        with self.sessions.begin() as db:
            user = db_models.User(username='cancel-error-test', password_hash='test')
            db.add(user)
            db.flush()
            video = db_models.Video(user_id=user.id, video_name='test.mp4', video_path='uploads/test.mp4',
                                    camera_location='test', width=10, height=10, fps=10, total_frames=10)
            db.add(video)
            db.flush()
            task = db_models.AnalysisTask(video_id=video.id, status='PENDING', bbox_xmin=1,
                                          bbox_ymin=1, bbox_xmax=5, bbox_ymax=5)
            db.add(task)
            db.flush()
            task_id = task.id

        def cancel_then_fail():
            with self.sessions.begin() as db:
                db.get(db_models.AnalysisTask, task_id).status = 'CANCELLED'
            raise RuntimeError('test error after cancellation')

        with patch.object(prediction_job, 'SessionLocal', self.sessions), \
             patch.object(prediction_job, '_get_model', side_effect=cancel_then_fail), \
             patch.object(prediction_job.run_prediction_task, 'update_state'):
            with self.assertRaises(RuntimeError):
                prediction_job.run_prediction_task.run(task_id)
        with self.sessions() as db:
            self.assertEqual(db.get(db_models.AnalysisTask, task_id).status, 'CANCELLED')

    def test_cancel_after_completion_does_not_overwrite_success(self):
        with self.sessions.begin() as db:
            user = db_models.User(username='cancel-after-complete-test', password_hash='test')
            db.add(user)
            db.flush()
            user_id = user.id
            video = db_models.Video(user_id=user.id, video_name='test.mp4', video_path='uploads/test.mp4',
                                    camera_location='test', width=10, height=10, fps=10, total_frames=10)
            db.add(video)
            db.flush()
            task = db_models.AnalysisTask(video_id=video.id, status='PROCESSING', bbox_xmin=1,
                                          bbox_ymin=1, bbox_xmax=5, bbox_ymax=5)
            db.add(task)
            db.flush()
            task_id = task.id
        get_task = analysis._get_owned_task

        def complete_after_read(task_id, db, user):
            task = get_task(task_id, db, user)
            with self.sessions.begin() as other:
                other.get(db_models.AnalysisTask, task_id).status = 'SUCCESS'
            return task

        with self.sessions() as db, patch.object(analysis, '_get_owned_task', side_effect=complete_after_read):
            result = analysis.cancel_task(task_id, db, SimpleNamespace(id=user_id))
            self.assertEqual(result.status, 'SUCCESS')
        with self.sessions() as db:
            self.assertEqual(db.get(db_models.AnalysisTask, task_id).status, 'SUCCESS')

    def test_cancel_before_worker_start_does_not_restart_task(self):
        from app import prediction_job

        with self.sessions.begin() as db:
            user = db_models.User(username='cancel-before-worker-test', password_hash='test')
            db.add(user)
            db.flush()
            video = db_models.Video(user_id=user.id, video_name='test.mp4', video_path='uploads/test.mp4',
                                    camera_location='test', width=10, height=10, fps=10, total_frames=10)
            db.add(video)
            db.flush()
            task = db_models.AnalysisTask(video_id=video.id, status='PENDING', bbox_xmin=1,
                                          bbox_ymin=1, bbox_xmax=5, bbox_ymax=5)
            db.add(task)
            db.flush()
            task_id = task.id
        session = self.sessions()
        original_get = session.get

        def get_then_cancel(model, identifier, *args, **kwargs):
            task = original_get(model, identifier, *args, **kwargs)
            if model is db_models.AnalysisTask:
                with self.sessions.begin() as other:
                    other.get(db_models.AnalysisTask, identifier).status = 'CANCELLED'
            return task

        with patch.object(session, 'get', side_effect=get_then_cancel), \
             patch.object(prediction_job, 'SessionLocal', return_value=session), \
             patch.object(prediction_job, '_get_model', side_effect=RuntimeError('worker must not start')) as get_model, \
             patch.object(prediction_job.run_prediction_task, 'update_state'):
            result = prediction_job.run_prediction_task.run(task_id)
        self.assertEqual(result['skipped'], 'CANCELLED')
        get_model.assert_not_called()

    def test_successful_worker_stores_event_and_success_status(self):
        from app import prediction_job

        with self.sessions.begin() as db:
            user = db_models.User(username='worker-success-test', password_hash='test')
            db.add(user)
            db.flush()
            video = db_models.Video(user_id=user.id, video_name='test.mp4', video_path='uploads/test.mp4',
                                    camera_location='test', width=10, height=10, fps=10, total_frames=10)
            db.add(video)
            db.flush()
            task = db_models.AnalysisTask(video_id=video.id, status='PENDING', bbox_xmin=1,
                                          bbox_ymin=1, bbox_xmax=5, bbox_ymax=5)
            db.add(task)
            db.flush()
            task_id = task.id
        inference = ModuleType('inference')
        inference.predict_events_and_clips = lambda *args, **kwargs: [{
            'clip_path': '/tmp/siot-success-test.mp4', 'start_sec': 0, 'start_frame': 0,
            'end_sec': 1, 'end_frame': 1, 'crash_prob': 0.5,
        }]
        storage = SimpleNamespace(upload_path=lambda *args, **kwargs: None, delete=lambda key: None)
        with patch.object(prediction_job, 'SessionLocal', self.sessions), \
             patch.object(prediction_job, '_get_model', return_value=object()), \
             patch.object(prediction_job, 'get_storage', return_value=storage), \
             patch.object(prediction_job, 'materialize', return_value=nullcontext(Path('/tmp/test.mp4'))), \
             patch.object(prediction_job.run_prediction_task, 'update_state'), \
             patch.dict(sys.modules, {'inference': inference}):
            result = prediction_job.run_prediction_task.run(task_id)
        self.assertEqual(result['events'], 1)
        with self.sessions() as db:
            task = db.get(db_models.AnalysisTask, task_id)
            self.assertEqual(task.status, 'SUCCESS')
            self.assertEqual(task.event_count, 1)
            self.assertEqual(db.query(db_models.CrashEvent).filter_by(task_id=task_id).count(), 1)

    def test_cancelled_clip_cleanup_is_retried_after_storage_failure(self):
        from app import prediction_job

        with self.sessions.begin() as db:
            user = db_models.User(username='cancel-cleanup-retry-test', password_hash='test')
            db.add(user)
            db.flush()
            video = db_models.Video(user_id=user.id, video_name='test.mp4', video_path='uploads/test.mp4',
                                    camera_location='test', width=10, height=10, fps=10, total_frames=10)
            db.add(video)
            db.flush()
            task = db_models.AnalysisTask(video_id=video.id, status='PENDING', bbox_xmin=1,
                                          bbox_ymin=1, bbox_xmax=5, bbox_ymax=5)
            db.add(task)
            db.flush()
            task_id = task.id
        inference = ModuleType('inference')
        inference.predict_events_and_clips = lambda *args, **kwargs: [{
            'clip_path': '/tmp/siot-cancel-retry-test.mp4', 'start_sec': 0, 'start_frame': 0,
            'end_sec': 1, 'end_frame': 1, 'crash_prob': 0.5,
        }]

        def cancel_while_uploading(*args, **kwargs):
            with self.sessions.begin() as db:
                db.get(db_models.AnalysisTask, task_id).status = 'CANCELLED'

        storage = SimpleNamespace(upload_path=cancel_while_uploading, delete=Mock(side_effect=OSError('down')))
        with patch.object(prediction_job, 'SessionLocal', self.sessions), \
             patch.object(prediction_job, '_get_model', return_value=object()), \
             patch.object(prediction_job, 'get_storage', return_value=storage), \
             patch.object(prediction_job, 'materialize', return_value=nullcontext(Path('/tmp/test.mp4'))), \
             patch.object(prediction_job.run_prediction_task, 'update_state'), \
             patch.dict(sys.modules, {'inference': inference}):
            prediction_job.run_prediction_task.run(task_id)
        with self.sessions() as db:
            self.assertEqual(db.get(db_models.AnalysisTask, task_id).status, 'CANCELLED')
            self.assertEqual(db.query(db_models.FileCleanup).filter_by(
                object_key='clips/siot-cancel-retry-test.mp4').count(), 1)

    def test_failed_raw_clip_upload_keeps_cleanup_retry_for_uploaded_clip(self):
        from app import prediction_job

        with self.sessions.begin() as db:
            user = db_models.User(username='upload-cleanup-retry-test', password_hash='test')
            db.add(user)
            db.flush()
            video = db_models.Video(user_id=user.id, video_name='test.mp4', video_path='uploads/test.mp4',
                                    camera_location='test', width=10, height=10, fps=10, total_frames=10)
            db.add(video)
            db.flush()
            task = db_models.AnalysisTask(video_id=video.id, status='PENDING', bbox_xmin=1,
                                          bbox_ymin=1, bbox_xmax=5, bbox_ymax=5)
            db.add(task)
            db.flush()
            task_id = task.id
        inference = ModuleType('inference')
        inference.predict_events_and_clips = lambda *args, **kwargs: [{
            'clip_path': '/tmp/siot-upload-retry.mp4', 'raw_clip_path': '/tmp/siot-upload-retry_raw.mp4',
            'start_sec': 0, 'start_frame': 0, 'end_sec': 1, 'end_frame': 1, 'crash_prob': 0.5,
        }]

        def fail_raw_upload(source, key, **kwargs):
            if key.endswith('_raw.mp4'):
                raise OSError('raw upload failed')

        storage = SimpleNamespace(upload_path=fail_raw_upload, delete=Mock(side_effect=OSError('down')))
        with patch.object(prediction_job, 'SessionLocal', self.sessions), \
             patch.object(prediction_job, '_get_model', return_value=object()), \
             patch.object(prediction_job, 'get_storage', return_value=storage), \
             patch.object(prediction_job, 'materialize', return_value=nullcontext(Path('/tmp/test.mp4'))), \
             patch.object(prediction_job.run_prediction_task, 'update_state'), \
             patch.dict(sys.modules, {'inference': inference}):
            with self.assertRaises(OSError):
                prediction_job.run_prediction_task.run(task_id)
        with self.sessions() as db:
            self.assertEqual(db.get(db_models.AnalysisTask, task_id).status, 'FAILURE')
            self.assertEqual(db.query(db_models.FileCleanup).filter_by(
                object_key='clips/siot-upload-retry.mp4').count(), 1)
