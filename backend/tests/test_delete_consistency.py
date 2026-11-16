"""Deleting database rows must not destroy files before the transaction commits."""
import tempfile
import unittest
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import urlparse

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app import db_models
from app.db_connection import Base
from app.object_storage import LocalObjectStorage
from app.routers import videos


class DeleteConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        url = os.environ.get('SIOT_TEST_DELETE_DATABASE_URL')
        if url:
            parsed = urlparse(url)
            if parsed.hostname not in ('127.0.0.1', 'localhost', '::1') or not parsed.path.lstrip('/').startswith('siot_test_'):
                raise RuntimeError('Deletion test requires a disposable local siot_test_ database')
        else:
            url = 'sqlite:///' + str(Path(self.temp.name) / 'test.sqlite')
        self.engine = create_engine(url)
        self.addCleanup(self.engine.dispose)
        if os.environ.get('SIOT_TEST_DELETE_DATABASE_URL'):
            Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)
        self.sessions = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.storage = LocalObjectStorage(Path(self.temp.name) / 'media')
        path = self.storage.path_for('uploads/original.mp4')
        path.parent.mkdir(parents=True)
        path.write_bytes(b'original video')
        with self.sessions.begin() as db:
            user = db_models.User(username='test', password_hash='test')
            db.add(user)
            db.flush()
            video = db_models.Video(user_id=user.id, video_name='original.mp4',
                                    video_path='uploads/original.mp4', camera_location='test',
                                    width=10, height=10, fps=10, total_frames=10)
            db.add(video)
            db.flush()
            self.user_id, self.video_id = user.id, video.id

    def test_failed_commit_keeps_original_video_file(self):
        with self.sessions() as db, patch('app.file_cleanup.get_storage', return_value=self.storage), \
             patch.object(db, 'commit', side_effect=RuntimeError('database failure')):
            with self.assertRaises(RuntimeError):
                videos.delete_video(self.video_id, db, SimpleNamespace(id=self.user_id))
            db.rollback()
        self.assertTrue(self.storage.exists('uploads/original.mp4'))
        with self.sessions() as db:
            self.assertIsNotNone(db.get(db_models.Video, self.video_id))

    def test_storage_failure_is_retried_after_database_commit(self):
        from app.file_cleanup import process_pending_cleanup
        with self.sessions() as db, patch('app.file_cleanup.get_storage', return_value=self.storage), \
             patch.object(self.storage, 'delete', side_effect=OSError('storage unavailable')):
            result = videos.delete_video(self.video_id, db, SimpleNamespace(id=self.user_id))
        self.assertEqual(result['deleted'], self.video_id)
        self.assertTrue(self.storage.exists('uploads/original.mp4'))
        with self.sessions() as db:
            self.assertIsNone(db.get(db_models.Video, self.video_id))
            self.assertGreater(db.query(db_models.FileCleanup).count(), 0)
        with self.sessions() as db:
            process_pending_cleanup(db, self.storage)
        self.assertFalse(self.storage.exists('uploads/original.mp4'))
        with self.sessions() as db:
            self.assertEqual(db.query(db_models.FileCleanup).count(), 0)

    def test_shared_clip_is_kept_until_last_event_is_deleted(self):
        for key in ('clips/shared.mp4', 'clips/shared_raw.mp4'):
            path = self.storage.path_for(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'clip')
        with self.sessions.begin() as db:
            task = db_models.AnalysisTask(video_id=self.video_id, status='SUCCESS', bbox_xmin=0,
                                          bbox_ymin=0, bbox_xmax=5, bbox_ymax=5)
            db.add(task)
            db.flush()
            first = db_models.CrashEvent(task_id=task.id, video_id=self.video_id,
                                        timestamp_sec=1, frame_number=1, clip_path='clips/shared.mp4')
            second = db_models.CrashEvent(task_id=task.id, video_id=self.video_id,
                                         timestamp_sec=2, frame_number=2, clip_path='clips/shared.mp4')
            db.add_all([first, second])
            db.flush()
            first_id, second_id = first.id, second.id
        with patch('app.file_cleanup.get_storage', return_value=self.storage):
            with self.sessions() as db:
                videos.delete_crash_event(self.video_id, first_id, db, SimpleNamespace(id=self.user_id))
            self.assertTrue(self.storage.exists('clips/shared.mp4'))
            self.assertTrue(self.storage.exists('clips/shared_raw.mp4'))
            with self.sessions() as db:
                videos.delete_crash_event(self.video_id, second_id, db, SimpleNamespace(id=self.user_id))
        self.assertFalse(self.storage.exists('clips/shared.mp4'))
        self.assertFalse(self.storage.exists('clips/shared_raw.mp4'))

    def test_failed_event_commit_preserves_clip(self):
        path = self.storage.path_for('clips/example.mp4')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'clip')
        with self.sessions.begin() as db:
            task = db_models.AnalysisTask(video_id=self.video_id, status='SUCCESS', bbox_xmin=0,
                                          bbox_ymin=0, bbox_xmax=5, bbox_ymax=5)
            db.add(task)
            db.flush()
            event = db_models.CrashEvent(task_id=task.id, video_id=self.video_id,
                                         timestamp_sec=1, frame_number=1, clip_path='clips/example.mp4')
            db.add(event)
            db.flush()
            event_id = event.id
        with self.sessions() as db, patch.object(db, 'commit', side_effect=RuntimeError('database failure')):
            with self.assertRaises(RuntimeError):
                videos.delete_crash_event(self.video_id, event_id, db, SimpleNamespace(id=self.user_id))
            db.rollback()
        self.assertTrue(self.storage.exists('clips/example.mp4'))
        with self.sessions() as db:
            self.assertIsNotNone(db.get(db_models.CrashEvent, event_id))

    def test_video_with_analysis_and_clip_deletes_after_commit(self):
        path = self.storage.path_for('clips/example.mp4')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'clip')
        with self.sessions.begin() as db:
            task = db_models.AnalysisTask(video_id=self.video_id, status='SUCCESS', bbox_xmin=0,
                                          bbox_ymin=0, bbox_xmax=5, bbox_ymax=5)
            db.add(task)
            db.flush()
            db.add(db_models.CrashEvent(task_id=task.id, video_id=self.video_id,
                                       timestamp_sec=1, frame_number=1, clip_path='clips/example.mp4'))
        with self.sessions() as db, patch('app.file_cleanup.get_storage', return_value=self.storage):
            result = videos.delete_video(self.video_id, db, SimpleNamespace(id=self.user_id))
        self.assertEqual(result['deleted'], self.video_id)
        self.assertFalse(self.storage.exists('uploads/original.mp4'))
        self.assertFalse(self.storage.exists('clips/example.mp4'))
        with self.sessions() as db:
            self.assertIsNone(db.get(db_models.Video, self.video_id))
            self.assertEqual(db.query(db_models.CrashEvent).count(), 0)
            self.assertEqual(db.query(db_models.FileCleanup).count(), 0)

    def test_periodic_cleanup_retries_even_when_retention_is_disabled(self):
        from app import retention
        with self.sessions() as db, patch('app.file_cleanup.get_storage', return_value=self.storage), \
             patch.object(self.storage, 'delete', side_effect=OSError('storage unavailable')):
            videos.delete_video(self.video_id, db, SimpleNamespace(id=self.user_id))
        with patch.object(retention, 'SessionLocal', self.sessions), \
             patch.object(retention.settings, 'VIDEO_RETENTION_DAYS', 0), \
             patch('app.file_cleanup.get_storage', return_value=self.storage):
            self.assertEqual(retention.delete_expired_videos(), [])
        self.assertFalse(self.storage.exists('uploads/original.mp4'))
