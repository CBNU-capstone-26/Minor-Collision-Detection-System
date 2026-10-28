import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.routers import videos


class EventSerializationTests(unittest.TestCase):
    def test_event_serialization_does_not_query_object_storage(self):
        event = SimpleNamespace(
            id=1,
            timestamp_sec=1.0,
            frame_number=30,
            end_timestamp_sec=2.0,
            end_frame_number=60,
            crash_prob=0.9,
            clip_path="clips/demo.mp4",
            clip_start_sec=0.5,
            created_at=None,
            task_id=1,
            video=SimpleNamespace(analysis_tasks=[]),
        )

        with patch("app.routers.videos.get_storage") as get_storage:
            result = videos.to_event_out(event)

        get_storage.assert_not_called()
        self.assertTrue(result.has_raw_clip)
