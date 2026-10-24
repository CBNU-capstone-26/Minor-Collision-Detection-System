"""Supabase 스키마와 R2 저장소 경로가 어긋나지 않게 하는 회귀 검사."""
import unittest
from types import SimpleNamespace

from fastapi.responses import RedirectResponse

from app.db_models import Video
from app.routers import videos


class CloudStorageIntegrationTests(unittest.TestCase):
    def test_video_model_does_not_request_removed_recording_start_time_column(self):
        self.assertNotIn("recording_start_time", Video.__table__.columns)

    def test_remote_video_stream_redirects_to_presigned_r2_url(self):
        class RemoteStorage:
            def exists(self, key):
                return key == "uploads/example.mp4"

            def presigned_url(self, key, expires):
                self.assert_key = key
                self.assert_expires = expires
                return "https://r2.example.test/uploads/example.mp4?signature=test"

        storage = RemoteStorage()
        original_storage = videos.get_storage
        original_lookup = videos._get_owned_video
        try:
            videos.get_storage = lambda: storage
            videos._get_owned_video = lambda *_: SimpleNamespace(video_path="uploads/example.mp4")
            response = videos.stream_video(1, db=None, user=None)
        finally:
            videos.get_storage = original_storage
            videos._get_owned_video = original_lookup

        self.assertIsInstance(response, RedirectResponse)
        self.assertEqual(response.headers["location"],
                         "https://r2.example.test/uploads/example.mp4?signature=test")


if __name__ == "__main__":
    unittest.main()
