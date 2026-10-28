import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from app import db_models
from app.auth_guard import create_token, MEDIA_COOKIE
from app.db_connection import get_db
from app.object_storage import LocalObjectStorage
from app.routers import videos
from app.thumbnails import ensure_thumbnail, thumbnail_key, delete_thumbnail_cache


class ThumbnailTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def legacy_thumbnail(self, storage):
        path = storage.path_for("thumbnails/example.jpg")
        path.parent.mkdir(parents=True, exist_ok=True)
        with Image.effect_noise((1920, 1080), 64).convert("RGB") as image:
            image.save(path, "JPEG", quality=95)
        return path

    def test_legacy_image_is_reduced_without_reading_the_original_video(self):
        storage = LocalObjectStorage(self.root / "objects")
        legacy = self.legacy_thumbnail(storage)
        original = legacy.read_bytes()

        preview = ensure_thumbnail(storage, "uploads/example.mp4", self.root / "cache")

        with Image.open(preview) as image:
            self.assertEqual(image.size, (640, 360))
        self.assertLess(preview.stat().st_size, len(original) // 5)
        self.assertEqual(legacy.read_bytes(), original)
        self.assertTrue(storage.exists(thumbnail_key("uploads/example.mp4")))

    def test_cached_preview_survives_remote_storage_becoming_unavailable(self):
        objects = LocalObjectStorage(self.root / "objects")
        self.legacy_thumbnail(objects)
        remote = Mock(wraps=objects)
        preview = ensure_thumbnail(remote, "uploads/example.mp4", self.root / "cache")
        remote.reset_mock()
        remote.exists.side_effect = RuntimeError("network unavailable")

        self.assertEqual(ensure_thumbnail(remote, "uploads/example.mp4", self.root / "cache"), preview)
        remote.exists.assert_not_called()
        remote.download_to_path.assert_not_called()
        remote.upload_path.assert_not_called()

    def test_new_upload_uses_the_temporary_video_without_remote_downloads(self):
        import cv2
        import numpy as np

        source = self.root / "upload.avi"
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 90))
        self.assertTrue(writer.isOpened())
        try:
            writer.write(np.full((90, 160, 3), (30, 60, 180), dtype=np.uint8))
        finally:
            writer.release()
        remote = Mock(wraps=LocalObjectStorage(self.root / "objects"))

        preview = ensure_thumbnail(remote, "uploads/new.avi", self.root / "cache",
                                   source_video=source)

        with Image.open(preview) as image:
            self.assertEqual(image.size, (160, 90))
        remote.exists.assert_not_called()
        remote.download_to_path.assert_not_called()
        remote.upload_path.assert_called_once()

    def test_existing_remote_preview_populates_disk_cache_without_upload(self):
        objects = LocalObjectStorage(self.root / "objects")
        key = thumbnail_key("uploads/example.mp4")
        remote_file = objects.path_for(key)
        remote_file.parent.mkdir(parents=True)
        Image.new("RGB", (640, 360)).save(remote_file)
        remote = Mock(wraps=objects)

        preview = ensure_thumbnail(remote, "uploads/example.mp4", self.root / "cache")

        self.assertEqual(preview.read_bytes(), remote_file.read_bytes())
        remote.download_to_path.assert_called_once()
        remote.upload_path.assert_not_called()
        delete_thumbnail_cache("uploads/example.mp4", self.root / "cache")
        self.assertFalse(preview.exists())

    def test_browser_cache_is_revalidated_only_after_authentication_and_ownership(self):
        image_path = self.root / "preview.jpg"
        Image.new("RGB", (64, 36)).save(image_path)
        owner = SimpleNamespace(id=1)
        other_user = SimpleNamespace(id=2)
        video = SimpleNamespace(id=1, user_id=1, video_path="uploads/example.mp4")
        db = Mock()
        db.get.side_effect = lambda model, key: (
            video if model is db_models.Video else (owner if key == 1 else other_user)
        )
        app = FastAPI()
        app.include_router(videos.router)
        app.dependency_overrides[get_db] = lambda: db
        with patch("app.auth_guard.auth_secret_key", return_value="test-signing-key-" * 4), \
                patch.object(videos, "ensure_thumbnail", return_value=image_path) as ensure, \
                patch.object(videos, "get_storage"), TestClient(app) as client:
            token = create_token(owner, scope="media")
            headers = {"Cookie": f"{MEDIA_COOKIE}={token}"}
            response = client.get("/api/videos/1/thumbnail", headers=headers)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content, image_path.read_bytes())
            self.assertEqual(response.headers["cache-control"], "private, no-cache")

            headers["If-None-Match"] = response.headers["etag"]
            cached = client.get("/api/videos/1/thumbnail", headers=headers)
            self.assertEqual(cached.status_code, 304)
            self.assertEqual(cached.content, b"")

            ensure.reset_mock()
            unauthorized = client.get("/api/videos/1/thumbnail", headers={"If-None-Match": "*"})
            self.assertEqual(unauthorized.status_code, 401)
            headers["Cookie"] = f"{MEDIA_COOKIE}={create_token(other_user, scope='media')}"
            self.assertEqual(client.get("/api/videos/1/thumbnail", headers=headers).status_code, 404)
            ensure.assert_not_called()
