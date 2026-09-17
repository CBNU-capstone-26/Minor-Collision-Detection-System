from pathlib import Path

import pytest

from backend.app.object_storage import LocalObjectStorage, create_storage


def test_local_storage_upload_download_and_delete(tmp_path):
    storage = LocalObjectStorage(tmp_path)
    source = tmp_path / "source.mp4"
    destination = tmp_path / "downloaded.mp4"
    source.write_bytes(b"video-data")

    storage.upload_path(source, "uploads/video.mp4")
    assert storage.exists("uploads/video.mp4")

    storage.download_to_path("uploads/video.mp4", destination)
    assert destination.read_bytes() == b"video-data"

    storage.delete("uploads/video.mp4")
    assert not storage.exists("uploads/video.mp4")


def test_local_storage_rejects_path_traversal(tmp_path):
    storage = LocalObjectStorage(tmp_path)

    with pytest.raises(ValueError):
        storage.path_for("../outside.mp4")


def test_create_storage_requires_remote_credentials(monkeypatch):
    monkeypatch.setenv("STORAGE_BACKEND", "s3")
    monkeypatch.delenv("S3_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("S3_BUCKET", raising=False)
    monkeypatch.delenv("S3_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("S3_SECRET_ACCESS_KEY", raising=False)

    with pytest.raises(ValueError, match="S3_ENDPOINT_URL"):
        create_storage(
            backend="s3",
            local_root=Path("/tmp/storage-test"),
            endpoint_url=None,
            bucket=None,
            access_key_id=None,
            secret_access_key=None,
            region="auto",
        )
