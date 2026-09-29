"""로컬 또는 S3 호환 객체 저장소 추상화.

DB에는 이 모듈이 반환하는 object key만 저장하고, 원격 객체는 presigned URL로 제공한다.
"""
from __future__ import annotations

import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO


class LocalObjectStorage:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if path != self.root and self.root not in path.parents:
            raise ValueError("storage key must stay inside the storage root")
        return path

    def upload_fileobj(self, fileobj: BinaryIO, key: str, content_type: str | None = None):
        del content_type
        destination = self.path_for(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as output:
            shutil.copyfileobj(fileobj, output)

    def upload_path(self, source: str | Path, key: str, content_type: str | None = None):
        del content_type
        destination = self.path_for(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    def download_to_path(self, key: str, destination: str | Path):
        shutil.copy2(self.path_for(key), destination)

    def delete(self, key: str):
        self.path_for(key).unlink(missing_ok=True)

    def exists(self, key: str) -> bool:
        return self.path_for(key).is_file()

    def local_path(self, key: str) -> Path:
        return self.path_for(key)

    def presigned_url(self, key: str, expires: int = 3600) -> str | None:
        del key, expires
        return None


class S3ObjectStorage:
    def __init__(
        self,
        *,
        endpoint_url: str,
        bucket: str,
        access_key_id: str,
        secret_access_key: str,
        region: str,
    ):
        import boto3

        self.bucket = bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name=region,
        )

    def upload_fileobj(self, fileobj: BinaryIO, key: str, content_type: str | None = None):
        extra_args = {"ContentType": content_type} if content_type else None
        self.client.upload_fileobj(fileobj, self.bucket, key, ExtraArgs=extra_args or {})

    def upload_path(self, source: str | Path, key: str, content_type: str | None = None):
        extra_args = {"ContentType": content_type} if content_type else None
        self.client.upload_file(str(source), self.bucket, key, ExtraArgs=extra_args or {})

    def download_to_path(self, key: str, destination: str | Path):
        self.client.download_file(self.bucket, key, str(destination))

    def delete(self, key: str):
        self.client.delete_object(Bucket=self.bucket, Key=key)

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
        except self.client.exceptions.ClientError as error:
            if error.response.get("Error", {}).get("Code") in {"404", "NoSuchKey"}:
                return False
            raise
        return True

    def local_path(self, key: str) -> Path:
        raise RuntimeError("remote objects do not have a persistent local path")

    def presigned_url(self, key: str, expires: int = 3600) -> str:
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires,
        )


def create_storage(
    *,
    backend: str,
    local_root: str | Path,
    endpoint_url: str | None,
    bucket: str | None,
    access_key_id: str | None,
    secret_access_key: str | None,
    region: str,
):
    if backend == "local":
        return LocalObjectStorage(local_root)
    if backend != "s3":
        raise ValueError("STORAGE_BACKEND must be 'local' or 's3'")

    missing = [
        name for name, value in {
            "S3_ENDPOINT_URL": endpoint_url,
            "S3_BUCKET": bucket,
            "S3_ACCESS_KEY_ID": access_key_id,
            "S3_SECRET_ACCESS_KEY": secret_access_key,
        }.items() if not value
    ]
    if missing:
        raise ValueError(f"missing object storage settings: {', '.join(missing)}")

    return S3ObjectStorage(
        endpoint_url=endpoint_url,
        bucket=bucket,
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        region=region,
    )


_storage = None


def get_storage():
    global _storage
    if _storage is None:
        from app.settings import settings

        _storage = create_storage(
            backend=settings.STORAGE_BACKEND,
            local_root=settings.STORAGE_DIR,
            endpoint_url=settings.S3_ENDPOINT_URL,
            bucket=settings.S3_BUCKET,
            access_key_id=settings.S3_ACCESS_KEY_ID,
            secret_access_key=settings.S3_SECRET_ACCESS_KEY,
            region=settings.S3_REGION,
        )
    return _storage


@contextmanager
def materialize(storage, key: str, suffix: str = ""):
    """객체를 OpenCV/모델이 읽을 수 있는 임시 로컬 파일로 준비한다."""
    if isinstance(storage, LocalObjectStorage):
        yield storage.local_path(key)
        return

    with tempfile.TemporaryDirectory(prefix="remote-object-") as temp_dir:
        path = Path(temp_dir) / (Path(key).name + suffix)
        storage.download_to_path(key, path)
        yield path
