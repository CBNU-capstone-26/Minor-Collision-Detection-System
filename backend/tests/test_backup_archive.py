"""Backups contain verifiable data and fail if a source fails."""

import hashlib
import io
import json
import tarfile
import unittest

from app.backup_archive import BackupError, build_archive, dump_connection, verify_archive


class FakeS3:
    def __init__(self, objects, fail=False):
        self.objects = objects
        self.fail = fail

    def get_paginator(self, _name):
        return self

    def paginate(self, Bucket):
        del Bucket
        yield {"Contents": [
            {"Key": key, "Size": len(value), "ETag": '"etag"'}
            for key, value in self.objects.items()
        ]}

    def get_object(self, *, Bucket, Key, IfMatch):
        del Bucket
        if self.fail:
            raise OSError("source object unavailable")
        assert IfMatch == '"etag"'
        data = self.objects[Key]
        return {"Body": io.BytesIO(data), "ContentLength": len(data),
                "ContentType": "video/mp4", "Metadata": {}}


class BackupArchiveTests(unittest.TestCase):
    def test_role_definitions_are_backed_up_and_verified(self):
        output = io.BytesIO()
        build_archive(output, io.BytesIO(b'db'), lambda: 0, FakeS3({}), 'source',
                      roles_dump=b'CREATE ROLE example;\n')
        data = output.getvalue()
        self.assertEqual(verify_archive(io.BytesIO(data)), {'objects': 0})
        with tarfile.open(fileobj=io.BytesIO(data), mode='r:') as archive:
            self.assertEqual(archive.extractfile('roles.sql').read(), b'CREATE ROLE example;\n')
            offset = archive.getmember('roles.sql').offset_data
        corrupted = bytearray(data)
        corrupted[offset] ^= 1
        with self.assertRaises(BackupError):
            verify_archive(io.BytesIO(corrupted))

    def test_dump_connection_does_not_put_password_in_uri(self):
        from sqlalchemy.engine import make_url

        url, password = dump_connection(make_url("postgresql+psycopg://user:p%40ss@db.example/test?sslmode=require"))
        self.assertNotIn("p%40ss", url)
        self.assertNotIn("p@ss", url)
        self.assertEqual(password, "p@ss")
        self.assertIn("sslmode=require", url)

    def test_dump_connection_rejects_unrecognized_connection_options(self):
        from sqlalchemy.engine import make_url

        with self.assertRaises(BackupError):
            dump_connection(make_url("postgresql+psycopg://user:secret@db.example/test?options=unknown"))

    def test_archive_contains_checked_database_chunks_and_safe_object_paths(self):
        database = b"test database snapshot"
        objects = {"uploads/../example.mp4": b"video contents", "clips/a.mp4": b"clip"}
        output = io.BytesIO()
        build_archive(output, io.BytesIO(database), lambda: 0, FakeS3(objects), "source",
                      chunk_size=7, required_keys={"clips/a.mp4"})

        output.seek(0)
        with tarfile.open(fileobj=output, mode="r:") as archive:
            manifest = json.load(archive.extractfile("manifest.json"))
            db = b"".join(archive.extractfile(part).read() for part in manifest["database"]["chunks"])
            self.assertEqual(db, database)
            self.assertEqual(manifest["database"]["sha256"], hashlib.sha256(database).hexdigest())
            self.assertEqual(manifest["required_keys"], ["clips/a.mp4"])
            for obj in manifest["objects"]:
                self.assertTrue(obj["archive_path"].startswith("objects/"))
                self.assertNotIn("..", obj["archive_path"])
                data = archive.extractfile(obj["archive_path"]).read()
                self.assertEqual(data, objects[obj["key"]])
                self.assertEqual(obj["sha256"], hashlib.sha256(data).hexdigest())
                self.assertEqual(obj["content_type"], "video/mp4")

    def test_failed_database_dump_aborts_the_archive(self):
        with self.assertRaises(BackupError):
            build_archive(io.BytesIO(), io.BytesIO(b"partial"), lambda: 1, FakeS3({}), "source")

    def test_failed_object_read_aborts_the_archive(self):
        with self.assertRaises(OSError):
            build_archive(io.BytesIO(), io.BytesIO(b"db"), lambda: 0,
                          FakeS3({"uploads/a": b"video"}, fail=True), "source")

    def test_missing_database_referenced_object_aborts_the_archive(self):
        with self.assertRaises(BackupError):
            build_archive(io.BytesIO(), io.BytesIO(b"db"), lambda: 0,
                          FakeS3({"uploads/a": b"video"}), "source",
                          required_keys={"uploads/missing"})

    def test_verifier_detects_corrupted_object_data(self):
        output = io.BytesIO()
        build_archive(output, io.BytesIO(b"db"), lambda: 0,
                      FakeS3({"uploads/a": b"video"}), "source", required_keys={"uploads/a"})
        self.assertEqual(verify_archive(io.BytesIO(output.getvalue())), {"objects": 1})
        with tarfile.open(fileobj=io.BytesIO(output.getvalue()), mode="r:") as archive:
            path = "objects/" + hashlib.sha256(b"uploads/a").hexdigest()
            offset = archive.getmember(path).offset_data
        corrupted = bytearray(output.getvalue())
        corrupted[offset] ^= 1
        with self.assertRaises(BackupError):
            verify_archive(io.BytesIO(corrupted))


if __name__ == "__main__":
    unittest.main()
