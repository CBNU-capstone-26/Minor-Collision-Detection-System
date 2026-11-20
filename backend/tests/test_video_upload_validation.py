"""Validate upload metadata before writing a file to object storage."""

import unittest
import tempfile
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException, UploadFile

from app.routers.videos import upload_video


class VideoUploadValidationTests(unittest.TestCase):
    def test_interrupted_upload_removes_partial_temporary_file(self):
        class InterruptedUpload(BytesIO):
            def read(self, size=-1):
                raise OSError('upload interrupted')

        upload = UploadFile(filename='sample.mp4', file=InterruptedUpload(b'video'))
        with tempfile.TemporaryDirectory() as directory:
            create_file = tempfile.NamedTemporaryFile
            with patch('app.routers.videos.get_storage'), \
                 patch('app.routers.videos.tempfile.NamedTemporaryFile',
                       side_effect=lambda **kwargs: create_file(dir=directory, **kwargs)):
                with self.assertRaises(OSError):
                    upload_video(file=upload, recording_date=None, title=None,
                                 db=None, user=SimpleNamespace(id=1))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_invalid_recording_date_is_rejected_before_storage_access(self):
        upload = UploadFile(filename="sample.mp4", file=BytesIO(b"video"))

        with patch("app.routers.videos.get_storage") as get_storage:
            with self.assertRaises(HTTPException) as raised:
                upload_video(
                    file=upload,
                    recording_date="not-a-date",
                    title=None,
                    db=None,
                    user=None,
                )

        self.assertEqual(raised.exception.status_code, 400)
        get_storage.assert_not_called()


if __name__ == "__main__":
    unittest.main()
