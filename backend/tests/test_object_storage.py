import unittest
from unittest.mock import patch

from app.object_storage import S3ObjectStorage


class S3ObjectStorageTests(unittest.TestCase):
    @patch("boto3.client")
    def test_client_has_bounded_network_timeouts(self, client):
        S3ObjectStorage(
            endpoint_url="https://example.r2.cloudflarestorage.com",
            bucket="videos",
            access_key_id="access",
            secret_access_key="secret",
            region="auto",
        )

        config = client.call_args.kwargs["config"]
        self.assertEqual(config.connect_timeout, 3)
        self.assertEqual(config.read_timeout, 5)
