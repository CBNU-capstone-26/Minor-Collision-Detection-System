import unittest
from unittest.mock import Mock, patch

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
        self.assertTrue(config.tcp_keepalive)

    def make_storage(self):
        return S3ObjectStorage(endpoint_url="https://r2.example.test", bucket="videos",
                               access_key_id="access", secret_access_key="secret", region="auto")

    @patch("app.object_storage.time.monotonic", return_value=0)
    @patch("boto3.client")
    def test_first_request_after_idle_uses_a_new_connection_pool(self, factory, clock):
        old, fresh = Mock(), Mock()
        factory.side_effect = [old, fresh]
        storage = self.make_storage()
        clock.return_value = 310

        self.assertTrue(storage.exists("thumbnails/preview.jpg"))

        old.head_object.assert_not_called()
        old.close.assert_called_once()
        fresh.head_object.assert_called_once_with(Bucket="videos", Key="thumbnails/preview.jpg")
        self.assertEqual(factory.call_count, 2)

    @patch("app.object_storage.time.monotonic", return_value=0)
    @patch("boto3.client")
    def test_signing_urls_does_not_keep_idle_network_connections_alive(self, factory, clock):
        storage = self.make_storage()
        clock.return_value = 100
        storage.presigned_url("video.mp4")
        clock.return_value = 130
        storage.exists("video.mp4")
        self.assertEqual(factory.call_count, 2)

    @patch("app.object_storage.time.monotonic", return_value=0)
    @patch("boto3.client")
    def test_an_active_transfer_is_not_closed_by_another_request(self, factory, clock):
        client = factory.return_value
        storage = self.make_storage()

        def transfer(*args, **kwargs):
            clock.return_value = 310
            storage.exists("another-video.mp4")
            client.close.assert_not_called()

        client.download_file.side_effect = transfer
        storage.download_to_path("long-video.mp4", "/tmp/video.mp4")
        self.assertEqual(factory.call_count, 1)
        client.close.assert_not_called()
