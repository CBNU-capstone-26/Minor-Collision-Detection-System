import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from app.backup_notify import send_failure_alert


class AlertTest(unittest.TestCase):
    def test_sends_failure_alert_over_tls(self):
        with tempfile.TemporaryDirectory() as tmp:
            secret = Path(tmp)/'password'
            secret.write_text('test-only-password')
            secret.chmod(0o600)
            with patch('app.backup_notify.smtplib.SMTP_SSL') as smtp:
                send_failure_alert('sender@example.com', 'recipient@example.com', secret)
            client = smtp.return_value.__enter__.return_value
            client.login.assert_called_once_with('sender@example.com', 'test-only-password')
            message = client.send_message.call_args.args[0]
            self.assertEqual(message['To'], 'recipient@example.com')
            self.assertIn('backup', message['Subject'].lower())

    def test_refuses_world_readable_password(self):
        with tempfile.TemporaryDirectory() as tmp:
            secret = Path(tmp)/'password'
            secret.write_text('test')
            secret.chmod(0o644)
            with self.assertRaises(ValueError):
                send_failure_alert('sender@example.com', 'recipient@example.com', secret)

    def test_test_message_is_clearly_identified(self):
        with tempfile.TemporaryDirectory() as tmp:
            secret = Path(tmp) / 'password'
            secret.write_text('test')
            secret.chmod(0o600)
            with patch('app.backup_notify.smtplib.SMTP_SSL') as smtp:
                send_failure_alert('sender@example.com', 'recipient@example.com', secret, test=True)
            message = smtp.return_value.__enter__.return_value.send_message.call_args.args[0]
            self.assertIn('test', message['Subject'].lower())
