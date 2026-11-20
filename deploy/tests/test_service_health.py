"""The installer must not report success when required services are down."""

from pathlib import Path
import subprocess
import unittest


INSTALLER = Path(__file__).resolve().parents[1] / "install.sh"


def completion_check(failed_service=""):
    script = INSTALLER.read_text(encoding="utf-8")
    start = script.index('  if curl -fsS http://127.0.0.1:8000/')
    end = script.index('  sleep 2\n', start)
    setup = f'''
curl() {{ return 0; }}
systemctl() {{ [[ "$*" != "is-active --quiet {failed_service}" ]]; }}
log() {{ :; }}
die() {{ exit 1; }}
HTTPS=""
DOMAIN=""
'''
    return subprocess.run(["bash"], input=setup + script[start:end], text=True).returncode


class ServiceHealthTests(unittest.TestCase):
    def test_installer_rejects_inactive_backend(self):
        self.assertNotEqual(completion_check("siot-backend"), 0)

    def test_installer_rejects_inactive_worker(self):
        self.assertNotEqual(completion_check("siot-worker"), 0)

    def test_installer_rejects_inactive_nginx(self):
        self.assertNotEqual(completion_check("nginx"), 0)

    def test_installer_accepts_running_services(self):
        self.assertEqual(completion_check(), 0)


if __name__ == "__main__":
    unittest.main()
