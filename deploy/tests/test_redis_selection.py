"""Check which Redis endpoint the deployment installer selects."""

import os
from pathlib import Path
import subprocess
import unittest


INSTALLER = Path(__file__).resolve().parents[1] / "install.sh"


def local_redis_selected(values):
    script = INSTALLER.read_text(encoding="utf-8")
    start = script.index('REDIS_URL="$(env_get REDIS_URL)"')
    end = script.index('[[ -n "$(env_get AUTH_SECRET_KEY)"', start)
    setup = r'''
env_get() { printf '%s' "${!1-}"; }
env_first() {
    local name value
    for name in "$@"; do
        value="$(env_get "$name")"
        if [[ -n "$value" ]]; then printf '%s' "$value"; return; fi
    done
}
'''
    env = {"PATH": os.environ["PATH"], **values}
    return subprocess.run(
        ["bash"], input=setup + script[start:end] + '\nprintf "%s" "$LOCAL_REDIS"\n',
        text=True, capture_output=True, check=True, env=env,
    ).stdout == "1"


class RedisSelectionTests(unittest.TestCase):
    def test_azure_connection_string_selects_remote_redis(self):
        self.assertFalse(local_redis_selected({
            "AZURE_REDIS_CONNECTIONSTRING": "cache.example.net:6380,password=test,ssl=True",
        }))

    def test_remote_host_with_localhost_in_credentials_is_remote(self):
        self.assertFalse(local_redis_selected({
            "REDIS_URL": "redis://localhost:test@cache.example.net:6379/0",
        }))

    def test_loopback_url_without_port_selects_local_redis(self):
        self.assertTrue(local_redis_selected({"REDIS_URL": "redis://127.0.0.1/0"}))

    def test_default_and_ipv6_loopback_select_local_redis(self):
        self.assertTrue(local_redis_selected({}))
        self.assertTrue(local_redis_selected({"REDIS_URL": "redis://[::1]:6379/0"}))

    def test_redis_url_takes_precedence_over_azure_connection(self):
        self.assertTrue(local_redis_selected({
            "REDIS_URL": "redis://localhost:6379/0",
            "AZURE_REDIS_CONNECTIONSTRING": "cache.example.net:6380,password=test",
        }))

    def test_explicit_host_overrides_azure_connection_string(self):
        self.assertTrue(local_redis_selected({
            "REDIS_HOST": "localhost",
            "AZURE_REDIS_CONNECTIONSTRING": "cache.example.net:6380,password=test",
        }))

    def test_colon_in_azure_password_does_not_create_a_host_port(self):
        self.assertTrue(local_redis_selected({
            "AZURE_REDIS_CONNECTIONSTRING": "cache.example.net,password=a:b",
        }))


if __name__ == "__main__":
    unittest.main()
