"""Verify a decrypted backup stream without writing its contents to disk."""

import sys

from app.backup_archive import BackupError, verify_archive


def main():
    try:
        result = verify_archive(sys.stdin.buffer)
    except BackupError:
        print("Backup verification failed.", file=sys.stderr)
        return 1
    print(f"Backup verified: database dump and {result['objects']} objects.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
