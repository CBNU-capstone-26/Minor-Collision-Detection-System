"""Create, verify and retain encrypted off-site restic snapshots."""

import fcntl
import json
import os
import stat
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit

from app.backup_archive import verify_archive


def validate_repository(repository, source_endpoint, source_bucket):
    if not repository.startswith('s3:https://'):
        raise ValueError('backup repository must use HTTPS S3')
    target = urlsplit(repository[3:])
    source = urlsplit(source_endpoint or '')
    parts = target.path.strip('/').split('/')
    if (not target.hostname or target.username or target.password or target.query or target.fragment
            or len(parts) < 2 or any(not p or p in ('.', '..') for p in parts)
            or '%' in target.path):
        raise ValueError('invalid backup repository')
    if not source.hostname or not source_bucket:
        raise ValueError('source storage is not configured')
    if target.hostname.lower() == source.hostname.lower() and parts[0] == source_bucket:
        raise ValueError('backup bucket must differ from source bucket')


def run_backup(restic, env, producer):
    args = [restic, 'backup', '--json', '--host', 'siot-capstone', '--tag', 'siot-capstone',
            '--stdin-filename', 'siot-snapshot.tar', '--stdin-from-command', '--', *producer]
    result = subprocess.run(args, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    summaries = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    ids = [row.get('snapshot_id') for row in summaries if row.get('message_type') == 'summary']
    if len(ids) != 1 or not ids[0]:
        raise ValueError('restic did not report a snapshot')
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen([restic, 'dump', ids[0], '/siot-snapshot.tar'],
                                   env=env, stdout=subprocess.PIPE, stderr=errors)
        try:
            verify_archive(process.stdout)
            while process.stdout.read(1024 * 1024):
                pass
            code = process.wait()
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
                process.wait()
        if code:
            raise ValueError('restic dump failed')
    subprocess.run([restic, 'forget', '--tag', 'siot-capstone', '--group-by', '',
                    '--keep-within', '90d', '--prune'], env=env,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return ids[0]


def main():
    from app.settings import settings

    env = os.environ.copy()
    validate_repository(env.get('RESTIC_REPOSITORY', ''), settings.S3_ENDPOINT_URL, settings.S3_BUCKET)
    for key in ('SIOT_BACKUP_ACCESS_KEY_ID', 'SIOT_BACKUP_SECRET_ACCESS_KEY', 'RESTIC_PASSWORD_FILE'):
        if not env.get(key):
            raise ValueError('missing backup configuration')
    password = os.stat(env['RESTIC_PASSWORD_FILE'], follow_symlinks=False)
    if not stat.S_ISREG(password.st_mode) or password.st_mode & 0o077:
        raise ValueError('backup password file must be private')
    env['AWS_ACCESS_KEY_ID'] = env['SIOT_BACKUP_ACCESS_KEY_ID']
    env['AWS_SECRET_ACCESS_KEY'] = env['SIOT_BACKUP_SECRET_ACCESS_KEY']
    with open('/tmp/siot-backup-' + str(os.getuid()) + '.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_backup(env.get('SIOT_RESTIC_BIN', 'restic'), env,
                   [sys.executable, '-m', 'app.backup_archive'])
    print('Backup created and verified; 90-day retention applied.')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print('Backup failed; inspect configuration and repository before retrying.', file=sys.stderr)
        sys.exit(1)
