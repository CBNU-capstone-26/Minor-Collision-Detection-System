"""Stream a database dump and S3 objects into one verifiable backup archive."""

import hashlib
import io
import json
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone


class BackupError(RuntimeError):
    pass


def dump_connection(url):
    from sqlalchemy.engine import URL

    if set(url.query) - {"sslmode", "connect_timeout", "target_session_attrs"}:
        raise BackupError("unsupported database URL option for containerized pg_dump")
    password = url.password or ""
    if "\n" in password or "\r" in password:
        raise BackupError("database password contains a line break")
    safe_url = URL.create(
        "postgresql", username=url.username, host=url.host, port=url.port,
        database=url.database, query=url.query,
    ).render_as_string(hide_password=False)
    return safe_url, password


def verify_archive(source):
    """Verify an archive without extracting paths or writing plaintext to disk."""
    observed = {}
    database_chunks = []
    database_hash = hashlib.sha256()
    database_size = 0
    manifest = None
    try:
        with tarfile.open(fileobj=source, mode="r|*") as archive:
            for member in archive:
                if not member.isfile() or member.name in observed or manifest is not None:
                    raise BackupError("unexpected or duplicate archive member")
                stream = archive.extractfile(member)
                if member.name == "manifest.json":
                    if member.size > 32 * 1024 * 1024:
                        raise BackupError("backup manifest is too large")
                    manifest = json.loads(stream.read())
                    continue
                if member.name.startswith("database/"):
                    expected = f"database/{len(database_chunks):08d}.part"
                    if member.name != expected:
                        raise BackupError("database chunks are out of order")
                    database_chunks.append(member.name)
                elif member.name != "roles.sql" and not member.name.startswith("objects/"):
                    raise BackupError("unexpected archive path")
                digest = hashlib.sha256()
                size = 0
                while True:
                    chunk = stream.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    size += len(chunk)
                    if member.name.startswith("database/"):
                        database_hash.update(chunk)
                        database_size += len(chunk)
                observed[member.name] = {"size": size, "sha256": digest.hexdigest()}
    except (tarfile.TarError, json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        raise BackupError("cannot read backup archive") from exc

    if not isinstance(manifest, dict) or manifest.get("version") not in (1, 2):
        raise BackupError("missing or unsupported backup manifest")
    db = manifest.get("database")
    objects = manifest.get("objects")
    required = manifest.get("required_keys")
    if not isinstance(db, dict) or not isinstance(objects, list) or not isinstance(required, list):
        raise BackupError("invalid backup manifest")
    if (not database_chunks or db.get("chunks") != database_chunks or
            db.get("size") != database_size or db.get("sha256") != database_hash.hexdigest()):
        raise BackupError("database dump checksum mismatch")

    keys = set()
    paths = set(database_chunks)
    if manifest['version'] == 2:
        roles = manifest.get('roles')
        if not isinstance(roles, dict) or roles.get('size', 0) <= 0 or observed.get('roles.sql') != roles:
            raise BackupError('role definitions checksum mismatch')
        paths.add('roles.sql')
    for obj in objects:
        if not isinstance(obj, dict) or not isinstance(obj.get("key"), str):
            raise BackupError("invalid object manifest")
        key = obj["key"]
        path = "objects/" + hashlib.sha256(key.encode("utf-8")).hexdigest()
        if key in keys or obj.get("archive_path") != path or path in paths:
            raise BackupError("duplicate or invalid object path")
        keys.add(key)
        paths.add(path)
        if observed.get(path) != {"size": obj.get("size"), "sha256": obj.get("sha256")}:
            raise BackupError("object checksum mismatch")
    if set(observed) != paths or any(not isinstance(key, str) or key not in keys for key in required):
        raise BackupError("missing or unexpected backup object")
    return {"objects": len(objects)}


def _add_bytes(archive, name, data):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o600
    info.mtime = 0
    archive.addfile(info, io.BytesIO(data))


class _HashingReader:
    def __init__(self, body):
        self.body = body
        self.digest = hashlib.sha256()
        self.size = 0

    def read(self, size):
        chunks = []
        remaining = size
        while remaining:
            chunk = self.body.read(remaining)
            if not chunk:
                break
            chunks.append(chunk)
            self.digest.update(chunk)
            self.size += len(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)


def build_archive(output, database_stream, database_status, client, bucket, *, chunk_size=8 * 1024 * 1024, required_keys=(), roles_dump=None):
    """Do not complete an archive when pg_dump or any source object fails."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    db_hash = hashlib.sha256()
    db_size = 0
    db_chunks = []
    objects = []

    with tarfile.open(fileobj=output, mode="w|") as archive:
        while True:
            chunk = database_stream.read(chunk_size)
            if not chunk:
                break
            path = f"database/{len(db_chunks):08d}.part"
            _add_bytes(archive, path, chunk)
            db_chunks.append(path)
            db_hash.update(chunk)
            db_size += len(chunk)
        if database_status() != 0 or not db_size:
            raise BackupError("database dump failed or was empty")

        if roles_dump is not None:
            if not roles_dump or len(roles_dump) > 10 * 1024 * 1024:
                raise BackupError('role definitions are empty or too large')
            _add_bytes(archive, 'roles.sql', roles_dump)

        pages = client.get_paginator("list_objects_v2").paginate(Bucket=bucket)
        entries = [item for page in pages for item in page.get("Contents", [])]
        required = {key for key in required_keys if key}
        if required - {item["Key"] for item in entries}:
            raise BackupError("database references an object missing from storage")
        for item in sorted(entries, key=lambda item: item["Key"]):
            key = item["Key"]
            response = client.get_object(Bucket=bucket, Key=key, IfMatch=item["ETag"])
            body = response["Body"]
            try:
                size = response["ContentLength"]
                if size != item["Size"]:
                    raise BackupError("source object changed during backup")
                path = "objects/" + hashlib.sha256(key.encode("utf-8")).hexdigest()
                info = tarfile.TarInfo(path)
                info.size = size
                info.mode = 0o600
                info.mtime = 0
                reader = _HashingReader(body)
                archive.addfile(info, reader)
                if reader.size != size:
                    raise BackupError("source object was incomplete")
                objects.append({
                    "key": key, "archive_path": path, "size": size,
                    "sha256": reader.digest.hexdigest(),
                    "content_type": response.get("ContentType"),
                    "cache_control": response.get("CacheControl"),
                    "content_encoding": response.get("ContentEncoding"),
                    "content_disposition": response.get("ContentDisposition"),
                    "metadata": response.get("Metadata", {}),
                })
            finally:
                body.close()

        manifest = {
            "version": 2 if roles_dump is not None else 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "database": {"chunks": db_chunks, "size": db_size, "sha256": db_hash.hexdigest()},
            "required_keys": sorted(required),
            "objects": objects,
        }
        if roles_dump is not None:
            manifest['roles'] = {'size': len(roles_dump), 'sha256': hashlib.sha256(roles_dump).hexdigest()}
        _add_bytes(archive, "manifest.json", json.dumps(manifest, ensure_ascii=False).encode("utf-8"))


def main():
    from sqlalchemy import text

    from app.db_connection import engine
    from app.object_storage import S3ObjectStorage, get_storage

    storage = get_storage()
    if not isinstance(storage, S3ObjectStorage) or engine.dialect.name != "postgresql":
        raise BackupError("backup requires PostgreSQL and S3 object storage")
    url, password = dump_connection(engine.url)
    roles_command = [
        'docker', 'run', '--rm', '-i', 'postgres:17-alpine', 'sh', '-c',
        'IFS= read -r SIOT_DB_URL; IFS= read -r SIOT_DB_PASSWORD; '
        'export PGPASSWORD="$SIOT_DB_PASSWORD"; exec pg_dumpall --no-password '
        '--roles-only --no-role-passwords --no-comments --dbname="$SIOT_DB_URL"',
    ]
    role_result = subprocess.run(roles_command, input=(url + '\n' + password + '\n').encode(),
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    command = [
        "docker", "run", "--rm", "-i", "postgres:17-alpine", "sh", "-c",
        'IFS= read -r SIOT_DB_URL; IFS= read -r SIOT_DB_PASSWORD; '
        'IFS= read -r SIOT_DB_SNAPSHOT; export PGPASSWORD="$SIOT_DB_PASSWORD"; '
        'exec pg_dump --no-password --format=custom --no-owner --schema=public '
        '--snapshot="$SIOT_DB_SNAPSHOT" --dbname="$SIOT_DB_URL"',
    ]
    with engine.connect().execution_options(isolation_level="REPEATABLE READ") as connection:
        transaction = connection.begin()
        try:
            connection.execute(text("SET TRANSACTION READ ONLY"))
            snapshot = connection.execute(text("SELECT pg_export_snapshot()")).scalar_one()
            required = set(connection.execute(text(
                "SELECT video_path FROM videos WHERE video_path IS NOT NULL"
            )).scalars())
            required.update(connection.execute(text(
                "SELECT cam_heatmap_path FROM crash_events WHERE cam_heatmap_path IS NOT NULL"
            )).scalars())
            with tempfile.TemporaryFile() as errors:
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors)
                try:
                    process.stdin.write((url + "\n" + password + "\n" + snapshot + "\n").encode("utf-8"))
                    process.stdin.close()

                    def finish_dump():
                        result = process.wait()
                        transaction.rollback()  # Release the exported snapshot before copying media.
                        connection.close()
                        return result

                    with storage._use_client() as client:
                        build_archive(sys.stdout.buffer, process.stdout, finish_dump, client, storage.bucket,
                                      required_keys=required, roles_dump=role_result.stdout)
                finally:
                    process.stdout.close()
                    if process.poll() is None:
                        process.kill()
                    process.wait()
        finally:
            if transaction.is_active:
                transaction.rollback()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("Backup archive failed; no complete snapshot was created.", file=sys.stderr)
        raise SystemExit(1)
