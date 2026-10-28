"""애플리케이션 설정 — DB/Redis 접속 정보, 로컬 저장소 경로.

GPU·대용량 스토리지 서버가 없는 환경을 가정한다.
영상/클립은 프로젝트 루트의 storage/ 폴더(로컬 파일시스템)에 저장한다.
"""
import functools
import os
import shlex
from pathlib import Path
from urllib.parse import quote, urlencode

from dotenv import load_dotenv


# backend/.env keeps precedence for local development credentials. The root
# .env is also loaded as a fallback so dev.sh uses the same provider settings
# as Docker Compose and the systemd deployment.
_BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(_BASE_DIR / "backend" / ".env")
load_dotenv(_BASE_DIR / ".env")


def _env(name: str, default: str = "") -> str:
    """Treat unset and blank dotenv values alike; deployment templates contain blanks."""
    return os.getenv(name, "").strip() or default


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _env_first(*names: str, default: str = "") -> str:
    for name in names:
        value = _env(name)
        if value:
            return value
    return default


def _database_url() -> str:
    configured = _env_first("DATABASE_URL", "AZURE_POSTGRESQL_CONNECTIONSTRING")
    if configured:
        if "://" not in configured:
            if ";" in configured:
                parts = (part.strip() for part in configured.split(";"))
            else:
                parts = shlex.split(configured)
            options = {}
            for part in parts:
                key, separator, value = part.partition("=")
                if separator:
                    options[key.strip().lower().replace(" ", "_")] = value.strip()
            host = options.get("host") or options.get("server")
            user = options.get("user") or options.get("username")
            password = options.get("password")
            database = options.get("dbname") or options.get("database")
            if not all((host, user, password, database)):
                raise ValueError(
                    "Azure PostgreSQL connection string must include host, user, password, and database"
                )
            query = {}
            sslmode = options.get("sslmode") or options.get("ssl_mode")
            if sslmode:
                query["sslmode"] = sslmode.lower()
            query_string = f"?{urlencode(query)}" if query else ""
            port = options.get("port", "5432")
            return (
                f"postgresql+psycopg://{quote(user, safe='')}:{quote(password, safe='')}"
                f"@{host}:{port}/{quote(database, safe='')}{query_string}"
            )
        return configured
    user = quote(_env("MYSQL_USER", "root"), safe="")
    password = quote(_env("MYSQL_ROOT_PASSWORD", "rootpassword"), safe="")
    host = _env("MYSQL_HOST", "127.0.0.1")
    port = _env("MYSQL_PORT", "3307")
    database = _env("MYSQL_DATABASE", "capstone_db")
    return f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"


def _redis_url() -> str:
    configured = _env("REDIS_URL")
    if configured:
        return configured
    azure_connection = _env("AZURE_REDIS_CONNECTIONSTRING")
    azure_options: dict[str, str] = {}
    azure_host = _env("AZURE_REDIS_HOST")
    azure_port = _env("AZURE_REDIS_PORT")
    if azure_connection and "://" not in azure_connection:
        server, *options = azure_connection.split(",")
        host, separator, server_port = server.strip().rpartition(":")
        if separator:
            azure_host = azure_host or host
            azure_port = azure_port or server_port
        for option in options:
            key, separator, value = option.partition("=")
            if separator:
                azure_options[key.strip().lower()] = value.strip()
    elif azure_connection:
        return azure_connection

    username = _env_first("REDIS_USERNAME", "AZURE_REDIS_USERNAME")
    password = _env_first(
        "REDIS_PASSWORD", "AZURE_REDIS_ACCESS_KEY",
        default=azure_options.get("password", azure_options.get("accesskey", "")),
    )
    auth = ""
    if username:
        auth = f"{quote(username, safe='')}:{quote(password, safe='')}@"
    elif password:
        auth = f":{quote(password, safe='')}@"
    use_tls = _env_first(
        "REDIS_SSL", "AZURE_REDIS_SSL", default=azure_options.get("ssl", "")
    ).lower() in {"1", "true", "yes"}
    scheme = "rediss" if use_tls else "redis"
    default_port = "6380" if use_tls else "6379"
    host = _env_first("REDIS_HOST", default=azure_host or "127.0.0.1")
    port = _env_first("REDIS_PORT", default=azure_port or default_port)
    database = _env("REDIS_DB", "0")
    return f"{scheme}://{auth}{host}:{port}/{database}"


def _storage_dir() -> Path:
    configured = _env("STORAGE_DIR")
    if not configured:
        return _BASE_DIR / "storage"
    path = Path(configured).expanduser()
    return (path if path.is_absolute() else _BASE_DIR / path).resolve()


def _cors_origins() -> list[str]:
    configured = [
        part.strip() for part in _env("CORS_ORIGINS").split(",") if part.strip()
    ]
    return configured or ["http://localhost:5173", "http://127.0.0.1:5173"]


# 사용할 백본 → 모델 코드 폴더. 폴더마다 config.py / hitandrun_model.py 가
# 같은 이름으로 들어 있고, prediction_job 이 sys.path 에 얹어 임포트한다.
# ⚠️ 모듈명(config, hitandrun_model)이 폴더마다 동일하므로 한 프로세스에는
#    한 백본만 올릴 수 있다. 다른 백본을 쓰려면 워커를 따로 띄운다.
# 서비스 모델은 model/(x3d). "s3d model/" 은 git 에서 제외된 로컬 비교용이라
# 그 폴더가 있는 PC 에서만 MODEL_VARIANT=s3d 로 쓸 수 있다.
MODEL_VARIANT_DIRS = {
    "x3d": "model",
    "s3d": "s3d model",
    "slowfast": "slowfast model",
}


class Settings:
    # capstone-26/backend/app/settings.py → parents[2] == capstone-26
    BASE_DIR: Path = _BASE_DIR

    # MODEL_VARIANT=s3d|x3d|slowfast (기본 x3d)
    MODEL_VARIANT: str = _env("MODEL_VARIANT", "x3d").lower()
    if MODEL_VARIANT not in MODEL_VARIANT_DIRS:
        raise ValueError(
            f"MODEL_VARIANT='{MODEL_VARIANT}' 는 지원하지 않습니다. "
            f"가능한 값: {', '.join(MODEL_VARIANT_DIRS)}")
    MODEL_DIR: Path = BASE_DIR / MODEL_VARIANT_DIRS[MODEL_VARIANT]
    # 배포 가중치 경로는 각 폴더의 config.py 의 SERVICE_WEIGHTS_PATH 에서 관리한다.

    STORAGE_DIR: Path = _storage_dir()
    UPLOAD_DIR: Path = STORAGE_DIR / "uploads"
    CLIP_DIR: Path = STORAGE_DIR / "clips"
    THUMBNAIL_DIR: Path = STORAGE_DIR / "thumbnails"

    DATABASE_URL: str = _database_url()
    REDIS_URL: str = _redis_url()

    # Existing DB rows store object keys (for example uploads/<uuid>.mp4).
    # Keep the former R2/S3 configuration so those videos remain playable.
    STORAGE_BACKEND: str = _env("STORAGE_BACKEND", "local").lower()
    S3_ENDPOINT_URL: str | None = _env_first(
        "S3_ENDPOINT_URL", "R2_ENDPOINT_URL",
        default=(f"https://{_env('CLOUDFLARE_R2_ACCOUNT_ID')}.r2.cloudflarestorage.com"
                 if _env("CLOUDFLARE_R2_ACCOUNT_ID") else ""),
    ) or None
    S3_BUCKET: str | None = _env_first("S3_BUCKET", "R2_BUCKET", "CLOUDFLARE_R2_BUCKET") or None
    S3_ACCESS_KEY_ID: str | None = _env_first(
        "S3_ACCESS_KEY_ID", "R2_ACCESS_KEY_ID", "CLOUDFLARE_R2_ACCESS_KEY_ID") or None
    S3_SECRET_ACCESS_KEY: str | None = _env_first(
        "S3_SECRET_ACCESS_KEY", "R2_SECRET_ACCESS_KEY", "CLOUDFLARE_R2_SECRET_ACCESS_KEY") or None
    S3_REGION: str = _env("S3_REGION", "auto")
    S3_PRESIGNED_URL_EXPIRES: int = _env_int("S3_PRESIGNED_URL_EXPIRES", 3600)

    # 로그인 토큰(JWT) 서명 키 · 유효 시간.
    # 키는 환경변수 AUTH_SECRET_KEY 를 우선 쓰고, 없으면 backend/.auth_secret 에 한 번 만들어
    # 두고 계속 쓴다(재시작해도 로그인이 유지되도록). 이 파일은 git 에서 제외한다.
    # 기본 8시간(하루 근무) — CCTV 영상(개인정보)을 다루므로 로그인 유지 시간을 짧게 둔다.
    # 개발할 때 자주 다시 로그인하기 싫으면: AUTH_TOKEN_TTL_HOURS=72 ./dev.sh
    AUTH_TOKEN_TTL_HOURS: float = _env_float("AUTH_TOKEN_TTL_HOURS", 8)

    # 업로드한 지 이만큼 지난 영상·클립·분석 이력을 정리한다. 0 이하면 비활성화.
    VIDEO_RETENTION_DAYS: int = _env_int("VIDEO_RETENTION_DAYS", 90)
    # 탐지 디버그 이미지는 원본 영상 보관 기간과 다르게 남을 수 있으므로 기본 비활성화.
    DETECT_DEBUG_IMAGES: bool = _env("DETECT_DEBUG_IMAGES").lower() in ("1", "true", "yes")

    # CORS 허용 오리진 (Vite 개발 서버)
    CORS_ORIGINS = _cors_origins()

    def abs_path(self, rel_path: str) -> Path:
        """storage 기준 상대경로 → 절대경로"""
        return self.STORAGE_DIR / rel_path

    def rel_path(self, abs_path) -> str:
        """절대경로 → storage 기준 상대경로 (DB 저장용)"""
        return str(Path(abs_path).resolve().relative_to(self.STORAGE_DIR))


@functools.lru_cache(maxsize=1)
def auth_secret_key() -> str:
    """JWT 서명 키. 토큰을 쓰는 웹 서버가 처음 필요할 때만 읽거나 만든다
    (워커가 import 만으로 키 파일을 만들지 않게 — 동시에 뜨면 키가 갈릴 수 있다)."""
    key = os.getenv("AUTH_SECRET_KEY", "").strip()
    if key:
        return key
    path = Path(__file__).resolve().parents[1] / ".auth_secret"
    if path.exists():
        saved_key = path.read_text(encoding="utf-8").strip()
        if saved_key:
            return saved_key
    import secrets
    key = secrets.token_hex(32)
    path.write_text(key, encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return key


settings = Settings()

# 저장소 폴더 보장
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.CLIP_DIR.mkdir(parents=True, exist_ok=True)
settings.THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)
