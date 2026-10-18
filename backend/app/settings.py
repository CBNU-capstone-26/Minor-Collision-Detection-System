"""애플리케이션 설정 — DB/Redis 접속 정보, 로컬 저장소 경로.

GPU·대용량 스토리지 서버가 없는 환경을 가정한다.
영상/클립은 프로젝트 루트의 storage/ 폴더(로컬 파일시스템)에 저장한다.
"""
import functools
import os
from pathlib import Path

from dotenv import load_dotenv


# Keep this branch's database credentials in backend/.env, rather than relying
# only on the process environment supplied by the UI/backend integration.
_BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(_BASE_DIR / "backend" / ".env")


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
    MODEL_VARIANT: str = os.getenv("MODEL_VARIANT", "x3d").strip().lower()
    if MODEL_VARIANT not in MODEL_VARIANT_DIRS:
        raise ValueError(
            f"MODEL_VARIANT='{MODEL_VARIANT}' 는 지원하지 않습니다. "
            f"가능한 값: {', '.join(MODEL_VARIANT_DIRS)}")
    MODEL_DIR: Path = BASE_DIR / MODEL_VARIANT_DIRS[MODEL_VARIANT]
    # 배포 가중치 경로는 각 폴더의 config.py 의 SERVICE_WEIGHTS_PATH 에서 관리한다.

    STORAGE_DIR: Path = BASE_DIR / "storage"
    UPLOAD_DIR: Path = STORAGE_DIR / "uploads"
    CLIP_DIR: Path = STORAGE_DIR / "clips"
    THUMBNAIL_DIR: Path = STORAGE_DIR / "thumbnails"

    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "mysql+pymysql://root:rootpassword@127.0.0.1:3307/capstone_db",
    )
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")

    # Existing DB rows store object keys (for example uploads/<uuid>.mp4).
    # Keep the former R2/S3 configuration so those videos remain playable.
    STORAGE_BACKEND: str = os.getenv("STORAGE_BACKEND", "local").lower()
    S3_ENDPOINT_URL: str | None = os.getenv("S3_ENDPOINT_URL")
    S3_BUCKET: str | None = os.getenv("S3_BUCKET")
    S3_ACCESS_KEY_ID: str | None = os.getenv("S3_ACCESS_KEY_ID")
    S3_SECRET_ACCESS_KEY: str | None = os.getenv("S3_SECRET_ACCESS_KEY")
    S3_REGION: str = os.getenv("S3_REGION", "auto")
    S3_PRESIGNED_URL_EXPIRES: int = int(os.getenv("S3_PRESIGNED_URL_EXPIRES", "3600"))

    # 로그인 토큰(JWT) 서명 키 · 유효 시간.
    # 키는 환경변수 AUTH_SECRET_KEY 를 우선 쓰고, 없으면 backend/.auth_secret 에 한 번 만들어
    # 두고 계속 쓴다(재시작해도 로그인이 유지되도록). 이 파일은 git 에서 제외한다.
    # 기본 8시간(하루 근무) — CCTV 영상(개인정보)을 다루므로 로그인 유지 시간을 짧게 둔다.
    # 개발할 때 자주 다시 로그인하기 싫으면: AUTH_TOKEN_TTL_HOURS=72 ./dev.sh
    AUTH_TOKEN_TTL_HOURS: float = float(os.getenv("AUTH_TOKEN_TTL_HOURS", "8"))

    # 업로드한 지 이만큼 지난 영상·클립·분석 이력을 정리한다. 0 이하면 비활성화.
    VIDEO_RETENTION_DAYS: int = int(os.getenv("VIDEO_RETENTION_DAYS", "90"))
    # 탐지 디버그 이미지는 원본 영상 보관 기간과 다르게 남을 수 있으므로 기본 비활성화.
    DETECT_DEBUG_IMAGES: bool = os.getenv("DETECT_DEBUG_IMAGES", "").strip().lower() in ("1", "true", "yes")

    # CORS 허용 오리진 (Vite 개발 서버)
    CORS_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]

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
