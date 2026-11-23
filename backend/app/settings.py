"""애플리케이션 설정입니다. DB와 Redis 접속 정보, 저장소 경로를 담습니다.

서버 한 대에 웹, 워커, DB, Redis, 저장소를 함께 두는 구성을 가정합니다.
영상과 클립은 프로젝트 루트의 storage/ 폴더에 저장합니다.
배포 서버에서는 프로젝트 루트의 .env로 값을 바꿉니다(deploy/env.example 참고). 값을 주지 않으면 개발용 기본값을 씁니다.
"""
import functools
import os
from pathlib import Path
from urllib.parse import quote


def _env_path(name: str, default: Path, base: Path) -> Path:
    """환경변수로 받은 경로를 돌려줍니다. 상대경로는 base 기준이고, 값이 없으면 default를 씁니다."""
    value = os.getenv(name, "").strip()
    if not value:
        return default
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


# 백본 이름과 모델 코드 폴더의 대응입니다. 폴더마다 config.py와 hitandrun_model.py가 같은 이름으로 있고,
# prediction_job이 sys.path에 넣어 불러옵니다. 모듈 이름이 같아서 한 프로세스에는 백본 하나만 올릴 수 있습니다.
# 서비스 모델은 model/(x3d)이고, 다른 폴더는 git에서 제외된 비교용이라 그 폴더가 있는 PC에서만 쓸 수 있습니다.
MODEL_VARIANT_DIRS = {
    "x3d": "model",
    "s3d": "s3d model",
    "slowfast": "slowfast model",
}


class Settings:
    # 프로젝트 루트(capstone-26)입니다.
    BASE_DIR: Path = Path(__file__).resolve().parents[2]

    # 기본값은 x3d입니다.
    MODEL_VARIANT: str = os.getenv("MODEL_VARIANT", "x3d").strip().lower()
    if MODEL_VARIANT not in MODEL_VARIANT_DIRS:
        raise ValueError(
            f"MODEL_VARIANT='{MODEL_VARIANT}' 는 지원하지 않습니다. "
            f"가능한 값: {', '.join(MODEL_VARIANT_DIRS)}")
    MODEL_DIR: Path = BASE_DIR / MODEL_VARIANT_DIRS[MODEL_VARIANT]
    # 서비스 가중치 경로는 모델 폴더의 config.py(SERVICE_WEIGHTS_PATH)에서 관리합니다.

    # 영상, 클립, 썸네일을 저장하는 폴더입니다. 영상용 디스크를 따로 붙였다면 STORAGE_DIR=/data/storage처럼 지정합니다.
    STORAGE_DIR: Path = _env_path("STORAGE_DIR", BASE_DIR / "storage", BASE_DIR)
    UPLOAD_DIR: Path = STORAGE_DIR / "uploads"
    CLIP_DIR: Path = STORAGE_DIR / "clips"
    THUMBNAIL_DIR: Path = STORAGE_DIR / "thumbnails"

    # DB와 Redis 접속 주소입니다. 주소 전체(DATABASE_URL, REDIS_URL)를 주거나,
    # 비밀번호(MYSQL_ROOT_PASSWORD, REDIS_PASSWORD)만 주면 이 서버 안의 MySQL과 Redis 주소를 만듭니다.
    DATABASE_URL: str = os.getenv("DATABASE_URL", "").strip() or (
        "mysql+pymysql://root:"
        + quote(os.getenv("MYSQL_ROOT_PASSWORD", "").strip() or "rootpassword", safe="")
        + "@127.0.0.1:3306/capstone_db")
    REDIS_URL: str = os.getenv("REDIS_URL", "").strip() or (
        f"redis://:{quote(os.getenv('REDIS_PASSWORD', '').strip(), safe='')}@127.0.0.1:6379/0"
        if os.getenv("REDIS_PASSWORD", "").strip() else "redis://127.0.0.1:6379/0")

    # 로그인 토큰(JWT)의 서명 키와 유효 시간입니다.
    # 키는 환경변수 AUTH_SECRET_KEY를 먼저 쓰고, 없으면 backend/.auth_secret에 한 번 만들어 두고 계속 씁니다.
    # 이렇게 하면 재시작해도 로그인이 유지됩니다. 이 파일은 git에서 제외합니다.
    # CCTV 영상(개인정보)을 다루므로 유효 시간은 기본 8시간으로 짧게 둡니다.
    AUTH_TOKEN_TTL_HOURS: float = float(os.getenv("AUTH_TOKEN_TTL_HOURS", "").strip() or "8")

    # 영상 보관 기간(일)입니다. 업로드한 지 이만큼 지나면 영상과 딸린 클립, 썸네일, 분석 기록을 모두 지웁니다(app/retention.py).
    # 0 이하면 지우지 않습니다.
    VIDEO_RETENTION_DAYS: int = int(os.getenv("VIDEO_RETENTION_DAYS", "").strip() or "90")

    # 차량 탐지 결과 그림을 outputs/에 남길지 정합니다(개발 확인용). 영상을 지워도 남으므로 기본은 끕니다.
    DETECT_DEBUG_IMAGES: bool = os.getenv("DETECT_DEBUG_IMAGES", "").strip().lower() in ("1", "true", "yes")

    # CORS 허용 주소입니다(기본은 Vite 개발 서버). 배포에서는 nginx가 화면과 /api를 같은 주소로 내주므로 필요 없고,
    # 화면을 다른 주소에서 띄울 때만 CORS_ORIGINS=https://a.com,https://b.com처럼 지정합니다.
    CORS_ORIGINS = [o.strip() for o in os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]

    def abs_path(self, rel_path: str) -> Path:
        """storage 기준 상대경로를 절대경로로 바꿉니다."""
        return self.STORAGE_DIR / rel_path

    def rel_path(self, abs_path) -> str:
        """절대경로를 DB에 저장할 storage 기준 상대경로로 바꿉니다."""
        return str(Path(abs_path).resolve().relative_to(self.STORAGE_DIR))


@functools.lru_cache(maxsize=1)
def auth_secret_key() -> str:
    """JWT 서명 키를 돌려줍니다. 웹 서버가 처음 필요할 때만 읽거나 만듭니다.
    워커가 불러오기만 해도 키 파일을 만들면, 동시에 뜰 때 서로 다른 키가 생길 수 있기 때문입니다.
    """
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

# 저장소 폴더를 만들어 둡니다.
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.CLIP_DIR.mkdir(parents=True, exist_ok=True)
settings.THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)
