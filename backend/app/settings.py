"""애플리케이션 설정 — DB/Redis 접속 정보, 로컬 저장소 경로.

서버 한 대에 웹·워커·DB·Redis·저장소를 함께 두는 구성을 가정한다.
영상/클립은 프로젝트 루트의 storage/ 폴더(로컬 파일시스템)에 저장한다.

배포 서버에선 프로젝트 루트 .env 로 값을 바꾼다(deploy/env.example 참고). 아무것도 안 주면
개발 환경 기본값 그대로다.
"""
import functools
import os
from pathlib import Path
from urllib.parse import quote


def _env_path(name: str, default: Path, base: Path) -> Path:
    """환경변수 경로(상대경로면 프로젝트 루트 기준). 없거나 비어 있으면 default."""
    value = os.getenv(name, "").strip()
    if not value:
        return default
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


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
    BASE_DIR: Path = Path(__file__).resolve().parents[2]

    # MODEL_VARIANT=s3d|x3d|slowfast (기본 x3d)
    MODEL_VARIANT: str = os.getenv("MODEL_VARIANT", "x3d").strip().lower()
    if MODEL_VARIANT not in MODEL_VARIANT_DIRS:
        raise ValueError(
            f"MODEL_VARIANT='{MODEL_VARIANT}' 는 지원하지 않습니다. "
            f"가능한 값: {', '.join(MODEL_VARIANT_DIRS)}")
    MODEL_DIR: Path = BASE_DIR / MODEL_VARIANT_DIRS[MODEL_VARIANT]
    # 배포 가중치 경로는 각 폴더의 config.py 의 SERVICE_WEIGHTS_PATH 에서 관리한다.

    # 영상·클립·썸네일 저장 폴더. 서버에서 영상용 디스크를 따로 붙였으면 STORAGE_DIR=/data/storage
    STORAGE_DIR: Path = _env_path("STORAGE_DIR", BASE_DIR / "storage", BASE_DIR)
    UPLOAD_DIR: Path = STORAGE_DIR / "uploads"
    CLIP_DIR: Path = STORAGE_DIR / "clips"
    THUMBNAIL_DIR: Path = STORAGE_DIR / "thumbnails"

    # DB·Redis 접속 주소. 주소를 통째로 주거나(DATABASE_URL·REDIS_URL — 나중에 Supabase 등),
    # 비밀번호만 주면(MYSQL_ROOT_PASSWORD·REDIS_PASSWORD — docker-compose 와 같은 .env 값)
    # 이 서버 안의 MySQL·Redis 주소를 만든다.
    DATABASE_URL: str = os.getenv("DATABASE_URL", "").strip() or (
        "mysql+pymysql://root:"
        + quote(os.getenv("MYSQL_ROOT_PASSWORD", "").strip() or "rootpassword", safe="")
        + "@127.0.0.1:3306/capstone_db")
    REDIS_URL: str = os.getenv("REDIS_URL", "").strip() or (
        f"redis://:{quote(os.getenv('REDIS_PASSWORD', '').strip(), safe='')}@127.0.0.1:6379/0"
        if os.getenv("REDIS_PASSWORD", "").strip() else "redis://127.0.0.1:6379/0")

    # 로그인 토큰(JWT) 서명 키 · 유효 시간.
    # 키는 환경변수 AUTH_SECRET_KEY 를 우선 쓰고, 없으면 backend/.auth_secret 에 한 번 만들어
    # 두고 계속 쓴다(재시작해도 로그인이 유지되도록). 이 파일은 git 에서 제외한다.
    # 기본 8시간(하루 근무) — CCTV 영상(개인정보)을 다루므로 로그인 유지 시간을 짧게 둔다.
    # 개발할 때 자주 다시 로그인하기 싫으면: AUTH_TOKEN_TTL_HOURS=72 ./dev.sh
    AUTH_TOKEN_TTL_HOURS: float = float(os.getenv("AUTH_TOKEN_TTL_HOURS", "").strip() or "8")

    # 영상 보관 기간(일) — 업로드한 지 이만큼 지나면 영상과 딸린 것(사고 클립·원본 클립·썸네일·
    # 분석 기록)을 모두 지운다(app/retention.py). 0 이하면 지우지 않는다.
    VIDEO_RETENTION_DAYS: int = int(os.getenv("VIDEO_RETENTION_DAYS", "").strip() or "90")

    # 차량 탐지 결과를 CCTV 화면 이미지로 outputs/ 에 남길지(개발 확인용). 영상을 지워도 남으므로
    # 기본은 끈다 — 필요할 때만 DETECT_DEBUG_IMAGES=1
    DETECT_DEBUG_IMAGES: bool = os.getenv("DETECT_DEBUG_IMAGES", "").strip().lower() in ("1", "true", "yes")

    # CORS 허용 오리진 (기본: Vite 개발 서버). 배포에선 nginx 가 화면과 /api 를 같은 주소로
    # 내주므로 필요 없다. 화면을 다른 주소에서 띄울 때만 CORS_ORIGINS=https://a.com,https://b.com
    CORS_ORIGINS = [o.strip() for o in os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]

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
