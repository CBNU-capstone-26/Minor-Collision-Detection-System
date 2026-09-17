"""애플리케이션 설정 — DB/Redis/파일 저장소 접속 정보.

GPU·대용량 스토리지 서버가 없는 환경을 가정한다.
기본값은 로컬 파일시스템이며, 배포 시 S3 호환 저장소(R2)를 선택할 수 있다.
"""
import os
from pathlib import Path

from dotenv import load_dotenv


_BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(_BASE_DIR / "backend" / ".env")


class Settings:
    # capstone-26/backend/app/settings.py → parents[2] == capstone-26
    BASE_DIR: Path = _BASE_DIR
    MODEL_DIR: Path = BASE_DIR / "model"
    # 배포 가중치 경로는 model/config.py 의 SERVICE_WEIGHTS_PATH 에서 관리한다.

    STORAGE_DIR: Path = BASE_DIR / "storage"
    UPLOAD_DIR: Path = STORAGE_DIR / "uploads"
    CLIP_DIR: Path = STORAGE_DIR / "clips"
    THUMBNAIL_DIR: Path = STORAGE_DIR / "thumbnails"

    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "mysql+pymysql://root:rootpassword@127.0.0.1:3306/capstone_db",
    )
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")

    # 파일 저장소: local 또는 s3. s3는 Cloudflare R2 등 S3 호환 서비스에 사용한다.
    STORAGE_BACKEND: str = os.getenv("STORAGE_BACKEND", "local").lower()
    S3_ENDPOINT_URL: str | None = os.getenv("S3_ENDPOINT_URL")
    S3_BUCKET: str | None = os.getenv("S3_BUCKET")
    S3_ACCESS_KEY_ID: str | None = os.getenv("S3_ACCESS_KEY_ID")
    S3_SECRET_ACCESS_KEY: str | None = os.getenv("S3_SECRET_ACCESS_KEY")
    S3_REGION: str = os.getenv("S3_REGION", "auto")
    S3_PRESIGNED_URL_EXPIRES: int = int(os.getenv("S3_PRESIGNED_URL_EXPIRES", "3600"))

    # CORS 허용 오리진 (Vite 개발 서버)
    CORS_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]

    def abs_path(self, rel_path: str) -> Path:
        """storage 기준 상대경로 → 절대경로"""
        return self.STORAGE_DIR / rel_path

    def rel_path(self, abs_path) -> str:
        """절대경로 → storage 기준 상대경로 (DB 저장용)"""
        return str(Path(abs_path).resolve().relative_to(self.STORAGE_DIR))


settings = Settings()

# 저장소 폴더 보장
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.CLIP_DIR.mkdir(parents=True, exist_ok=True)
settings.THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)
