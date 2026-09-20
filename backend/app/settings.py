"""애플리케이션 설정 — DB/Redis 접속 정보, 로컬 저장소 경로.

GPU·대용량 스토리지 서버가 없는 환경을 가정한다.
영상/클립은 프로젝트 루트의 storage/ 폴더(로컬 파일시스템)에 저장한다.
"""
import os
from pathlib import Path


# 사용할 백본 → 모델 코드 폴더. 폴더마다 config.py / hitandrun_model.py 가
# 같은 이름으로 들어 있고, prediction_job 이 sys.path 에 얹어 임포트한다.
# ⚠️ 모듈명(config, hitandrun_model)이 세 폴더에서 동일하므로 한 프로세스에는
#    한 백본만 올릴 수 있다. 다른 백본을 쓰려면 워커를 따로 띄운다.
MODEL_VARIANT_DIRS = {
    "s3d": "model",
    "x3d": "x3d model",
    "slowfast": "slowfast model",
}


class Settings:
    # capstone-26/backend/app/settings.py → parents[2] == capstone-26
    BASE_DIR: Path = Path(__file__).resolve().parents[2]

    # MODEL_VARIANT=s3d|x3d|slowfast (기본 s3d)
    MODEL_VARIANT: str = os.getenv("MODEL_VARIANT", "s3d").strip().lower()
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
        "mysql+pymysql://root:rootpassword@127.0.0.1:3306/capstone_db",
    )
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")

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
