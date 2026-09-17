"""FastAPI 진입점 — CORS, 라우터 등록, 시작 시 테이블 생성."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db_connection import engine, Base
from app import db_models  # noqa: F401  (테이블 등록을 위해 임포트)
from app.settings import settings
from app.routers import auth, videos, analysis

app = FastAPI(title="물피도주 자동감지 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 서버 시작 시 테이블이 없으면 자동 생성
Base.metadata.create_all(bind=engine)

# 기존 테이블에 detected_vehicles 컬럼 마이그레이션 보장
with engine.connect() as conn:
    try:
        from sqlalchemy import text
        conn.execute(text("ALTER TABLE videos ADD COLUMN detected_vehicles TEXT NULL;"))
        conn.commit()
    except Exception:
        pass

# 기존 DB에 User 모델의 연락처 컬럼이 없을 수 있으므로 보정한다.
# create_all()은 이미 존재하는 테이블에 새 컬럼을 추가하지 않는다.
with engine.connect() as conn:
    try:
        from sqlalchemy import text
        conn.execute(text("ALTER TABLE users ADD COLUMN phone VARCHAR(50) NULL;"))
        conn.commit()
    except Exception:
        # 컬럼이 이미 존재하는 경우 등에는 정상 기동을 계속한다.
        pass

app.include_router(auth.router)
app.include_router(videos.router)
app.include_router(analysis.router)



@app.get("/")
def read_root():
    return {"message": "물피도주 자동감지 API 가동 중"}
