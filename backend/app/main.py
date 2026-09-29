"""FastAPI 진입점 — CORS, 라우터 등록, 시작 시 테이블 생성."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import inspect, text

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

# create_all()은 이미 존재하는 테이블에 새 컬럼을 추가하지 않으므로,
# MySQL과 PostgreSQL(Supabase) 모두에서 필요한 최소 컬럼을 보정한다.
def ensure_column(table_name: str, column_name: str, column_ddl: str):
    columns = {column["name"] for column in inspect(engine).get_columns(table_name)}
    if column_name in columns:
        return
    with engine.begin() as conn:
        conn.execute(text(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_ddl}"
        ))


ensure_column("videos", "detected_vehicles", "TEXT NULL")
ensure_column("users", "phone", "VARCHAR(50) NULL")

app.include_router(auth.router)
app.include_router(videos.router)
app.include_router(analysis.router)



@app.get("/")
def read_root():
    return {"message": "물피도주 자동감지 API 가동 중"}
