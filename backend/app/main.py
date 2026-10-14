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

def _wait_for_db(timeout_sec=60):
    """DB(MySQL)가 연결을 받을 때까지 기다린다.

    DB 컨테이너를 막 띄운 직후엔 MySQL 이 준비되기 전이라 첫 연결이 끊기고, 예전엔 그때
    백엔드가 바로 죽어 화면에서 로그인이 502 가 났다. 바로 죽지 않고 잠깐씩 다시 시도한다
    (어떤 OS·실행 방법으로 띄워도 같다).
    """
    import time
    from sqlalchemy import text
    from sqlalchemy.exc import OperationalError
    deadline = time.monotonic() + timeout_sec
    while True:
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return
        except OperationalError as exc:
            if time.monotonic() > deadline:
                raise
            print(f"[backend] DB 연결 대기 중(최대 {timeout_sec}초): {getattr(exc, 'orig', exc)}", flush=True)
            time.sleep(2)


# 서버 시작 시 테이블이 없으면 자동 생성 — DB 가 준비될 때까지 기다린 뒤
_wait_for_db()
Base.metadata.create_all(bind=engine)

def _ensure_column(table, column_ddl):
    """나중에 추가한 컬럼을 기존 테이블에도 넣는다 (create_all 은 이미 있는 테이블을 고치지 않는다).
    이미 있으면 MySQL 이 오류를 내므로 그냥 넘어간다."""
    from sqlalchemy import text
    with engine.connect() as conn:
        try:
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column_ddl};"))
            conn.commit()
        except Exception:
            pass


_ensure_column("videos", "detected_vehicles TEXT NULL")
_ensure_column("analysis_tasks", "finished_at DATETIME NULL")   # 분석 이력: 끝난 시각
_ensure_column("analysis_tasks", "event_count INT NULL")        # 분석 이력: 찾은 클립 수

app.include_router(auth.router)
app.include_router(videos.router)
app.include_router(analysis.router)



@app.get("/")
def read_root():
    return {"message": "물피도주 자동감지 API 가동 중"}
