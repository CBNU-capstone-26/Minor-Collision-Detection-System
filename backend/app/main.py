"""FastAPI 진입점 — CORS, 라우터 등록, 시작 시 테이블 생성, 보관 기간 지난 영상 정리."""
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import inspect, text

from app.db_connection import engine, Base
from app import db_models  # noqa: F401  (테이블 등록을 위해 임포트)
from app.schema_init import create_tables_safely
from app.settings import settings
from app.routers import auth, videos, analysis


async def _retention_loop():
    """보관 기간(VIDEO_RETENTION_DAYS)이 지난 영상을 서버가 켜질 때 한 번, 그 뒤 1시간마다 지운다."""
    from app.retention import CHECK_INTERVAL_SEC, delete_expired_videos
    while True:
        try:
            deleted = await asyncio.to_thread(delete_expired_videos)
            if deleted:
                print(f"[보관기간] {settings.VIDEO_RETENTION_DAYS}일 지난 영상 {len(deleted)}개를 지웠습니다: {deleted}", flush=True)
        except Exception as exc:  # noqa: BLE001 — 정리에 실패해도 서버는 계속 돈다(다음 회차에 다시)
            print(f"[보관기간] 정리 실패(다음에 다시): {exc}", flush=True)
        await asyncio.sleep(CHECK_INTERVAL_SEC)


@asynccontextmanager
async def lifespan(_app):
    task = asyncio.create_task(_retention_loop())
    yield
    task.cancel()


app = FastAPI(title="물피도주 자동감지 API", lifespan=lifespan)

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


# DB가 준비되면 없는 테이블을 만듭니다.
_wait_for_db()
create_tables_safely(engine, Base.metadata)

def _ensure_column(table, column, column_type):
    """나중에 추가한 컬럼을 기존 테이블에도 넣는다 (create_all 은 이미 있는 테이블을 고치지 않는다).
    이미 있는 컬럼은 건너뛰고, 마이그레이션 오류는 서버 시작 오류로 드러낸다."""
    if column in {item["name"] for item in inspect(engine).get_columns(table)}:
        return
    with engine.begin() as conn:
        conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {column_type};"))


_ensure_column("videos", "detected_vehicles", "TEXT NULL")
_ensure_column("videos", "title", "VARCHAR(100) NULL")             # 업로드할 때 넣은 제목
_ensure_column("videos", "video_type", "VARCHAR(20) NULL")
_ensure_column("videos", "collision_label", "VARCHAR(20) NULL")
# 모델과 같은 DateTime을 각 DB 방언에 맞게 생성한다(MySQL DATETIME, PostgreSQL TIMESTAMP).
_finished_at_type = db_models.AnalysisTask.__table__.c.finished_at.type.compile(dialect=engine.dialect)
_ensure_column("analysis_tasks", "finished_at", f"{_finished_at_type} NULL")
_ensure_column("analysis_tasks", "event_count", "INT NULL")        # 분석 이력: 찾은 클립 수
_ensure_column("crash_events", "clip_start_sec", "FLOAT NULL")    # CAM 클립이 원본의 몇 초부터인지


def _drop_column(table, column):
    """더 이상 쓰지 않는 컬럼을 기존 테이블에서 지운다.
    이미 없으면 건너뛰고, 마이그레이션 오류는 서버 시작 오류로 드러낸다."""
    if column not in {item["name"] for item in inspect(engine).get_columns(table)}:
        return
    with engine.begin() as conn:
        conn.execute(text(f"ALTER TABLE {table} DROP COLUMN {column};"))


# 쓰지 않는 녹화 시작 시각 컬럼을 지웁니다.
_drop_column("videos", "recording_start_time")

app.include_router(auth.router)
app.include_router(videos.router)
app.include_router(analysis.router)



@app.get("/")
def read_root():
    return {"message": "물피도주 자동감지 API 가동 중"}
