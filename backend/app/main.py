"""FastAPI 진입점입니다. CORS와 라우터를 등록하고, 시작할 때 테이블을 만들고, 보관 기간이 지난 영상을 정리합니다."""
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db_connection import engine, Base
from app import db_models  # noqa: F401  테이블을 등록하려고 불러옵니다.
from app.settings import settings
from app.routers import auth, videos, analysis


async def _retention_loop():
    """보관 기간(VIDEO_RETENTION_DAYS)이 지난 영상을 서버가 켜질 때 한 번 지우고, 그 뒤로는 1시간마다 지웁니다."""
    from app.retention import CHECK_INTERVAL_SEC, delete_expired_videos
    while True:
        try:
            deleted = await asyncio.to_thread(delete_expired_videos)
            if deleted:
                print(f"[보관기간] {settings.VIDEO_RETENTION_DAYS}일 지난 영상 {len(deleted)}개를 지웠습니다: {deleted}", flush=True)
        except Exception as exc:  # noqa: BLE001  정리에 실패해도 서버는 계속 돌고, 다음 회차에 다시 시도합니다.
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
    """DB가 연결을 받을 때까지 기다립니다.

    DB 컨테이너를 막 띄운 직후에는 MySQL이 아직 준비되지 않아 첫 연결이 끊길 수 있습니다.
    이때 백엔드가 바로 종료되지 않도록 잠깐씩 다시 시도합니다.
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
Base.metadata.create_all(bind=engine)

def _ensure_column(table, column_ddl):
    """나중에 추가한 컬럼을 기존 테이블에도 넣습니다. create_all은 이미 있는 테이블을 고치지 않기 때문입니다.
    컬럼이 이미 있으면 MySQL이 오류를 내므로 그냥 넘어갑니다.
    """
    from sqlalchemy import text
    with engine.connect() as conn:
        try:
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column_ddl};"))
            conn.commit()
        except Exception:
            pass


_ensure_column("videos", "detected_vehicles TEXT NULL")
_ensure_column("videos", "title VARCHAR(100) NULL")             # 업로드할 때 입력한 제목
_ensure_column("analysis_tasks", "finished_at DATETIME NULL")   # 분석이 끝난 시각
_ensure_column("analysis_tasks", "event_count INT NULL")        # 분석이 찾은 사고 의심 클립 수
_ensure_column("crash_events", "clip_start_sec FLOAT NULL")     # 클립이 원본 영상의 몇 초부터인지(재생 화면에서 클립을 겹쳐 보일 때 씁니다)


def _drop_column(table, column):
    """더 이상 쓰지 않는 컬럼을 기존 테이블에서 지웁니다.
    컬럼이 이미 없으면 MySQL이 오류를 내므로 그냥 넘어갑니다.
    """
    from sqlalchemy import text
    with engine.connect() as conn:
        try:
            conn.execute(text(f"ALTER TABLE {table} DROP COLUMN {column};"))
            conn.commit()
        except Exception:
            pass


# 쓰지 않는 녹화 시작 시각 컬럼을 지웁니다.
_drop_column("videos", "recording_start_time")

app.include_router(auth.router)
app.include_router(videos.router)
app.include_router(analysis.router)



@app.get("/")
def read_root():
    return {"message": "물피도주 자동감지 API 가동 중"}
