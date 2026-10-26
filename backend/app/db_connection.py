"""SQLAlchemy 엔진 / 세션 / Base / get_db 의존성."""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from app.settings import settings


def normalize_database_url(url: str) -> str:
    """SQLAlchemy가 Supabase PostgreSQL URL을 psycopg로 열도록 정규화한다."""
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url.removeprefix("postgres://")
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url.removeprefix("postgresql://")
    return url


def create_database_engine(url: str):
    """Create an engine that does not keep unusable cloud DB connections."""
    database_url = normalize_database_url(url)
    connect_args = {"connect_timeout": 5}
    if database_url.startswith("postgresql+"):
        # A simple list query must not occupy a worker forever when the
        # managed PostgreSQL connection is unhealthy.
        connect_args["options"] = "-c statement_timeout=5000"
    return create_engine(
        database_url,
        pool_pre_ping=True,
        pool_recycle=300,
        pool_timeout=5,
        connect_args=connect_args,
    )


engine = create_database_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI 의존성: 요청 단위 DB 세션."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
