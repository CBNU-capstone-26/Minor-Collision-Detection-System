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


engine = create_engine(normalize_database_url(settings.DATABASE_URL), pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI 의존성: 요청 단위 DB 세션."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
