"""SQLAlchemy 엔진, 세션, Base, get_db 의존성을 정의합니다."""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from app.settings import settings

engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """요청마다 DB 세션을 열고 닫는 FastAPI 의존성입니다."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
