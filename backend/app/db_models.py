"""SQLAlchemy DB 모델.

테이블 관계: users → videos → analysis_tasks → crash_events
(crash_events 는 video_id 도 함께 보유 — 빠른 조회용 반정규화)
"""
from datetime import datetime

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Date, ForeignKey, Text,
)
from sqlalchemy.orm import relationship

from app.db_connection import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    name = Column(String(100), nullable=True)      # 회원가입: 실명
    email = Column(String(255), unique=True, nullable=True)
    role = Column(String(50), default="USER")
    created_at = Column(DateTime, default=datetime.utcnow)

    videos = relationship("Video", back_populates="owner")


class Video(Base):
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    video_name = Column(String(255), nullable=False)   # 원본 파일명
    video_path = Column(String(500), nullable=False)   # storage 기준 상대경로
    recording_date = Column(Date, nullable=True)       # 업로드 시 지정한 녹화일자
    camera_location = Column(String(255), nullable=False, default="주차장")   # 녹화 CCTV 위치
    recording_start_time = Column(String(5), nullable=False, default="20:30")  # 녹화 시작 시각 "HH:MM"(24h) — 기본값 20:30

    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    fps = Column(Float, nullable=False)
    total_frames = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    owner = relationship("User", back_populates="videos")
    analysis_tasks = relationship("AnalysisTask", back_populates="video")
    crash_events = relationship("CrashEvent", back_populates="video")


class AnalysisTask(Base):
    __tablename__ = "analysis_tasks"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False)
    celery_task_id = Column(String(255), unique=True, index=True, nullable=True)

    # 사용자가 지정한 대상 차량 bbox (원본 해상도 픽셀 좌표)
    bbox_xmin = Column(Integer, nullable=False)
    bbox_ymin = Column(Integer, nullable=False)
    bbox_xmax = Column(Integer, nullable=False)
    bbox_ymax = Column(Integer, nullable=False)

    status = Column(String(50), default="PENDING")  # PENDING/PROCESSING/SUCCESS/FAILURE
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    video = relationship("Video", back_populates="analysis_tasks")
    crash_events = relationship("CrashEvent", back_populates="task")


class CrashEvent(Base):
    """사고 의심 구간(이벤트) 1건. 구간별 CAM 클립 1개."""
    __tablename__ = "crash_events"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("analysis_tasks.id"), nullable=False)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False)

    # 구간 시작
    timestamp_sec = Column(Float, nullable=False)
    frame_number = Column(Integer, nullable=False)
    # 구간 종료 (클립 길이/구간 표시용)
    end_timestamp_sec = Column(Float, nullable=True)
    end_frame_number = Column(Integer, nullable=True)

    crash_prob = Column(Float, nullable=True)          # 구간 대표 확률 (0~1)
    cam_heatmap_path = Column(String(500), nullable=True)  # 사고구간 CAM 클립 경로(storage 상대경로)

    created_at = Column(DateTime, default=datetime.utcnow)

    task = relationship("AnalysisTask", back_populates="crash_events")
    video = relationship("Video", back_populates="crash_events")
