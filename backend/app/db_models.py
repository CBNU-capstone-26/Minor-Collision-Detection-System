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
    phone = Column(String(50), nullable=True)      # 연락처
    role = Column(String(50), default="USER")
    created_at = Column(DateTime, default=datetime.utcnow)

    videos = relationship("Video", back_populates="owner")


class Video(Base):
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    video_name = Column(String(255), nullable=False)   # 원본 파일명
    title = Column(String(100), nullable=True)         # 업로드할 때 넣은 표시 제목
    video_path = Column(String(500), nullable=False)   # storage 기준 상대경로
    recording_date = Column(Date, nullable=True)       # 업로드 시 지정한 녹화일자
    camera_location = Column(String(255), nullable=False, default="주차장")   # 녹화 CCTV 위치

    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    fps = Column(Float, nullable=False)
    total_frames = Column(Integer, nullable=False)
    detected_vehicles = Column(Text, nullable=True)  # YOLO 감지 BBOX JSON 매핑 정보
    created_at = Column(DateTime, default=datetime.utcnow)


    owner = relationship("User", back_populates="videos")
    analysis_tasks = relationship("AnalysisTask", back_populates="video")
    vehicle_detection_tasks = relationship("VehicleDetectionTask", back_populates="video")
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

    status = Column(String(50), default="PENDING")  # PENDING/PROCESSING/SUCCESS/FAILURE/CANCELLED
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    finished_at = Column(DateTime, nullable=True)
    event_count = Column(Integer, nullable=True)

    video = relationship("Video", back_populates="analysis_tasks")
    crash_events = relationship("CrashEvent", back_populates="task")


class VehicleDetectionTask(Base):
    """GPU 워커가 실행하는 프레임별 차량 탐지 작업."""
    __tablename__ = "vehicle_detection_tasks"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False)
    celery_task_id = Column(String(255), unique=True, index=True, nullable=True)
    timestamp_sec = Column(Float, nullable=False, default=0.0)
    status = Column(String(50), nullable=False, default="PENDING")
    error_message = Column(Text, nullable=True)
    result_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    finished_at = Column(DateTime, nullable=True)

    video = relationship("Video", back_populates="vehicle_detection_tasks")


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
    # DB 컬럼명은 기존 cam_heatmap_path 를 유지하고, 코드에서는 실제 역할인 clip_path 로 쓴다.
    clip_path = Column("cam_heatmap_path", String(500), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    task = relationship("AnalysisTask", back_populates="crash_events")
    video = relationship("Video", back_populates="crash_events")
