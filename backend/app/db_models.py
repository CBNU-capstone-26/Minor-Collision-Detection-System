"""SQLAlchemy DB 모델입니다.

테이블 관계는 users, videos, analysis_tasks, crash_events 순입니다.
crash_events는 빠른 조회를 위해 video_id도 함께 가집니다.
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
    name = Column(String(100), nullable=True)      # 실명
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
    title = Column(String(100), nullable=True)         # 업로드할 때 입력한 제목(없으면 화면에 '녹화일 녹화본 #N'으로 표시)
    video_path = Column(String(500), nullable=False)   # storage 기준 상대경로
    recording_date = Column(Date, nullable=True)       # 업로드할 때 지정한 녹화일
    camera_location = Column(String(255), nullable=False, default="주차장")   # 녹화한 CCTV 위치

    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    fps = Column(Float, nullable=False)
    total_frames = Column(Integer, nullable=False)
    detected_vehicles = Column(Text, nullable=True)  # YOLO로 찾은 차량 박스(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)


    owner = relationship("User", back_populates="videos")
    analysis_tasks = relationship("AnalysisTask", back_populates="video")
    crash_events = relationship("CrashEvent", back_populates="video")


class AnalysisTask(Base):
    __tablename__ = "analysis_tasks"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False)
    celery_task_id = Column(String(255), unique=True, index=True, nullable=True)

    # 사용자가 지정한 대상 차량 박스(원본 해상도 픽셀 좌표)
    bbox_xmin = Column(Integer, nullable=False)
    bbox_ymin = Column(Integer, nullable=False)
    bbox_xmax = Column(Integer, nullable=False)
    bbox_ymax = Column(Integer, nullable=False)

    status = Column(String(50), default="PENDING")  # PENDING, PROCESSING, SUCCESS, FAILURE, CANCELLED
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)   # 분석을 요청한 시각(UTC)
    # 분석 이력용 컬럼입니다. 이 컬럼을 넣기 전의 분석은 NULL입니다.
    finished_at = Column(DateTime, nullable=True)   # 끝난 시각(UTC). 성공, 실패, 취소 모두 기록합니다.
    event_count = Column(Integer, nullable=True)    # 분석이 찾은 사고 의심 클립 수(성공했을 때)

    video = relationship("Video", back_populates="analysis_tasks")
    crash_events = relationship("CrashEvent", back_populates="task")


class CrashEvent(Base):
    """사고 의심 구간 하나입니다. 구간마다 CAM 클립이 하나씩 있습니다."""
    __tablename__ = "crash_events"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("analysis_tasks.id"), nullable=False)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False)

    # 구간 시작
    timestamp_sec = Column(Float, nullable=False)
    frame_number = Column(Integer, nullable=False)
    # 구간 끝
    end_timestamp_sec = Column(Float, nullable=True)
    end_frame_number = Column(Integer, nullable=True)

    crash_prob = Column(Float, nullable=True)          # 구간 대표 확률(0~1)
    # 사고 구간 클립 경로(storage 상대경로)입니다. 테이블을 바꾸지 않으려고 DB 컬럼 이름은 cam_heatmap_path를 그대로 씁니다.
    clip_path = Column("cam_heatmap_path", String(500), nullable=True)
    # 클립(구간 앞뒤 여유 포함)이 원본 영상의 몇 초부터인지입니다. 재생 화면이 원본 시간에 맞춰 클립을 겹쳐 보여 줄 때 씁니다.
    # 이 값이 없는 예전 분석은 화면에 원본만 보여 줍니다.
    clip_start_sec = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    task = relationship("AnalysisTask", back_populates="crash_events")
    video = relationship("Video", back_populates="crash_events")
