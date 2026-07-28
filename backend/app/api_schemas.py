"""Pydantic 요청/응답 스키마."""
from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------- 인증 ----------
class SignupRequest(BaseModel):
    username: str
    name: str = Field(min_length=1)   # 이름 필수
    email: Optional[str] = None
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    username: str
    name: Optional[str] = None
    email: Optional[str] = None
    role: str


class LoginResponse(BaseModel):
    token: str
    user: UserOut


# ---------- 이벤트(사고 구간) ----------
class EventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    timestamp_sec: float
    frame_number: int
    end_timestamp_sec: Optional[float] = None
    end_frame_number: Optional[int] = None
    crash_prob: Optional[float] = None
    has_clip: bool = False


# ---------- 영상 ----------
class VideoOut(BaseModel):
    id: int
    video_name: str
    recording_date: Optional[date] = None
    camera_location: str = "주차장"
    recording_start_time: str = "20:30"
    width: int
    height: int
    fps: float
    total_frames: int
    duration_sec: float
    created_at: datetime
    events: list[EventOut] = []


# ---------- 분석 ----------
class AnalyzeRequest(BaseModel):
    bbox_xmin: int
    bbox_ymin: int
    bbox_xmax: int
    bbox_ymax: int


class AnalyzeResponse(BaseModel):
    task_id: int
    celery_task_id: Optional[str] = None
    status: str


class TaskStatusOut(BaseModel):
    task_id: int
    status: str
    error_message: Optional[str] = None
    events: list[EventOut] = []
