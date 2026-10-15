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
    phone: Optional[str] = None
    role: str


class UpdateUserRequest(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


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
    has_raw_clip: bool = False   # 합성 없는 원본 사고 클립을 내려받을 수 있는지


class ActiveTaskOut(BaseModel):
    """로그인한 사용자의 대기·진행 중 분석 — 창을 닫았다 다시 열어도 화면이 이어서 추적한다."""
    task_id: int
    video_id: int
    video_name: str
    recording_date: Optional[date] = None
    camera_location: str = "주차장"
    status: str
    progress: Optional[int] = None
    eta_sec: Optional[int] = None
    eta_is_min: bool = False
    created_at: Optional[datetime] = None


# ---------- 차량 자동 탐지 ----------
class DetectedVehicleBox(BaseModel):
    id: int
    class_name: str
    confidence: float
    bbox: list[int]  # [xmin, ymin, xmax, ymax]


class VehicleDetectionResponse(BaseModel):
    video_id: int
    total_detected: int
    detected_vehicles: list[DetectedVehicleBox] = []


# ---------- 영상 ----------
class LastTaskOut(BaseModel):
    """영상의 가장 최근 분석 작업 — 실패·중단을 화면에서 알리고 '다시 실행'하기 위함."""
    task_id: int
    status: str
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None


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
    detected_vehicles: Optional[str] = None
    created_at: datetime
    events: list[EventOut] = []
    last_task: Optional[LastTaskOut] = None
    analysis_done: bool = False   # 분석을 한 번이라도 완료(SUCCESS)했는지



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
    # 같은 영상에 이미 대기·진행 중인 분석이 있으면 새로 만들지 않고 그 작업을 돌려준다
    already_running: bool = False


class TaskStatusOut(BaseModel):
    task_id: int
    status: str
    progress: Optional[int] = None   # 진행도 %(PROCESSING 중 추론 진행률, 없으면 None)
    eta_sec: Optional[int] = None    # 남은 시간(초) — 실제로 잰 처리 속도 기준, 모르면 None
    eta_is_min: bool = False         # True 면 '최소 이만큼'(거친 탐색 중이라 더 늘 수 있음)
    error_message: Optional[str] = None
    events: list[EventOut] = []
