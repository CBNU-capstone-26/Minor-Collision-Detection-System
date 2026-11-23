"""Pydantic 요청, 응답 스키마입니다."""
from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------- 인증 ----------
class SignupRequest(BaseModel):
    username: str
    name: str = Field(min_length=1)   # 이름은 필수입니다.
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
    analyzed_at: Optional[datetime] = None   # 분석이 끝나 이 클립이 만들어진 시각(UTC, 시간대 포함)
    analysis_no: Optional[int] = None        # 이 클립을 만든 분석이 그 영상의 몇 번째 분석인지(실패, 취소 포함, 1부터)
    clip_start_sec: Optional[float] = None   # 클립이 원본 영상의 몇 초부터인지(값이 없는 예전 분석은 None)


class ActiveTaskOut(BaseModel):
    """로그인한 사용자의 대기 중이거나 진행 중인 분석입니다. 브라우저를 닫았다 다시 열어도 화면이 이어서 추적합니다."""
    task_id: int
    video_id: int
    video_name: str
    recording_date: Optional[date] = None
    camera_location: str = "주차장"
    status: str
    progress: Optional[float] = None
    eta_sec: Optional[int] = None
    eta_is_min: bool = False
    stage: Optional[str] = None        # 지금 단계(TaskStatusOut 참고)
    stage_count: Optional[int] = None
    stage_index: Optional[int] = None
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
    """영상의 가장 최근 분석 작업입니다. 실패나 중단을 화면에서 알리고 '다시 실행'할 때 씁니다."""
    task_id: int
    status: str
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None


class AnalysisHistoryOut(BaseModel):
    """영상의 분석 한 번입니다(재생 화면 오른쪽 '분석 이력'). 시각은 모두 UTC(시간대 포함)입니다."""
    analysis_no: int                       # 이 영상의 몇 번째 분석인지(실패, 취소 포함, 1부터)
    task_id: int
    status: str                            # PENDING, PROCESSING, SUCCESS, FAILURE, CANCELLED
    error_message: Optional[str] = None
    requested_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None  # 이력 기능 이전 분석은 클립이 만들어진 시각으로 대신합니다(없으면 None).
    event_count: Optional[int] = None       # 찾은 클립 수(이력 기능 이전 분석은 None)


class VideoOut(BaseModel):
    id: int
    video_name: str
    title: Optional[str] = None   # 업로드할 때 입력한 제목(없으면 None)
    recording_date: Optional[date] = None
    camera_location: str = "주차장"
    width: int
    height: int
    fps: float
    total_frames: int
    duration_sec: float
    detected_vehicles: Optional[str] = None
    created_at: datetime
    # 보관 기간이 끝나 지워질 시각과 남은 날입니다(서버 시계 기준, 올림). 보관 기간을 끄면 None입니다.
    expires_at: Optional[datetime] = None
    expires_in_days: Optional[int] = None
    events: list[EventOut] = []
    last_task: Optional[LastTaskOut] = None
    analysis_done: bool = False   # 분석을 한 번이라도 완료(SUCCESS)했는지
    analysis_history: list[AnalysisHistoryOut] = []   # 영상 상세(GET /videos/{id})에서만 채웁니다.



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
    # 같은 영상에 이미 대기 중이거나 진행 중인 분석이 있으면 새로 만들지 않고 그 작업을 돌려줍니다.
    already_running: bool = False


class TaskStatusOut(BaseModel):
    task_id: int
    status: str
    progress: Optional[float] = None  # 진행률 %(소수 첫째 자리, PROCESSING일 때만)
    eta_sec: Optional[int] = None    # 지금 단계의 남은 시간(초)입니다. 실제로 잰 처리 속도로 계산하고, 모르면 None입니다.
    eta_is_min: bool = False         # True면 '최소 이만큼'이라는 뜻입니다(지금은 쓰지 않지만 이전 워커와 호환하려고 둡니다).
    # 지금 단계: loading(분석 엔진 준비), decode(영상 불러오기), motion(차량 움직임 계산),
    # coarse(영상 전체 훑어보기), refine(의심 구간 자세히 보기), scan(전부 보기),
    # events(사고 구간 정리), render(사고 클립 만들기)
    stage: Optional[str] = None
    stage_count: Optional[int] = None  # coarse는 찾은 의심 지점 수, refine은 의심 구간 수, render는 클립 수
    stage_index: Optional[int] = None  # refine, render에서 지금 몇 번째인지(1부터)
    error_message: Optional[str] = None
    events: list[EventOut] = []
