import os
from pathlib import Path

# ==========================================
# [파라미터 사용자 설정]
# 실행 환경에 맞게 이 파일의 값만 수정하세요.
# ==========================================

# 프로젝트 루트 기준 경로 (model/ 의 상위 폴더)
_ROOT = Path(__file__).resolve().parent.parent

# ---------- 디바이스 설정 ----------
# "cuda" : NVIDIA GPU (CUDA 빌드)
#          또는 AMD GPU (네이티브 Linux + ROCm 빌드)
# "cpu"  : CPU
#
# 학습(train)과 추론(predict/eval)의 디바이스를 분리합니다.
#  - 학습은 GPU 서버에서 수행하므로 기본값 "cuda".
#  - 추론은 GPU 서버가 없는 서비스 환경(Celery 워커)에서 돌 수 있으므로 "cpu".
TRAIN_DEVICE_TYPE = "cuda"   # 학습 전용 디바이스 (기본 GPU)
INFER_DEVICE_TYPE = "cpu"    # 예측·평가(추론) 전용 디바이스

# ---------- 공통 설정 ----------
# 기본 학습 데이터 위치는 data/train 이며, 필요하면 환경변수로 바꿀 수 있다.
# 예: HITANDRUN_DATA_DIR=/path/to/train python3 -m model.main --mode train
DATA_DIR = Path(os.getenv("HITANDRUN_DATA_DIR", _ROOT / "data" / "train")).expanduser()
MODEL_NUM_CLASSES = 2
CLIP_LENGTH = 30
RESIZE = (224, 224)
R_VALUE = 1.0
TARGET_ID = 0
USE_AMP = True
USE_CHANNELS_LAST = True

# ---------- 사전학습 / 입력 정규화 ----------
# 백본은 torchvision S3D. 학습 시 Kinetics-400 사전학습 가중치로 초기화한다.
# (사전학습은 파라미터 '초기값'만 바꾸므로 모델 크기·추론 속도는 동일)
PRETRAINED = True
# 입력 정규화 통계 — S3D Kinetics-400 사전학습과 동일한 값 사용 (전이 효율 최대화)
NORM_MEAN = (0.43216, 0.394666, 0.37645)
NORM_STD = (0.22803, 0.22145, 0.216989)

# ---------- 학습 전용 ----------
# 학습 '작업용' 경로. 학습 중 best 가중치를 해당 파일로 갱신 저장했놓고, 학습 종료 시
# 규칙 파일명: (hitandrun_[YYMMDD]_[N]ep_[earlyY|N]_[손실율]) 으로 rename 된다(train.py).
# [YYMMDD] : 학습 시작 날짜, [N]ep : 학습 종료 시점 epoch, [earlyY|N] : 조기 종료 여부, [손실율] : 최종 검증 손실율(val loss)
TRAIN_BEST_MODEL_SAVE_PATH = _ROOT / "weights" / "hitandrun_model_best.pth"
TRAIN_BATCH_SIZE = 8  # GPU VRAM 상황에 맞게 조절 (예: 16, 32, 64 등)(기본값: 15)
TRAIN_NUM_EPOCHS = 100
TRAIN_SPLIT_RATIO = 0.8
TRAIN_EARLY_STOPPING_PATIENCE = 15 # patience 값 변경 10 -> 15로 변경 (이정주)
TRAIN_LEARNING_RATE = 0.00003  # S3D 미세조정 (헤드 기준; 백본은 train.py에서 자동 ×0.1 → 3e-6). 진동 억제 위해 1e-4에서 하향

# ---------- 웹 서비스(백엔드 Celery 워커) 전용 ----------
# 백엔드 prediction_job이 로드하는 배포 가중치. 반드시 S3D 구조(.pth)여야 한다.
SERVICE_WEIGHTS_PATH = Path("/Users/leezungzoo/Desktop/가중치/hitandrun_260828_32ep_earlyY_0.3807.pth")

# ---------- 단일 영상 예측/CAM 출력 전용 ----------
PREDICT_WEIGHTS_PATH = Path("/Users/leezungzoo/Desktop/가중치/hitandrun_260828_32ep_earlyY_0.3807.pth")
PREDICT_VIDEO_PATH = _ROOT / "data" / "eval" / "real01.mp4"
PREDICT_TXT_PATH = _ROOT / "data" / "eval" / "real01.txt"
PREDICT_OUTPUT_DIR = _ROOT / "data" / "predict_cam_result"
PREDICT_INFER_BATCH_SIZE = 2 # batch size 2로 바꾸었음(배기원)
PREDICT_WINDOW_STRIDE = 15  # CPU 웹 서비스 기본: 15 (GPU면 1로 낮춰 정확도↑)

# ---------- 웹 사고 이벤트 후처리 ----------
# 모델이 A 클래스를 조금이라도 높게 본 순간을 모두 이벤트로 만들면 주차 차량,
# 조명, 그림자에서 오탐이 많아지므로 확률/지속시간/움직임 조건을 함께 본다.
ACCIDENT_PROB_THRESHOLD = 0.70
ACCIDENT_HIGH_PROB_THRESHOLD = 0.90
ACCIDENT_HIGH_PROB_MOTION_THRESHOLD = 0.60
ACCIDENT_MIN_WINDOWS = 1
ACCIDENT_MAX_GAP_WINDOWS = 1
ACCIDENT_MIN_DURATION_SEC = 0.5
ACCIDENT_MOTION_THRESHOLD = 1.0

# ---------- 실제영상 정확도 평가 전용 ----------
EVAL_WEIGHTS_PATH = Path("/Users/leezungzoo/Desktop/가중치/hitandrun_260828_32ep_earlyY_0.3807.pth")
EVAL_FOLDER_PATH = _ROOT / "data" / "eval"
EVAL_NUM_SAMPLES = 10
EVAL_INFER_BATCH_SIZE = 8 # batch size 8로 바꾸었음(이정주)
EVAL_WINDOW_STRIDE = 1  # 기본값: 1 (올리면 속도↑ 정확도 소폭↓)
