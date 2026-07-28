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
DATA_DIR = _ROOT / "data" / "train"
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
TRAIN_BEST_MODEL_SAVE_PATH = _ROOT / "weights" / "hitandrun_model_best.pth"
TRAIN_BATCH_SIZE = 8  # GPU VRAM 상황에 맞게 조절 (예: 16, 32, 64 등)(기본값: 15)
TRAIN_NUM_EPOCHS = 100
TRAIN_SPLIT_RATIO = 0.8
TRAIN_EARLY_STOPPING_PATIENCE = 10
TRAIN_LEARNING_RATE = 0.00003  # S3D 미세조정 (헤드 기준; 백본은 train.py에서 자동 ×0.1 → 3e-6). 진동 억제 위해 1e-4에서 하향

# ---------- 단일 영상 예측/CAM 출력 전용 ----------
PREDICT_WEIGHTS_PATH = _ROOT / "weights" / "hitandrun_model_best.finally.pth"
PREDICT_VIDEO_PATH = _ROOT / "data" / "real" / "real01.mp4"
PREDICT_TXT_PATH = _ROOT / "data" / "real" / "real01.txt"
PREDICT_OUTPUT_DIR = _ROOT / "data" / "predict_cam_result"
PREDICT_INFER_BATCH_SIZE = 16
PREDICT_WINDOW_STRIDE = 15  # CPU 추론 기본: 15 (GPU면 1로 낮춰 정확도↑)

# ---------- 실제영상 정확도 평가 전용 ----------
EVAL_WEIGHTS_PATH = _ROOT / "weights" / "hitandrun_model_best.pth"
EVAL_FOLDER_PATH = _ROOT / "data" / "eval"
EVAL_NUM_SAMPLES = 10
EVAL_INFER_BATCH_SIZE = 16
EVAL_WINDOW_STRIDE = 1  # 기본값: 1 (올리면 속도↑ 정확도 소폭↓)
