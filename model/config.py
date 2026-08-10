from pathlib import Path

# ==========================================
# [파라미터 사용자 설정]
# 실행 환경에 맞게 이 파일의 값만 수정하세요.
# ==========================================

# 프로젝트 루트 기준 경로 (model/ 의 상위 폴더)
_ROOT = Path(__file__).resolve().parent.parent

# ---------- 디바이스 설정 ----------
# "cuda" : NVIDIA GPU (CUDA 빌드) ← 현재 설정
#          또는 AMD GPU (네이티브 Linux + ROCm 빌드)
# "cpu"  : CPU
DEVICE_TYPE = "cuda"

# ---------- 공통 설정 ----------
DATA_DIR = _ROOT / "data" / "train"
MODEL_NUM_CLASSES = 2
CLIP_LENGTH = 30
RESIZE = (224, 224)
R_VALUE = 1.0
TARGET_ID = 0
USE_AMP = True
USE_CHANNELS_LAST = True

# ---------- 학습 전용 ----------
TRAIN_BEST_MODEL_SAVE_PATH = _ROOT / "weights" / "hitandrun_model_best.pth"
TRAIN_BATCH_SIZE = 4  # GPU VRAM 상황에 맞게 조절 (예: 16, 32, 64 등)(기본값: 4)(3080 12GB VRAM 기준)
TRAIN_NUM_EPOCHS = 30
TRAIN_SPLIT_RATIO = 0.8
TRAIN_EARLY_STOPPING_PATIENCE = 10
TRAIN_LEARNING_RATE = 0.001

# ---------- 단일 영상 예측/CAM 출력 전용 ----------
PREDICT_WEIGHTS_PATH = _ROOT / "weights" / "hitandrun_model_best.pth"
PREDICT_VIDEO_PATH = _ROOT / "data" / "real" / "real01.mp4"
PREDICT_TXT_PATH = _ROOT / "data" / "real" / "real01.txt"
PREDICT_OUTPUT_DIR = _ROOT / "data" / "predict_cam_result"
PREDICT_INFER_BATCH_SIZE = 4
PREDICT_WINDOW_STRIDE = 1  # 기본값: 1 (올리면 속도↑ 정확도 소폭↓)

# ---------- 실제영상 정확도 평가 전용 ----------
EVAL_WEIGHTS_PATH = _ROOT / "weights" / "hitandrun_model_best.pth"
EVAL_FOLDER_PATH = _ROOT / "data" / "eval"
EVAL_NUM_SAMPLES = 10
EVAL_INFER_BATCH_SIZE = 4
EVAL_WINDOW_STRIDE = 1  # 기본값: 1 (올리면 속도↑ 정확도 소폭↓)
