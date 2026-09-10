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
# 이 폴더가 학습/추론하는 백본 이름. 가중치 파일명에 붙어 모델을 구분한다.
MODEL_NAME = "slowfast"
MODEL_NUM_CLASSES = 2
CLIP_LENGTH = 30
RESIZE = (224, 224)
R_VALUE = 1.0
# r 증강(학습 전용): 논문은 r∈{1,1.5,2,2.33,3}로 클립을 확장(833→1457, 약 1.75배).
# 우리는 인덱스를 늘리면 train/val 분할이 깨지거나 같은 영상이 양쪽에 들어가므로,
# 학습 시 클립마다 r을 확률적으로 뽑아 같은 효과를 낸다(검증/추론은 항상 r=1.0).
#
# ⚠️ r=1.0이 지배적이어야 한다: 추론은 언제나 r=1.0이므로 균등 추출하면 학습의
#    80%가 '추론에서 안 쓰는 조건'이 된다. 논문의 확장도 1.75배(≈5배가 아님)라
#    r=1이 주류였음을 시사한다. 그래서 AUG_PROB 확률로만 큰 r을 쓴다.
# 부수 효과: 사용자가 드래그하거나 YOLO가 잡은 bbox의 '타이트함 편차'에 강건해진다.
# 비활성화하려면 TRAIN_R_AUGMENT_PROB = 0.0. ※ 다음 학습부터 적용됨
TRAIN_R_AUGMENT_PROB = 0.4            # 이 확률로만 r을 바꾸고, 나머지 60%는 r=1.0
TRAIN_R_AUGMENT_VALUES = (1.5, 2.0, 2.33, 3.0)

# ---------- 비충돌(S) 클립 슬라이싱 ----------
# 기존 구현은 S 영상당 '맨 앞 30프레임' 1개만 학습에 넣었다(start_f=0 고정).
# 그러면 모델이 배우는 '비충돌'이 사실상 "아직 차가 안 들어온 빈 장면"이 되어,
# "차가 지나가지만 부딪히지는 않는" 장면을 학습하지 못한다 → 오탐의 직접 원인.
# 논문은 이 문제를 명시하고 비충돌 영상을 통째로 잘라 전부 학습에 넣었다:
#   "the appearance of a slowly moving vehicle and a vehicle slightly displaced
#    due to a collision are quite similar. To reduce the false alarm rate ...
#    we divided the non-collision videos into frames of a specific length"
# ※ 학습 데이터가 크게 늘고 A:S 불균형도 커진다. 끄려면 False.
# 시간 오프셋 지터 폭(초). 프레임이 아니라 '초'로 두어야 fps가 다른 영상들에서
# 같은 물리적 의미를 갖는다. 0.33초 = 30fps 기준 0~10프레임(기존 동작과 동일).
TRAIN_TIME_JITTER_SEC = 0.33

TRAIN_S_SLICE_ENABLED = True
TRAIN_S_SLICE_STRIDE = CLIP_LENGTH    # 30 = 겹치지 않게 연속 분할(논문 방식)
TRAIN_S_MAX_CLIPS_PER_VIDEO = 0       # 0=제한 없음. 영상이 매우 길 때 상한용

# 클래스 가중치(CrossEntropyLoss weight). S 슬라이싱으로 A:S 불균형이 커지면
# 모델이 S로 치우쳐 '미검출'이 늘 수 있다. (없음=None, 예: (1.0, 3.0) → A 3배)
# ※ 검증 없이 켜지 말 것 — 값에 따라 오탐이 급증할 수 있다.
TRAIN_CLASS_WEIGHTS = None
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
TRAIN_BEST_MODEL_SAVE_PATH = _ROOT / "weights" / f"hitandrun_{MODEL_NAME}_best.pth"
TRAIN_BATCH_SIZE = 8  # GPU VRAM 상황에 맞게 조절 (예: 16, 32, 64 등)(기본값: 15)
TRAIN_NUM_EPOCHS = 100
TRAIN_SPLIT_RATIO = 0.8
TRAIN_EARLY_STOPPING_PATIENCE = 15 # patience 값 변경 10 -> 15로 변경 (이정주)
TRAIN_LEARNING_RATE = 0.00003  # S3D 미세조정 (헤드 기준; 백본은 train.py에서 자동 ×0.1 → 3e-6). 진동 억제 위해 1e-4에서 하향

# ---------- 웹 서비스(백엔드 Celery 워커) 전용 ----------
# 백엔드 prediction_job이 로드하는 배포 가중치. 반드시 S3D 구조(.pth)여야 한다.
SERVICE_WEIGHTS_PATH = _ROOT / "weights" / "hitandrun_260828_32ep_earlyY_0.3807.pth"

# ---------- 단일 영상 예측/CAM 출력 전용 ----------
PREDICT_WEIGHTS_PATH = _ROOT / "weights" / "hitandrun_260828_32ep_earlyY_0.3807.pth"
PREDICT_VIDEO_PATH = _ROOT / "data" / "eval" / "real01.mp4"
PREDICT_TXT_PATH = _ROOT / "data" / "eval" / "real01.txt"
PREDICT_OUTPUT_DIR = _ROOT / "data" / "predict_cam_result"
PREDICT_INFER_BATCH_SIZE = 2 # batch size 2로 바꾸었음(배기원)
PREDICT_WINDOW_STRIDE = 5  # CPU 추론 절충값 (윈도우 83% 중첩 → 정확도 거의 유지 + 속도↑). GPU면 1로 낮춰 정확도↑
# ⚠️ 이 값을 바꾸면 ui/src/App.jsx 의 예상시간 공식(totalFrames/stride)도 같이 맞춰야 함

# ---------- 2단계 파이프라인: 광학흐름 사전선별 ----------
# 멀티-아워 영상 대비: 피해차 크롭 영역의 옵티컬 플로우로 '사고 의심 시점'만
# 먼저 뽑고, 무거운 3D-CNN은 그 근처 윈도우에만 실행한다(고재현율 그물→CNN 확정).
# 검증: 라벨된 충돌 5/5에서 플로우 피크가 충돌구간과 일치, 현저도 10~797x.
PREDICT_USE_FLOW_PRESCREEN = True
PREDICT_FLOW_THRESHOLD_FACTOR = 2.0   # τ=median+factor·std. 낮을수록 고재현율(후보↑)
PREDICT_FLOW_PRESCREEN_PAD = CLIP_LENGTH  # 스파이크 주변 ±pad 프레임 윈도우까지 평가
PREDICT_FLOW_SAMPLE_STEP = 2          # 선별 단계 플로우 계산 프레임 간격(속도)
PREDICT_FLOW_MAX_SUSPICIOUS_RATIO = 0.8  # 의심 프레임이 이 비율 초과면 전체스캔 폴백

# ---------- 이벤트 후처리 (분할 방지 / 깜빡임 제거) ----------
# 모델 판정이 A→S→A로 깜빡이면 한 충돌이 여러 이벤트로 쪼개진다(실측: 실차 영상
# 1건이 6개로 분할). 아래 두 단계로 정리한다. 순서는 '병합 → 길이필터'.
#   1) 간격이 가까운 이벤트는 같은 사고로 보고 병합
#   2) 그래도 너무 짧은(단발 깜빡임) 이벤트는 제거
# 근거(실측 stride=1, 구간길이 프레임): 단일 윈도우 깜빡임=29프레임에 몰려 있고,
# 지속 검출은 48프레임 이상에 분포 → 그 사이인 40으로 컷.
PREDICT_EVENT_MERGE_GAP_FRAMES = 45   # 이벤트 간 간격 ≤ 45면 하나로 병합(≈1.5초)
PREDICT_MIN_EVENT_SPAN_FRAMES = 40    # 구간 길이가 이보다 짧으면 깜빡임으로 보고 제거

# ---------- 실제영상 정확도 평가 전용 ----------
EVAL_WEIGHTS_PATH = _ROOT / "weights" / "hitandrun_260828_32ep_earlyY_0.3807.pth"
EVAL_FOLDER_PATH = _ROOT / "data" / "eval"
EVAL_INFER_BATCH_SIZE = 8 # batch size 8로 바꾸었음(이정주)
EVAL_WINDOW_STRIDE = 1  # 기본값: 1 (올리면 속도↑ 정확도 소폭↓)
