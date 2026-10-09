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
#  - 학습은 GPU 서버에서 수행하므로 "cuda".
#  - 추론도 GPU 서버에서 평가를 돌리므로 "cuda"로 둔다.
#    GPU가 없는 환경(로컬 Celery 워커)에서는 get_device()가 경고를 찍고
#    CPU로 자동 폴백하므로 이대로 둬도 서비스는 동작한다(느릴 뿐).
TRAIN_DEVICE_TYPE = "cuda"   # 학습 전용 디바이스 (기본 GPU)
INFER_DEVICE_TYPE = "cuda"    # 예측·평가(추론) 전용 디바이스

# ---------- 공통 설정 ----------
DATA_DIR = _ROOT / "data" / "train"
# 이 폴더가 학습/추론하는 백본 이름. 가중치 파일명에 붙어 모델을 구분한다.
MODEL_NAME = "x3d"
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

# ---------- 충돌(A) 클립 슬라이싱 ----------
# A는 원래 영상당 1클립(start_f 기준 30프레임)만 넣었다. 그런데 라벨된 충돌
# 구간은 중앙값 32프레임·최대 143프레임이라, 긴 충돌의 대부분이 학습에서 버려졌다.
# S는 영상 전체를 잘라 넣으므로(아래) A:S가 1:10.4까지 벌어져 모델이 '전부 S'로
# 붕괴한다(실측: 3에포크 내내 A recall 0.000).
# 여기서는 **라벨 구간 [start_f, end_f] 안에서만** 창을 밀어 A 클립을 늘린다.
#   · 구간 밖으로는 절대 나가지 않는다(나가면 틀린 라벨을 학습시키는 것).
#   · 마지막 창은 end_f에 정렬한다 — 논문이 최고 성능을 낸 '충돌 종료 기준' 클립.
# stride 15 기준 A 58 → 203클립, 불균형 1:10.4 → 1:3.0.
# ※ 끄려면 False (기존 동작 = 영상당 1클립)
# ⚠️ 슬라이싱을 켜면 첫 창을 뺀 나머지가 전부 '충돌 중'(접근 장면 없음)이 되어,
#    아래 TRAIN_A_PRE_FRAMES 로 얻는 이점을 잃는다. 불균형은 S 쪽에서 줄이고
#    A는 '접근이 담긴 창 1개'로 두는 편이 논문 방식에 가깝다.
TRAIN_A_SLICE_ENABLED = False
TRAIN_A_SLICE_STRIDE = 15

# ---------- A 클립에 '접근 장면'을 담기 ----------
# 창을 start_f(충돌 시작)부터 잡으면 클립에 '차가 다가오는 장면'이 한 프레임도
# 없다. 그러면 모델은 흔들림만으로 판단해야 하는데, 논문이 오탐의 주원인으로
# 지목한 게 바로 그 구분이다:
#   "the appearance of a vehicle moving slowly and the appearance of a vehicle
#    being slightly pushed out due to a collision is very similar"
# 논문은 창 끝을 '충돌 종료'에 맞췄고, rcdata 충돌 구간이 12~20프레임으로 짧아
# 창 앞쪽 10~18프레임이 자동으로 접근 장면이 됐다(그래서 30프레임이 최고 성능).
#   "In the case of the 30 frame, the vehicle approaches and collides, and the
#    vehicle shakes."
# 우리 realdata는 구간이 중앙 39프레임이라 end_f 정렬로는 창이 구간 안에 갇힌다.
# 그래서 구간 길이와 무관하게 start_f 보다 이만큼 앞에서 창을 시작한다.
# 15 = 창의 절반을 접근에 배분(접근 15 + 충돌 15). 논문 rcdata의 14와 비슷하다.
TRAIN_A_PRE_FRAMES = 15

TRAIN_S_SLICE_ENABLED = True
# 논문은 비충돌 영상을 빈틈없이(=CLIP_LENGTH 간격) 잘라 전부 넣는다. 그대로 하면
# S 클립이 A의 10배가 되고(1:10.4), batch 15의 33%에 A가 한 개도 안 들어가
# 모델이 '전부 S'로 붕괴한다(실측: A recall 0.000).
# 연속한 S 클립은 같은 주차장·같은 차를 1초 차이로 본 것이라 사실상 중복이므로,
# 간격을 3초(90프레임)로 벌려 중복만 덜어낸다. A:S 1:10.4 → 약 1:4.3,
# 에포크 시간도 2.6배 짧아진다. 모든 클립을 매 에포크 한 번씩 보는 성질은 유지된다
# (WeightedRandomSampler 로 비율을 맞추면 S 클립 일부가 에포크마다 누락된다).
TRAIN_S_SLICE_STRIDE = 90
TRAIN_S_MAX_CLIPS_PER_VIDEO = 0       # 0=제한 없음. 영상이 매우 길 때 상한용

# 클래스 가중치(CrossEntropyLoss weight). S 슬라이싱으로 A:S 불균형이 커지면
# 모델이 S로 치우쳐 '미검출'이 늘 수 있다. (없음=None, 예: (1.0, 3.0) → A 3배)
# ※ 검증 없이 켜지 말 것 — 값에 따라 오탐이 급증할 수 있다.
#
# 실측: A 슬라이싱을 켜도 A:S ≈ 1:12.9 (rcdata 833 + realdata 기준).
#   rcdata의 A 구간은 12~20프레임이라 A 슬라이싱이 거의 작동하지 않는다.
#   가중치 없이 학습했더니 3에포크 내내 A recall 0.000 — '전부 S'로 붕괴했다.
# 값 선택: 역빈도 그대로(12.9)는 오탐이 급증할 위험이 커서 제곱근 완화값을 쓴다.
#   sqrt(12.9) ≈ 3.6. 이래도 recall이 0이면 12.9 쪽으로 올린다.
# → stride 90 으로 불균형을 1:4.3 까지 낮췄으므로 손실 가중은 끈다.
#   둘을 같이 걸면 A를 이중으로 강조해 오탐이 급증한다.
TRAIN_CLASS_WEIGHTS = None
TARGET_ID = 0
USE_AMP = True
USE_CHANNELS_LAST = True

# ---------- 사전학습 / 입력 정규화 ----------
# 백본은 pytorchvideo X3D-M. 학습 시 Kinetics-400 사전학습 가중치로 초기화한다.
# (사전학습은 파라미터 '초기값'만 바꾸므로 모델 크기·추론 속도는 동일)
# 사전학습 사용 여부. 환경변수로 덮어써서 같은 코드로 A/B를 돌릴 수 있다:
#     PRETRAINED=0 python main.py --mode train      (scratch 학습)
# ※ 논문(Hwang & Lee 2024)은 3D-CNN을 scratch로 학습했으나 그 이유나 비교실험을
#   제시하지 않았다. 어느 쪽이 나은지는 우리 데이터로 직접 비교해야 한다.
PRETRAINED = os.getenv("PRETRAINED", "1").strip().lower() not in ("0", "false", "no")
# 결과물이 서로 덮어쓰이지 않도록 사전학습 여부를 파일명에 박는다(ptY/ptN).
PRETRAIN_TAG = "ptY" if PRETRAINED else "ptN"
# 입력 정규화 통계 — Kinetics-400 사전학습과 동일한 값 사용 (전이 효율 최대화)
# ⚠️ pytorchvideo 계열(X3D/SlowFast)은 torchvision(S3D)과 통계가 다르다.
#    pytorchvideo 기본값: transforms/transforms_factory.py 의 video_mean/video_std.
#    학습과 추론이 반드시 같은 값을 써야 하므로 여기서 일괄 관리한다.
NORM_MEAN = (0.45, 0.45, 0.45)
NORM_STD = (0.225, 0.225, 0.225)

# ---------- 학습 전용 ----------
# 학습 '작업용' 경로. 학습 중 best 가중치를 해당 파일로 갱신 저장했놓고, 학습 종료 시
# 규칙 파일명: (hitandrun_[YYMMDD]_[N]ep_[earlyY|N]_[손실율]) 으로 rename 된다(train.py).
# [YYMMDD] : 학습 시작 날짜, [N]ep : 학습 종료 시점 epoch, [earlyY|N] : 조기 종료 여부, [손실율] : 최종 검증 손실율(val loss)
TRAIN_BEST_MODEL_SAVE_PATH = (
    _ROOT / "weights" / f"hitandrun_{MODEL_NAME}_{PRETRAIN_TAG}_best.pth")
TRAIN_BATCH_SIZE = 15  # 논문과 동일(Hwang&Lee 2024: batch 15). VRAM 부족하면 낮출 것
# ⚠️ 배치가 작으면 A:S 1:12.9 에서 '배치에 A가 한 개도 없는' 일이 잦다
#    (batch 8 → 55%, batch 15 → 33%, batch 32 → 9%). 그런 배치에서는
#    클래스 가중치를 얼마로 올려도 A 학습 신호가 0이다.
TRAIN_NUM_EPOCHS = 100
TRAIN_SPLIT_RATIO = 0.8
TRAIN_EARLY_STOPPING_PATIENCE = 10

# ---------- best 판단 지표 / 후보 저장 ----------
# best 저장·LR 감소(ReduceLROnPlateau)·조기종료가 모두 아래 '판단 지표' 하나로 움직인다.
#   판단 지표 = val_real 의 클래스 균형 손실 (충돌 평균 손실 + 비충돌 평균 손실) / 2
#               의 최근 TRAIN_MONITOR_SMOOTH 에포크 평균
# 왜 그냥 val_real loss 가 아닌가 (x3d-ptY 260922 학습 로그로 확인):
#   · val_real 51클립 중 40개가 비충돌이라 그냥 손실의 78%가 '비충돌을 얼마나
#     확신하느냐'로 정해진다. 실제로 26에포크(검출 9/11·오탐 1)가 33에포크
#     (8/11·오탐 2)보다 둘 다 나았는데 손실이 높아 선택되지 않았다.
#   · 한 에포크 값은 수렴 후에도 표준편차 0.013 으로 흔들려, 운 좋게 한 번 낮은
#     에포크가 뽑힌다. 평균하면 흔들림이 약 √N 배 줄고, 대신 약 1에포크 늦게 반응한다.
TRAIN_MONITOR_SMOOTH = 3
# 후보: 판단 지표 상위 K개를 저장하되 서로 MIN_GAP 에포크 이상 떨어진 것만 남긴다
# (붙어 있는 에포크는 사실상 같은 모델이라 비교할 의미가 없다). 1위는 best 와 같다.
# 최종 선택은 학습 후, 학습에 안 쓴 영상으로 서비스 경로 평가를 돌려서 한다.
TRAIN_TOPK_CANDIDATES = 3
TRAIN_TOPK_MIN_GAP = 3
TRAIN_LEARNING_RATE = 0.00003  # X3D-M 미세조정 (헤드 기준; 백본은 train.py에서 자동 ×0.1 → 3e-6). 진동 억제 위해 1e-4에서 하향
# AdamW의 decoupled weight decay. ⚠️ Adam에 weight_decay를 주면 L2 페널티가
# 적응적 학습률에 의해 파라미터마다 왜곡되므로, 정규화 목적이면 AdamW를 써야 한다.
#
# 0.01(AdamW 기본)에서 0.1로 올림. 근거는 실측된 과적합이다:
#   s3d+ptY 5에포크 — train 0.452→0.318 (↓) 인데 val_real 0.570→0.612 (↑),
#   best 가 1에포크. train 손실이 2에포크부터 'A를 전혀 못 맞히고 낼 수 있는
#   최소 손실'(0.407) 아래로 내려갔다 = 학습 A는 맞히지만 검증 A로 일반화 실패.
# ⚠️ decoupled weight decay 는 LR과 곱해진다(θ ← θ − lr·λ·θ). 우리 LR이
#    헤드 3e-5 / 백본 3e-6 이라 λ=0.01 로는 100에포크 전체에서 가중치가 0.2%
#    밖에 줄지 않아 사실상 정규화가 걸리지 않았다. λ=0.1 이면 헤드 약 2%.
TRAIN_WEIGHT_DECAY = 0.1
# 에포크별 학습 곡선 CSV 저장 위치 (과적합 판단·모델 비교용)
TRAIN_LOG_DIR = _ROOT / "outputs" / "trainlogs"
# 평가 결과 로그(콘솔 전문 + 5가지 모델 비교용 요약 CSV) 저장 위치
EVAL_LOG_DIR = _ROOT / "outputs" / "evallogs"

# ---------- 웹 서비스(백엔드 Celery 워커) 전용 ----------
# 백엔드 prediction_job이 로드하는 배포 가중치. 반드시 X3D-M 래퍼 구조(.pth)여야 한다.
SERVICE_WEIGHTS_PATH = _ROOT / "weights" / "hitandrun_x3d_260922_33ep_earlyY_ptY_0.1999.pth"

# ---------- 단일 영상 예측/CAM 출력 전용 ----------
PREDICT_WEIGHTS_PATH = _ROOT / "weights" / "hitandrun_x3d_260922_33ep_earlyY_ptY_0.1999.pth"
PREDICT_VIDEO_PATH = _ROOT / "data" / "eval" / "real01.mp4"
PREDICT_TXT_PATH = _ROOT / "data" / "eval" / "real01.txt"
PREDICT_OUTPUT_DIR = _ROOT / "data" / "predict_cam_result"
PREDICT_INFER_BATCH_SIZE = 2 # batch size 2로 바꾸었음(배기원)
# 슬라이딩 윈도우 간격(프레임). 서비스는 GPU 추론을 가정하므로 모든 위치를 본다(1).
# stride 5 는 CPU 추론 시절의 절충값이었다(창 수 1/5 → 속도 5배).
# ⚠️ 아래 PREDICT_MIN_EVENT_SPAN_FRAMES(35)는 stride 1 기준 값이다. stride 를 바꾸면
#    '창 몇 개짜리 이벤트까지 지우는가'가 달라지므로 함께 다시 봐야 한다.
#    (stride 1: 연속 A 창 7개 미만이면 삭제 / stride 5: 2개 미만이면 삭제)
# ※ UI 의 예상시간은 실제 진행률로 계산하므로 이 값과 무관하다
#   (App.jsx 의 estimatedSec 은 만들기만 하고 쓰이지 않는다).
PREDICT_WINDOW_STRIDE = 1

# ---------- 2단계 파이프라인: 모델 거친 탐색 → 의심 구간만 촘촘히 ----------
# 30분~1시간 영상의 모든 창(stride 1)을 3D-CNN 에 넣으면 느리다. 그래서
#   1) PREDICT_COARSE_STEP 프레임 간격의 창만 먼저 모델로 본다(거친 탐색 = 전체의 1/10).
#   2) P(A) > PREDICT_COARSE_PROB 인 창 주변 ±PREDICT_COARSE_PAD 프레임의 창을 모두 본다.
#   3) 끝까지 안 본 창은 비충돌로 간주한다.
# 예전엔 광학흐름 크기로 선별했는데, 미세 충돌은 충돌 중 움직임이 평소 흔들림과 구분되지
# 않아 기준선을 어떻게 잡아도 걸러졌다(ReA_0023: 흐름 최대 0.33 < 기준선 0.42 라 평가조차
# 안 됐지만, 모델은 P(A) 0.72~0.96 으로 잡는다). 모델로 선별하면 '모델이 잡을 수 있는
# 충돌'이 선별 때문에 빠지지 않는다.
# 시뮬레이션(27개 영상, stride 5 확률 근사, 병합 3초·최소 길이 35):
#   전체 스캔 충돌 14/16·오탐 클립 4 / 거친 탐색 10간격 14/16·4 (같음)
#   / 흐름 3·MAD(긴 영상 가정, 폴백 없음) 12/16.
#   비충돌 11개 중 6개는 P(A)>0.3 이 한 번도 없어 거친 탐색 10%로 끝났다(평균 10% + 24%).
# ⚠️ 연속 A 창 10개 이상인 충돌은 거친 탐색 창을 반드시 하나 품으므로 놓치지 않는다.
#    빠질 수 있는 건 7~9창(최소 길이 35를 겨우 넘는)짜리이면서 주변 P(A)도 0.3 이하인
#    경우뿐이다. 0.3 은 A 판정(0.5)보다 낮춰 약한 신호도 줍기 위한 값.
# ⚠️ 긴 영상에서 실제로 몇 %를 보는지는 서버 로그의 '[2단계] 거친 탐색 … N%' 줄로 확인.
PREDICT_USE_COARSE_SCAN = True
PREDICT_COARSE_STEP = 10       # 거친 탐색 간격(프레임). stride 의 배수로 맞춰 쓴다
PREDICT_COARSE_PROB = 0.3      # 이 P(A)를 넘는 거친 창 주변을 촘촘히 본다
PREDICT_COARSE_PAD = CLIP_LENGTH  # 의심 창 주변 ±pad 프레임의 창을 모두 평가
PREDICT_FLOW_SAMPLE_STEP = 2   # 충격 시점 판정용 광학흐름 계산 프레임 간격(속도)

# ---------- 이벤트 후처리 (분할 방지 / 깜빡임 제거) ----------
# 모델 판정이 A→S→A로 깜빡이면 한 충돌이 여러 이벤트로 쪼개진다(실측: 실차 영상
# 1건이 6개로 분할). 아래 두 단계로 정리한다. 순서는 '병합 → 길이필터'.
#   1) 간격이 가까운 이벤트는 같은 사고로 보고 병합
#   2) 그래도 너무 짧은(단발 깜빡임) 이벤트는 제거
# 병합 간격은 '초'로 정하고 영상 fps로 프레임 환산한다. 사용자에게 중요한 건 충돌이
# '시작된 지점'이고, 몇 초 안에 또 판정이 나오면 같은 사고의 연속으로 보는 게 자연스럽다.
# 프레임으로 고정하면 fps(10~60)에 따라 같은 값이 9초~1.5초로 달라진다.
# 3초 = 30fps 기준 90프레임. (이전 값: 45프레임 ≈ 1.5초)
PREDICT_EVENT_MERGE_GAP_SEC = 3.0
# 최소 길이 — 이벤트 길이 = 28 + (A 로 판정된 연속 창 수)  [stride 1 기준]
#   · 창이 30프레임이라 A 창이 1개만 있어도 29프레임이다. 그래서 29 이하로 두면 필터를
#     끈 것과 같다.
#   · 이벤트 길이는 '충돌이 실제로 지속된 시간'이 아니다. 짧은 충돌도 그 장면을 담는
#     창이 30개라, 모델이 잡으면 여러 창이 연속으로 A가 된다.
#   · 실측(x3d-ptY 33ep, 선별 끔): 진짜 충돌 14건의 연속 A 창 수 30~220, 단발 오탐 약 5.
#     35(연속 7창)와 40(12창)은 이 데이터에서 결과가 같았다. 약하게 잡힌 미세 충돌은
#     연속 창이 30보다 짧을 수 있어, 놓치지 않는 쪽으로 여유를 둔 35를 쓴다.
#   · 초가 아니라 프레임으로 둔다 — 창 길이가 fps와 무관하게 30프레임이기 때문.
#   (이전 값 40: 옛 모델로 stride 1 에서 '단발 29f / 지속 48f 이상' 사이를 자른 값)
PREDICT_MIN_EVENT_SPAN_FRAMES = 35

# ---------- 실제영상 정확도 평가 전용 ----------
# 평가할 가중치. 여러 학습 결과를 비교할 땐 환경변수로 골라 쓴다:
#     EVAL_WEIGHTS=weights/hitandrun_s3d_..._ptN_0.31.pth python main.py --mode eval
EVAL_WEIGHTS_PATH = (Path(os.environ["EVAL_WEIGHTS"]) if os.getenv("EVAL_WEIGHTS")
                     else _ROOT / "weights" / "hitandrun_x3d_260922_33ep_earlyY_ptY_0.1999.pth")
EVAL_FOLDER_PATH = _ROOT / "data" / "eval"
# 서비스 경로 평가에서 '검출'로 인정하는 시작 시점 허용 오차(초).
# 이벤트가 라벨 start_f ± 이 값 안에서 시작해야 검출로 센다. 이벤트가 나오긴 했어도
# 엉뚱한 곳이면 검출이 아니다. 1초 = 30fps 기준 윈도우 하나 길이.
EVAL_HIT_TOLERANCE_SEC = 1.0
EVAL_INFER_BATCH_SIZE = 8 # batch size 8로 바꾸었음(이정주)
EVAL_WINDOW_STRIDE = 1  # 기본값: 1 (올리면 속도↑ 정확도 소폭↓)
