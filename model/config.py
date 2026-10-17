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

# ---------- A 클립 방식 (실험용 스위치) ----------
# "pre" (기본): 위 방식 그대로 — 충돌 영상 1개당 창 1개(충돌 시작 PRE 프레임 전부터 30프레임).
#        충돌이 15프레임보다 길면 뒤쪽(충돌 끝)은 학습에 안 들어간다.
# "span": 접근(충돌 시작 PRE 프레임 전)부터 충돌 종료까지를 30프레임 창 여러 개로 덮고,
#        그 창을 '전부' A로 학습하되 충돌 1건의 무게 합은 1로 나눠 준다.
#          · 첫 창(접근 + 충돌 시작, = 'pre'의 창) : TRAIN_SPAN_FIRST_WEIGHT (기본 ½)
#          · 나머지 창들(충돌 끝 쪽)                : 남은 무게를 똑같이 나눔 (창 5개면 각 1/10)
#        → 구간 전체가 학습되고, 충돌은 1건으로 세며(A:S 균형 그대로), 접근 창은 항상 절반을
#          가져가 '접근 없는 흔들림 위주 창'으로 쏠리지 않는다.
#        구간이 창 하나에 들어가면 창 1개(무게 1) = 'pre'와 같다. RC 충돌(12~20프레임)은 접근
#        15를 더해 27~35프레임이라 창 1~2개(2개여도 1~5프레임 차이)라 사실상 'pre'와 같고,
#        realdata는 대부분 창 2~6개다.
#        ⚠️ 충돌 끝 쪽 창도 A로 배우므로 라벨의 충돌 종료 프레임이 정확해야 한다.
#        모델 구조·추론(서비스·평가)은 그대로다.
# 학습 명령에서 고른다:  python -m model.main --mode train --a-mode span
# (환경변수 TRAIN_A_MODE=span 으로도 된다. 둘 다 없으면 "pre")
# ⚠️ 켜면 TRAIN_A_SLICE_ENABLED 는 무시된다.
TRAIN_A_MODE = os.getenv("TRAIN_A_MODE", "pre").strip().lower()
TRAIN_SPAN_FIRST_WEIGHT = 0.5  # 첫 창(접근 + 충돌 시작)의 무게. 나머지 창이 1 − 이 값을 나눈다
# 창 간격 상한(프레임). 구간을 이 간격 이하로 고르게 나눠 창을 놓는다
# (15 = 창 절반씩 겹침. 30 이하면 구간에 빈틈이 없다).
TRAIN_SPAN_WINDOW_STRIDE = 15
# 충돌 하나의 최대 창 수. 넘으면 간격을 넓혀 이 개수로 구간 전체를 덮는다
# (6개면 접근 포함 175프레임까지 빈틈 없음).
TRAIN_SPAN_MAX_WINDOWS = 6
# 한 번에 GPU에 올리는 최대 창 수. 충돌 창이 늘어 배치의 창 수가 이 값을 넘으면 섞어서
# 고르게 나눠 역전파하고 기울기를 합쳐 한 번에 갱신한다(손실은 같고, BN 통계만 조각
# 단위로 계산된다). None = TRAIN_BATCH_SIZE × 2.
# 배치 15의 창 수는 평균 약 17개로 추정된다(로컬 realdata 평균 2.6개·RC 1.75개 기준).
# 긴 실제 충돌 여러 개가 한 배치에 몰리면 30개 가까이 된다. 'pre' 학습이 이미 GPU
# 메모리를 꽉 채운다면 TRAIN_BATCH_SIZE 로 낮춘다(그러면 최대 메모리가 'pre'와 같다).
TRAIN_SPAN_MAX_WINDOWS_PER_STEP = None

TRAIN_S_SLICE_ENABLED = True
# 비충돌(S) 영상을 몇 프레임 간격으로 잘라 클립을 만들지.
# 30 (2026-10-12~) = 논문과 같이 빈틈없이(=CLIP_LENGTH 간격) 잘라 S 영상의 거의 모든
#   순간을 학습한다. 90(3초 간격, 2026-09~10-12)은 실제 S 영상에서 움직임 있는 순간
#   (차가 박스 근처를 지나가는 등)을 35%만 담아(30은 96%), 그런 순간을 충돌로 보는
#   오탐(S→A)을 학습으로 막지 못했다. RC S는 25% → 81%.
#   대가: A:S 약 1:7.1 → 1:18.6(서버 추정, RC 833 + realdata), 에포크 시간 약 2.4배.
# ⚠️ 가중치 없이 1:10.4~1:12.9 에서 3에포크 내내 A recall 0.000('전부 S'로 붕괴)이
#   나온 적이 있다(아래 TRAIN_CLASS_WEIGHTS 주석). 첫 몇 에포크의 val_real recall 을
#   지켜보고, 계속 0이면 TRAIN_CLASS_WEIGHTS 를 켠다.
TRAIN_S_SLICE_STRIDE = 30
TRAIN_S_MAX_CLIPS_PER_VIDEO = 0       # 0=제한 없음. 영상이 매우 길 때 상한용
# 실제 영상(realdata) S 는 0번 차량 박스 외에 1번 차량 박스로도 클립을 만든다(시작점은 같고 박스만 다름).
#   · 1번 차량도 주차된 차이고 옆으로 차가 지나가는 장면이 담겨, 서비스에서 사용자가 박스를 치는
#     상황(주차된 피해 차량)과 같은 종류의 비충돌 샘플이 된다. 두 크롭은 대부분 겹치지 않는다.
#   · 2026-10-14 평가 S 를 늘리려 ReS 8개를 평가로 옮기면서 줄어든 실제 S 를 보충한다
#     (실제 S 클립 426 → 852, 전체 A:S 는 약 1:19 로 거의 그대로).
#   · RC S 는 0번만 쓴다(RC 까지 넣으면 S 클립이 약 1.7배가 되어 불균형이 커진다).
#   · 학습/검증 분할은 영상 단위라 같은 영상의 0번·1번 크롭이 학습과 검증으로 갈리지 않는다.
#     평가(data/eval)는 지금처럼 0번 박스만 쓴다.
TRAIN_S_REAL_USE_CAR1 = True

# 클래스 가중치(CrossEntropyLoss weight). S 슬라이싱으로 A:S 불균형이 커지면
# 모델이 S로 치우쳐 '미검출'이 늘 수 있다. (없음=None, 예: (1.0, 3.0) → A 3배)
# ※ 검증 없이 켜지 말 것 — 값에 따라 오탐이 급증할 수 있다.
#
# 실측: A 슬라이싱을 켜도 A:S ≈ 1:12.9 (rcdata 833 + realdata 기준).
#   rcdata의 A 구간은 12~20프레임이라 A 슬라이싱이 거의 작동하지 않는다.
#   가중치 없이 학습했더니 3에포크 내내 A recall 0.000 — '전부 S'로 붕괴했다.
# 값 선택: 역빈도 그대로(12.9)는 오탐이 급증할 위험이 커서 제곱근 완화값을 쓴다.
#   sqrt(12.9) ≈ 3.6. 이래도 recall이 0이면 12.9 쪽으로 올린다.
# → S 간격 90(A:S 약 1:7.1) 동안은 손실 가중을 껐다. 2026-10-12 간격을 30으로 바꿔
#   약 1:18.6이 됐지만 일단 가중 없이 학습해 본다. val_real recall 이 0에 머물면
#   sqrt(18.6) ≈ 4.3 → (1.0, 4.3) 부터 켠다.
TRAIN_CLASS_WEIGHTS = None
TARGET_ID = 0
USE_AMP = True
USE_CHANNELS_LAST = True

# ---------- 사전학습 / 입력 정규화 ----------
# 백본은 pytorchvideo X3D-M. 학습 시 Kinetics-400 사전학습 가중치로 초기화한다.
# (사전학습은 파라미터 '초기값'만 바꾸므로 모델 크기·추론 속도는 동일)
# 사전학습 사용 여부. 학습 명령으로 고른다(환경변수 PRETRAINED=0 도 같다):
#     python -m model.main --mode train --scratch    (무작위 초기화 = scratch 학습)
# scratch 면 백본도 분류 헤드와 같은 학습률로 학습하고(사전학습 땐 백본 ×0.1),
# 학습률 기본값은 TRAIN_SCRATCH_LEARNING_RATE 다(아래).
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
# 최종 가중치를 둘 폴더(= 이 경로의 폴더, 파일 이름은 쓰지 않는다). 학습 중 best·후보는
# 회차 폴더(outputs/trainlogs/<회차>/work_best.pth, work_cand_ep*.pth)에 저장하고, 학습이
# 끝나면 규칙 파일명 (hitandrun_[모델]_[YYMMDD_HHMM]_[N]ep_[earlyY|N]_[ptY|N]_[손실율]) 으로
# 이 폴더에 옮긴다(train.py). 회차마다 따로라 같은 조건의 학습을 동시에 돌려도 섞이지 않는다.
# [YYMMDD_HHMM] : 학습 시작 시각(= outputs/trainlogs 회차 폴더 이름의 시각), [N]ep : best 에포크,
# [earlyY|N] : 조기 종료 여부, [손실율] : best 저장 당시 판단 지표 값
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
# scratch(무작위 초기화) 학습률 — 모든 파라미터에 같은 값(AdamW). 검증된 값이 아니라 출발점이다.
#   · 논문(같은 과제·RC 데이터, Adam, batch 15, scratch): 1e-5. 다만 모델이 I3D 계열이다.
#   · X3D 공식 scratch(Kinetics 24만 영상): SGD 0.1/batch 128(batch 15면 약 0.012),
#     300에포크, 35에포크 워밍업, cosine — 대규모 데이터용이라 그대로 옮길 수 없다.
#   · 그 사이(1e-4)에서 시작하고, 서버에서 1e-5 / 1e-4 / 1e-3 을 짧게(약 10에포크) 돌려
#     val_real 균형 손실이 가장 잘 내려가는 값을 고른다:  --scratch --lr 1e-3
TRAIN_SCRATCH_LEARNING_RATE = 0.0001
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
# 학습·평가 산출물은 1회마다 '<YYMMDD_HHMM>_<모델>_<ptY|ptN>' 회차 폴더를 새로 만들어
# 그 안에 모은다(시각 = 시작 시각, model/runlog.py). 폴더 이름으로 언제 돌린 결과인지 구분한다.
# 학습: outputs/trainlogs/<회차>/  학습 곡선 CSV·그래프·결과 json, 그때의 config 사본,
#        학습/검증에 쓴 영상 목록(split.json). 가중치는 weights/ 에 같은 시각을 붙여 저장.
TRAIN_LOG_DIR = _ROOT / "outputs" / "trainlogs"
# 평가: outputs/evallogs/<회차>/  콘솔 전문 로그, 영상별 결과 CSV.
#        모델 비교용 요약은 outputs/evallogs/eval_summary.csv 한 파일에 계속 덧붙인다.
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
# 클립에 판정(S/A)·CAM 을 그리는 프레임 = 창 시작 + PREDICT_DISPLAY_OFFSET (0 ~ CLIP_LENGTH-1).
# 예전엔 창 끝(+29)에 붙여 A 표시가 실제 충돌보다 늦었다. 충돌 영상 14개를 stride 1 로
# 실측한 A 표시 시작 오차(프레임, +는 늦음):
#   +29(예전) 중앙값 +4 · 평균 +4.8 · 늦음 9/14  /  +25 중앙값 0 · 평균 +0.8 · 늦음 5/14
#   +15(창 가운데) 중앙값 -10 — 너무 이르다.
# 학습 A 클립은 충돌 시작 15프레임 전부터지만, 모델은 충돌 약 25프레임 전부터 시작하는 창을
# 처음 A로 판정하기 때문이다. A→S 전환은 어느 값이든 충돌 끝보다 중앙값 약 50프레임 늦다 —
# 모델이 충돌 뒤 흔들림이 담긴 창도 A로 보기 때문이라 이 값으로는 고칠 수 없다.
# 이벤트 구간·DB 기록·충격 시점(광학흐름 피크) 계산에는 영향이 없다.
PREDICT_DISPLAY_OFFSET = 25
# 서비스 사고 클립에 CAM 히트맵을 합성할지. 사용자 요청(2026-10-10)으로 기본은 끔 —
# 클립에는 차량 박스와 아래 상태 띠(S/A)만 그린다. 끄면 CAM 계산 자체를 건너뛴다.
# (CLI 전체 영상 출력 predict_hit_and_run_final 은 이 설정과 무관하게 CAM 을 그린다)
PREDICT_CLIP_CAM_OVERLAY = False
# 사고 클립 앞뒤로 붙이는 여유(프레임). 사고 판정 구간 앞뒤로 이만큼 더 잘라 보여 준다.
# (30fps 기준 1초. 프레임 단위라 fps 에 따라 실제 시간은 달라진다)
PREDICT_CLIP_PAD_FRAMES = 30

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

# ---------- 추론 실행 방식 (서비스·평가 공통, infer_runtime.py) ----------
# 서버 컴퓨터 사양이 정해져 있지 않아, 그 컴퓨터에서 쓸 수 있는 방식을 처음 한 번 자가 검사해
# 가장 빠른 것을 고른다. PyTorch 와 결과(P(A))가 같은 방식만 통과한다(정밀도는 FP32 그대로).
#   auto(기본) | torch | openvino | onnxruntime        (환경변수 INFER_BACKEND 로도 지정)
#   · 모델이 GPU(CUDA)에 있으면 auto 는 PyTorch GPU 를 쓴다.
#   · 2026-10-14 개발 PC(i5-1135G7, Linux) 측정: PyTorch 대비 OpenVINO 약 3~4배, ONNX Runtime 약 1.8배.
#   · openvino·onnxruntime 이 설치 안 됐거나, 변환·검사가 실패하면 다음 방식(최후엔 PyTorch)으로 넘어간다.
#   · 검사 결과·변환 파일은 INFER_CACHE_DIR 에 저장되고, 가중치·패키지 버전·CPU/GPU·OS 가 같으면
#     다음부터는 속도를 다시 재지 않고 서버를 띄울 때마다 창 1개로 결과만 빠르게 확인한다(수 초).
#     처음 한 번의 전체 검사(수십 초)를 미리 해 두기: python -m model.main --mode check-backend
#   · CAM 히트맵 합성(PREDICT_CLIP_CAM_OVERLAY, 개발용)을 켜면 그 경로는 PyTorch 로 돈다.
INFER_BACKEND = os.getenv("INFER_BACKEND", "auto").strip().lower()
# GPU FP16(autocast). Tensor Core 가 있는 NVIDIA GPU(연산능력 7.0 이상)에서만 켜지고 자가 검사도 거친다.
# 학습 서버에서 python -m model.fp16_check 로 속도·평가 결과를 확인한 뒤 켠다(환경변수 INFER_FP16=1).
INFER_FP16 = os.getenv("INFER_FP16", "0").strip().lower() in ("1", "true", "yes")
INFER_CHECK_MAX_PROB_DIFF = 1e-3   # 자가 검사: FP32 방식들의 P(A) 허용 차이 (개발 PC 실측 최대 8e-5)
INFER_FP16_MAX_PROB_DIFF = 1e-2    # 자가 검사: FP16 의 P(A) 허용 차이
INFER_CACHE_DIR = _ROOT / "weights" / ".infer_cache"   # 변환 파일(ONNX)·자가 검사 결과

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
