import os
from pathlib import Path

# ==========================================
# [사용자 설정]
# 실행 환경에 맞게 이 파일의 값만 수정합니다.
# ==========================================

# 프로젝트 루트 기준 경로(model/의 상위 폴더)
_ROOT = Path(__file__).resolve().parent.parent

# ---------- 장치 설정 ----------
# "cuda": NVIDIA GPU(CUDA) 또는 AMD GPU(Linux + ROCm)
# "cpu": CPU
# 학습과 추론(예측, 평가)의 장치를 나눠 둡니다. GPU가 없는 환경에서는 get_device()가 경고를 남기고
# CPU로 바꾸므로 그대로 두어도 서비스는 동작합니다(느릴 뿐입니다).
TRAIN_DEVICE_TYPE = "cuda"   # 학습 장치(기본 GPU)
INFER_DEVICE_TYPE = "cuda"    # 예측, 평가(추론) 장치

# ---------- 공통 설정 ----------
DATA_DIR = _ROOT / "data" / "train"
# 검증 영상 폴더입니다. 이 폴더에 영상이 있으면 학습은 DATA_DIR 전부, 검증은 이 폴더 전부를 씁니다(분할 계산 없음).
# 옮기기: python -m model.val_move --apply(model/split_lock.json의 목록을 같은 상대경로로). 없으면 아래 TRAIN_SPLIT_LOCK을 씁니다.
VAL_DATA_DIR = _ROOT / "data" / "val"
# 이 폴더의 백본 이름입니다. 가중치 파일 이름에 붙어 모델을 구분합니다.
MODEL_NAME = "x3d"
MODEL_NUM_CLASSES = 2
CLIP_LENGTH = 30
# 모델 입력 크기(정사각형 한 변, px)입니다. 차량 크롭을 이 크기로 바꿔 넣으며 기본은 224입니다.
# 환경변수 INPUT_SIZE=256으로 바꿀 수 있습니다. 큰 차의 미세한 흔들림이 덜 뭉개지지만 계산과 메모리가 약 1.31배가 됩니다.
# 학습한 크기와 같은 값으로 평가와 서비스를 해야 합니다. 224가 아닌 학습은 이름에 _s256이 붙고,
# 가중치를 읽을 때(hitandrun_model.check_input_size) 크기가 다르면 멈춥니다. 검증 클립 저장본은 크기별로 따로 만듭니다.
INPUT_SIZE = int(os.getenv("INPUT_SIZE", "224") or 224)
RESIZE = (INPUT_SIZE, INPUT_SIZE)
# 헤드 시간 풀링입니다. 헤드는 백본 특징(30시점)을 시점마다 432채널로 넓힌 뒤 시간 축을 하나로 모읍니다.
# avg는 30시점 평균(기본), top3는 채널과 칸마다 가장 큰 3시점의 평균, avgtop3는 둘의 평균입니다.
# 짧게 반응한 충돌이 평균에 묻히지 않게 하려고 top3를 둡니다. 계산량은 셋 다 같습니다.
# 학습에서 켜려면 환경변수 HEAD_TEMPORAL_POOL=top3를 주며, 가중치 이름에 _top3 또는 _avgtop3가 붙습니다.
# 평가와 서비스는 환경변수를 주지 않으면 가중치 이름의 표시를 따르고, 줬는데 이름과 다르면 멈춥니다(hitandrun_model.check_weights).
HEAD_TEMPORAL_POOL = (os.getenv("HEAD_TEMPORAL_POOL", "avg") or "avg").strip().lower()
HEAD_TEMPORAL_POOL_SET = bool((os.getenv("HEAD_TEMPORAL_POOL") or "").strip())
HEAD_TEMPORAL_TOPK = 3
R_VALUE = 1.0
# 묶음 하나를 충돌(A)로 보는 P(A) 기준입니다. 0.5면 두 클래스 중 큰 쪽(argmax)과 같습니다.
# 학습 검증 지표, suggest F1, 평가, 서비스 이벤트, 클립 A/S 띠, robust_check가 모두 이 값을 씁니다(infer_runtime.is_accident).
# 판단 지표(균형 손실)와 구분력은 기준과 무관합니다.
# 바꿀 때 주의할 점
#     1) 최종 가중치 하나에 대해 검증 데이터로 정합니다(python -m model.postproc_check). 평가 영상으로 고르면 점수가 부풀려집니다.
#     2) PREDICT_COARSE_PROB은 이 값보다 낮아야 합니다(아래에서 함께 맞춰집니다).
#     3) PREDICT_MIN_EVENT_SPAN_FRAMES도 다시 봅니다.
#     4) 평가 결과 이름에 _th<값>이 붙어 0.5 결과와 구분됩니다. 이미 분석한 영상(DB)은 예전 기준 그대로입니다.
# 코드를 고치지 않고 시험하려면 환경변수 A_THRESHOLD=0.4처럼 줍니다.
# 지금 값 0.45는 서비스 가중치로 검증 영상에서 정했습니다. 검증에서 0.4~0.7이 같은 결과였고, 평가에서는 0.45일 때 오탐 이벤트가 가장 적었습니다.
A_THRESHOLD = float(os.getenv("A_THRESHOLD", "0.45") or 0.45)
# r 증강(학습 전용, 기본 끔)입니다. 클립마다 TRAIN_R_AUGMENT_PROB 확률로 r을 TRAIN_R_AUGMENT_VALUES 중 하나로 골라
# 넓게 자릅니다(그 클립의 모든 프레임에 같은 r). 검증과 서비스는 항상 r=1(R_VALUE)입니다.
# 넓게 자를수록 충돌 전에 다가오는 장면이 많이 들어와 '접근 = 사고'를 배울 위험이 커지므로 값을 1.5~2로 둡니다.
# 켜기: 환경변수 TRAIN_R_AUGMENT_PROB=0.4(이름에 _raug). 값 바꾸기 예: TRAIN_R_AUGMENT_VALUES="1.5,2".
# 확률이 0이면 난수를 쓰지 않아 끈 것과 같은 클립이 나옵니다.
TRAIN_R_AUGMENT_PROB = float(os.getenv("TRAIN_R_AUGMENT_PROB", "0") or 0)
TRAIN_R_AUGMENT_VALUES = tuple(float(v) for v in (os.getenv("TRAIN_R_AUGMENT_VALUES", "1.5,2") or "").split(",") if v.strip())

# ---------- 기본 영상 증강 세기(dataset._apply_augmentation) ----------
# 밝기 하한 0.4는 실제 야간 크롭(낮의 0.4~0.6배 밝기)까지 담으려는 값입니다.
# 흑백은 야간 적외선 CCTV를 흉내 냅니다.
# 노이즈는 실제 압축 영상처럼 클립 내내 같은 잔무늬로 넣습니다. 프레임마다 바뀌는 잡음은 실제 영상에 없고 미세한 흔들림을 덮기 때문입니다.
# 흑백 클립에는 흑백 잔무늬를 넣습니다.
# 나머지는 좌우 뒤집기 50%, 색 흔들기 50%(대비와 채도 0.7~1.3, 색조 -0.1~0.1), 흐림 20%입니다.
TRAIN_AUG_BRIGHTNESS = (0.4, 1.4)     # 색 흔들기(50%) 때 밝기 배율 범위
TRAIN_AUG_GRAY_PROB = 0.2             # 흑백(야간 적외선)
TRAIN_AUG_NOISE_PROB = 0.2            # 고정 잔무늬 노이즈
TRAIN_AUG_NOISE_SIGMA = (2.0, 8.0)    # 잔무늬 세기(0~255)

# ---------- 저화질(압축) 증강(기본 끔) ----------
# 학습 데이터의 저화질 영상이 사고 쪽에 몰려 있어 모델이 압축 깨짐을 사고 단서로 쓰는 것을 막습니다.
# 사고와 정상 구분 없이 이 확률로 클립을 실제 CCTV처럼 H.264로 압축했다 풉니다(CRF 범위, 키프레임 간격 무작위).
# TRAIN_LQ_SIZE_PROB 비율은 먼저 작게 줄였다가 다시 키웁니다.
# 카메라가 찍은 뒤 저장할 때 압축되는 순서에 맞춰 색, 노이즈 증강 뒤에 겁니다.
# 크롭 한 변이 TRAIN_LQ_MIN_CROP보다 작은 클립은 건너뜁니다(TRAIN_LQ_SMALL 참고).
# 켜기: 환경변수 TRAIN_LQ_AUG_PROB=0.3(이름에 _lq). PyAV의 libx264를 쓰며, 없으면 끄고 계속합니다.
TRAIN_LQ_AUG_PROB = float(os.getenv("TRAIN_LQ_AUG_PROB", "0") or 0)
TRAIN_LQ_MIN_CROP = 128              # 원본 프레임 크롭 한 변(px)이 이보다 작으면 적용하지 않습니다.
TRAIN_LQ_SIZE_RANGE = (48, 128)      # 줄일 때 크기(px)입니다. 원래 크롭의 3/4보다 작은 값만 씁니다.
# 줄이는 비율과 CRF 상한입니다. 서비스 가중치는 줄이기 0, CRF 상한 36(약한 저화질 증강)으로 학습했습니다.
# 줄이기가 심하면 저해상도 사고를 더 놓쳤습니다. 기본값(0.5, 42)이 아니면 이름에 _lq 대신 _lqw가 붙습니다.
TRAIN_LQ_SIZE_PROB = float(os.getenv("TRAIN_LQ_SIZE_PROB", "0.5") or 0)   # 압축할 때 먼저 줄이는 비율
TRAIN_LQ_CRF_RANGE = (28, int(os.getenv("TRAIN_LQ_CRF_MAX", "42") or 42))  # H.264 CRF(클수록 압축이 심함)
# 작은 크롭도 저화질 증강(기본 끔, 저화질 증강이 켜져 있어야 씁니다)
# TRAIN_LQ_MIN_CROP보다 작은 크롭은 압축된 모습을 학습한 적이 없어, 압축된 작은 차의 충돌을 놓칠 수 있습니다.
# 켜면 TRAIN_LQ_SMALL_MIN 이상인 작은 크롭도 같은 확률로, 실제 CCTV처럼 원래 크기(짝수)로 줄여 압축한 뒤 다시 키웁니다.
# 줄이는 비율은 쓰지 않고, 사고와 정상 구분이 없으며, CRF 상한은 TRAIN_LQ_SMALL_CRF_MAX입니다.
# 켜기: 환경변수 TRAIN_LQ_SMALL=1(이름의 _lq, _lqw가 _lqs, _lqws로). 끄면 난수를 더 쓰지 않습니다.
TRAIN_LQ_SMALL = os.getenv("TRAIN_LQ_SMALL", "").strip() not in ("", "0")
TRAIN_LQ_SMALL_MIN = 32              # 이보다 작은 크롭은 그대로 둡니다.
TRAIN_LQ_SMALL_CRF_MAX = int(os.getenv("TRAIN_LQ_SMALL_CRF_MAX", "36") or 36)

# ---------- fps 증강(기본 끔) ----------
# 모델은 늘 30프레임을 보므로 10fps 영상에서는 묶음 하나가 3초가 되어 충돌 흔들림이 몇 프레임에 몰립니다.
# 이 확률로 학습 클립을 k장마다 한 장씩 골라(k는 TRAIN_FPS_AUG_STRIDES 중 무작위, 30fps면 15fps나 10fps처럼) 저fps 영상을 흉내 냅니다.
# 사고 클립은 충돌 시작이 원래처럼 묶음의 TRAIN_A_PRE_FRAMES번째 프레임에 오도록 앞쪽을 k배로 늘려 읽습니다.
# 결과 fps가 TRAIN_FPS_AUG_MIN_FPS보다 낮아지는 영상과, 영상이 짧아 k배를 읽지 못하는 클립은 그대로 둡니다.
# 'pre' 학습의 묶음 하나짜리 샘플에만 쓰고, 검증은 그대로입니다. 읽는 프레임이 k배라 클립 만들기가 그만큼 느려집니다.
# 켜기: 환경변수 TRAIN_FPS_AUG_PROB=0.3(이름에 _fps). 확률이 0이면 난수를 쓰지 않습니다.
TRAIN_FPS_AUG_PROB = float(os.getenv("TRAIN_FPS_AUG_PROB", "0") or 0)
TRAIN_FPS_AUG_STRIDES = (2, 3)       # k장마다 한 장(30fps 원본이면 15fps, 10fps)
TRAIN_FPS_AUG_MIN_FPS = 8.0          # 결과 fps가 이보다 낮아지는 k는 쓰지 않습니다.

# ---------- 불빛 증강(기본 끔) ----------
# 야간에 다른 차 불빛이 주차된 차를 훑으면 충돌로 오탐할 수 있습니다. 색 증강은 클립 전체 밝기를 똑같이 바꿔
# 시간에 따라 바뀌는 불빛을 보여 주지 못하므로, 크롭을 어둡게 한 뒤 움직이는 빛 덩어리(훑기)나 서서히 밝아지는 빛(비추기)을
# 몇 프레임에 걸쳐 더합니다. 사고와 정상에 같은 확률로 겁니다. 정상에만 넣으면 '불빛 = 정상'을 배워 전조등을 켠 차의
# 야간 충돌을 놓치기 때문입니다. 불빛은 밝기만 바꾸고 차체 윤곽은 그대로라 흔들림만 판단 단서로 남습니다.
# 장면에 비치는 빛이라 색, 노이즈, 저화질 증강보다 먼저 겁니다. 확률이 0이면 난수를 쓰지 않습니다.
# 켜기: 환경변수 TRAIN_LIGHT_AUG_PROB=0.2(이름에 _light). 효과는 robust_check의 불빛 항목과 야간 영상 성적으로 확인합니다.
TRAIN_LIGHT_AUG_PROB = float(os.getenv("TRAIN_LIGHT_AUG_PROB", "0") or 0)
TRAIN_LIGHT_DARK_PROB = 0.7          # 불빛을 넣을 때 먼저 밤처럼 어둡게 할 확률(배율 0.25~0.6)

# ---------- 야간 영상 목록(눈으로 확인) ----------
# 평가(evaluate)와 postproc_check가 야간 영상만 따로 성적을 냅니다(불빛 오탐 추적용).
# 화면 밝기로 자동 판정하면 모니터를 찍은 낮 영상이나 흐린 낮, 화면 UI가 밝은 야간을 틀리게 가르므로 목록으로 둡니다.
# 새 야간 영상을 넣으면 여기에 이름(확장자 없이)을 더합니다.
NIGHT_VIDEOS = frozenset({
    # 학습: 사고 3, 정상 3
    '260827_ReA_0021', '260827_ReA_0032', '260827_ReA_0044', '260909_ReS_0019', '260909_ReS_0035',
    '260909_ReS_0039',
    # 검증: 사고 1, 정상 3
    '260827_ReA_0031', '260909_ReS_0024', '260909_ReS_0026', '260909_ReS_0036',
    # 평가: 사고 1, 정상 2
    '260828_TA_0020', '260909_TS_0007', '260909_TS_0011',
})

# ---------- 검증 대신 학습에 넣을 영상 ----------
# 파일 이름(확장자 없이)입니다. 검증 폴더 없이 분할할 때, 검증에 뽑힌 이 영상만 학습으로 옮겨 다른 영상의 분할은 바뀌지 않습니다.
# 검증 폴더를 쓸 때는 파일 위치가 곧 분할이므로 검증 폴더에 들어 있으면 알리기만 합니다.
# ReA_0044는 야간 미세 충돌이라 학습에 넣습니다.
TRAIN_FORCE_TRAIN_VIDEOS = ('260827_ReA_0044',)

# ---------- 검증 영상 고정 ----------
# 검증 영상은 VAL_DATA_DIR로 옮겨 폴더로 나눕니다. 이 파일은 옮길 목록(val_move)이자, 아직 옮기지 않은 컴퓨터용입니다.
# VAL_DATA_DIR가 비어 있으면 검증은 파일에 적힌 영상(데이터 폴더 기준 상대경로), 학습은 데이터 폴더의 나머지 전부입니다.
# 새로 넣은 영상은 자동으로 학습 쪽에 들어갑니다. 예전 방식(seed 42 분할)으로 돌리려면 환경변수 TRAIN_SPLIT_LOCK=(빈 값)을 줍니다.
TRAIN_SPLIT_LOCK = (Path(os.environ["TRAIN_SPLIT_LOCK"]) if os.environ.get("TRAIN_SPLIT_LOCK")
                    else None if "TRAIN_SPLIT_LOCK" in os.environ
                    else _ROOT / "model" / "split_lock.json")

# ---------- 실제 화면 갱신이 10fps 미만인 영상 제외 ----------
# 파일 이름(확장자 없이)입니다. 영상에 적힌 fps가 아니라 실제로 화면이 바뀌는 횟수로 잰 값입니다.
# 30fps로 적혀 있어도 같은 장면을 여러 번 반복하는 영상이 있습니다.
# 영상 파일은 LOW_FPS_EXCLUDED_DIR로 옮깁니다: python -m model.lowfps_move --apply
# 아직 옮기지 않은 컴퓨터에서도 같은 결과가 나오도록 학습, 평가, postproc_check, robust_check, fp16_check, val_recheck가
# 이 목록을 한 번 더 뺍니다. 딱 10fps인 영상은 남깁니다. 목록으로 빼는 것만 끄려면 환경변수 EXCLUDE_LOW_FPS=0을 줍니다.
LOW_FPS_EXCLUDED_DIR = _ROOT / "data" / "excluded_lowfps"   # 안에 train/과 eval/이 있고 원래 폴더 구조를 그대로 둡니다.
LOW_FPS_VIDEOS = (
    # 학습 9개
    '260827_ReA_0029', '260827_ReA_0030',
    '260909_ReS_0002', '260909_ReS_0003', '260909_ReS_0004', '260909_ReS_0010',
    '260909_ReS_0016', '260909_ReS_0022', '260909_ReS_0025',
    # 검증 1개
    '260827_ReA_0005',
    # 평가 7개
    '260828_TA_0001', '260828_TA_0002', '260828_TA_0005',
    '260909_TS_0003', '260909_TS_0004', '260909_TS_0006', '260909_TS_0008',
)
EXCLUDE_LOW_FPS = os.getenv("EXCLUDE_LOW_FPS", "1").strip() != "0"
EXCLUDE_VIDEOS = frozenset(LOW_FPS_VIDEOS) if EXCLUDE_LOW_FPS else frozenset()

# ---------- 학습 클립 위치 흔들기 ----------
# 시간 오프셋 흔들기 폭(초)입니다. 프레임이 아니라 초로 두어 fps가 다른 영상에서도 같은 의미를 갖게 합니다.
# 0.33초는 30fps 기준 10프레임입니다.
TRAIN_TIME_JITTER_SEC = 0.33

# ---------- 충돌(A) 클립 슬라이싱(기본 끔) ----------
# 레이블한 충돌 구간 [start_f, end_f] 안에서만 묶음을 밀어 사고 클립을 늘립니다. 구간 밖으로는 나가지 않습니다.
# 켜면 첫 묶음을 뺀 나머지는 모두 충돌 중 장면(접근 장면 없음)이 되어 아래 TRAIN_A_PRE_FRAMES의 이점을 잃습니다.
# 그래서 불균형은 정상 쪽에서 줄이고, 사고는 접근이 담긴 묶음 하나로 두는 편을 기본으로 합니다.
TRAIN_A_SLICE_ENABLED = False
TRAIN_A_SLICE_STRIDE = 15

# ---------- 사고(A) 클립에 접근 장면 담기 ----------
# 묶음을 충돌 시작부터 잡으면 다가오는 장면이 한 프레임도 없어 흔들림만으로 판단하게 됩니다.
# 천천히 움직이는 차와 충돌로 살짝 밀린 차는 모습이 비슷해 오탐의 주원인이 되므로, 논문처럼 접근과 충돌을 함께 담습니다.
# 충돌 구간 길이와 상관없이 충돌 시작보다 이만큼 앞에서 묶음을 시작합니다(접근 15 + 충돌 15).
TRAIN_A_PRE_FRAMES = 15

# ---------- 사고(A) 클립 방식 ----------
# "pre"(기본): 사고 영상 하나에 묶음 하나(충돌 시작 TRAIN_A_PRE_FRAMES 전부터 30프레임)입니다.
#        충돌이 15프레임보다 길면 뒤쪽은 학습에 들어가지 않습니다.
# "span": 접근부터 충돌 끝까지를 30프레임 묶음 여러 개로 덮고 모두 A로 학습하되, 충돌 한 건의 무게 합은 1로 나눕니다.
#        첫 묶음(접근과 충돌 시작, 'pre'의 묶음)은 TRAIN_SPAN_FIRST_WEIGHT, 나머지 묶음은 남은 무게를 똑같이 나눠 갖습니다.
#        구간이 묶음 하나에 들어가면 'pre'와 같습니다. 충돌 끝 쪽 묶음도 A로 배우므로 레이블의 충돌 끝이 정확해야 합니다.
#        모델 구조와 추론(서비스, 평가)은 그대로입니다.
# 학습 명령으로 고릅니다: python -m model.main --mode train --a-mode span(환경변수 TRAIN_A_MODE=span도 같습니다).
# span이면 TRAIN_A_SLICE_ENABLED는 무시합니다.
TRAIN_A_MODE = os.getenv("TRAIN_A_MODE", "pre").strip().lower()
TRAIN_SPAN_FIRST_WEIGHT = 0.5  # 첫 묶음(접근과 충돌 시작)의 무게입니다. 나머지 묶음이 1 - 이 값을 나눕니다.
# 묶음 간격 상한(프레임)입니다. 구간을 이 간격 이하로 고르게 나눠 묶음을 놓습니다(15면 절반씩 겹치고, 30 이하면 빈틈이 없습니다).
TRAIN_SPAN_WINDOW_STRIDE = 15
# 충돌 하나의 최대 묶음 수입니다. 넘으면 간격을 넓혀 이 개수로 구간 전체를 덮습니다.
TRAIN_SPAN_MAX_WINDOWS = 6
# 한 번에 GPU에 올리는 최대 묶음 수입니다. 배치의 묶음 수가 이 값을 넘으면 나눠서 역전파하고 기울기를 합쳐 한 번에 갱신합니다
# (손실은 같고 BN 통계만 나눈 단위로 계산됩니다). None이면 TRAIN_BATCH_SIZE * 2입니다.
# GPU 메모리가 빠듯하면 TRAIN_BATCH_SIZE로 낮춥니다.
TRAIN_SPAN_MAX_WINDOWS_PER_STEP = None

TRAIN_S_SLICE_ENABLED = True
# ---------- 정상(S) 클립 슬라이싱 ----------
# 정상 영상을 이 간격(프레임)으로 잘라 클립을 만듭니다. 30이면 빈틈없이 잘라, 차가 박스 근처를 지나가는 것처럼
# 움직임 있는 순간까지 모두 정상으로 학습합니다. 대신 사고:정상 비율이 약 1:19로 벌어지고 에포크가 길어집니다.
TRAIN_S_SLICE_STRIDE = 30
TRAIN_S_MAX_CLIPS_PER_VIDEO = 0       # 0이면 제한이 없습니다. 영상이 매우 길 때 상한으로 씁니다.
# 실제 영상 정상은 0번 차량 박스와 함께 1번 차량 박스로도 클립을 만듭니다(시작점은 같고 박스만 다름).
# 1번 차량도 주차된 차라, 서비스에서 사용자가 주차된 피해 차량에 박스를 치는 상황과 같은 종류의 정상 샘플이 됩니다.
# 학습과 검증 분할은 영상 단위라 같은 영상의 두 크롭이 학습과 검증으로 갈리지 않습니다. 평가는 0번 박스만 씁니다.
TRAIN_S_REAL_USE_CAR1 = True
# RC 정상 영상도 1번 차량 박스로 정상 클립을 만들지 정합니다(학습 쪽만, 기본 끔).
# 움직이는 RC 차가 1번 차량 옆을 지나가는 장면을 정상으로 넣어 '대상 차 앞으로 다른 차가 지나가는' 오탐을 겨냥합니다.
# RC 정상 클립이 약 2배가 되어 에포크가 길어지고 불균형이 커집니다. 켜기: 환경변수 TRAIN_S_RC_USE_CAR1=1(이름에 _rc1).
TRAIN_S_RC_USE_CAR1 = os.getenv("TRAIN_S_RC_USE_CAR1", "0").strip().lower() in ("1", "true", "yes")
# 사고 영상의 충돌 전(접근)과 후(여파) 장면을 정상으로 학습합니다(학습 쪽만, 기본 끔).
# 충돌 전은 묶음 끝이 충돌 시작보다 TRAIN_A_CONTEXT_NEG_MARGIN_SEC 이상 앞인 묶음, 충돌 후는 묶음 시작이 충돌 끝보다 그만큼 뒤인 묶음입니다.
# 한쪽에 2개씩 넣습니다. 충돌에 가장 가까운 묶음 하나와, 그보다 TRAIN_A_CONTEXT_NEG_STRIDE 이상 먼 곳에서 에포크마다 무작위로 고른 묶음 하나입니다.
# 묶음 위치 흔들기는 이 범위 밖(충돌 쪽)으로 나가지 않고, fps 증강은 걸지 않습니다. 매 에포크 전부 씁니다.
# 기본은 실제 영상만('real', 이름 _ctx)이고, 'all'이면 RC도 씁니다(_ctxall).
# 실험에서는 충돌 밖 A 판정은 줄었지만 사고를 더 놓쳐 서비스 가중치에는 쓰지 않았습니다.
# 켜기: 환경변수 TRAIN_A_CONTEXT_NEG=real(또는 1)이나 all. 비우면 끕니다.
_ctx = os.getenv("TRAIN_A_CONTEXT_NEG", "").strip().lower()
TRAIN_A_CONTEXT_NEG = {"": None, "0": None, "false": None, "no": None, "1": "real", "true": "real", "yes": "real"}.get(_ctx, _ctx)
if TRAIN_A_CONTEXT_NEG not in (None, "real", "all"):
    raise ValueError(f"TRAIN_A_CONTEXT_NEG 는 real · all · 빈 값 중 하나여야 합니다: {_ctx!r}")
TRAIN_A_CONTEXT_NEG_MARGIN_SEC = float(os.getenv("TRAIN_A_CONTEXT_NEG_MARGIN_SEC", "0.5") or 0.5)
TRAIN_A_CONTEXT_NEG_STRIDE = 30

# ---------- 헷갈리는 정상 묶음(하드 네거티브) 추가 학습(기본 끔) ----------
# model/hardneg_mine.py가 학습 영상에서 정상인데 P(A)가 높은 묶음을 찾아 CSV로 냅니다. 켜면 그 묶음을 정상 클립으로
# 학습 데이터셋 맨 뒤에 붙입니다. 검증은 그대로입니다.
# 묶음 위치 흔들기는 찾은 위치에서 TRAIN_HARDNEG_JITTER 프레임 안으로 두어 지나가는 장면이 묶음에 남게 합니다.
# fps 증강은 걸지 않고, balance에서도 솎지 않고 매 에포크 전부 쓰며, TRAIN_HARDNEG_REPEAT번 넣습니다.
# 실험에서는 오탐은 줄었지만 사고를 더 놓쳐 서비스 가중치에는 쓰지 않았습니다.
# 켜기: 환경변수 TRAIN_HARDNEG_CSV=outputs/hardneg/<시각>/hardneg.csv(프로젝트 루트 기준 또는 절대 경로). 이름에 _hn(반복이 1이 아니면 _hn<k>).
_hn = os.getenv("TRAIN_HARDNEG_CSV", "").strip()
TRAIN_HARDNEG_CSV = (Path(_hn) if Path(_hn).is_absolute() else _ROOT / _hn) if _hn else None
TRAIN_HARDNEG_REPEAT = max(1, int(os.getenv("TRAIN_HARDNEG_REPEAT", "1") or 1))
TRAIN_HARDNEG_JITTER = 5

# ---------- 학습 배치 구성: 빈 장면 정상 솎기와 배치마다 충돌 고르게 넣기(기본 끔) ----------
# 정상 클립의 절반가량이 빈 장면이고, 무작위로 섞으면 배치 절반가량에 충돌이 없습니다.
# 빈 장면도 서비스의 긴 정지 구간을 배우는 데 필요하므로 버리지 않고 일부씩 돌려 씁니다.
# TRAIN_S_MOTION_CSV: s_motion_check 결과 clips.csv입니다. 비우면 솎기를 끕니다. 학습 쪽 정상만 솎고 검증은 그대로입니다.
#     빈 장면은 움직임 < TRAIN_S_EMPTY_MOTION_PCT인 클립입니다. 매 에포크 TRAIN_S_EMPTY_KEEP만큼만 쓰되
#     에포크마다 다음 구간을 써서 1/KEEP 에포크마다 전부 한 번씩 봅니다. CSV에 없는 정상 클립은 움직임 있음으로 보고 매번 씁니다.
#     기준 0.25는 충돌 클립이 거의 걸리지 않는 상한입니다.
# TRAIN_SPREAD_A: 충돌 클립을 배치마다 고르게 나눠 넣습니다(복제 없음). 충돌이 배치 수 이상이면 모든 배치에 충돌이 들어갑니다.
# 켜는 법: python -m model.main --mode train --balance outputs/s_motion/<시각>/clips.csv
#     (둘 다 켭니다. 구성만 보려면 --dry-run, 환경변수 TRAIN_S_MOTION_CSV와 TRAIN_SPREAD_A=1로도 됩니다.)
#     켜면 회차 폴더와 가중치 이름에 _bal이 붙습니다. 상대경로는 프로젝트 루트 기준입니다.
# 정상이 줄어 P(A)가 전반적으로 조금 오르므로, 학습 뒤 평가에서 오탐과 판정 기준을 다시 봅니다.
TRAIN_S_MOTION_CSV = os.getenv("TRAIN_S_MOTION_CSV", "").strip() or None
TRAIN_S_EMPTY_MOTION_PCT = 0.25
TRAIN_S_EMPTY_KEEP = 1 / 3
TRAIN_SPREAD_A = os.getenv("TRAIN_SPREAD_A", "0").strip().lower() in ("1", "true", "yes")
# RC 정상도 빈 장면 솎기를 할지 정합니다(기본 1, 솎음). 0이면 RC 정상은 움직임과 상관없이 매 에포크 전부 쓰고
# 실제 영상 정상만 솎습니다(이름에 _rcfull). 빈 장면 솎기(TRAIN_S_MOTION_CSV)를 켰을 때만 의미가 있습니다.
TRAIN_S_THIN_RC = os.getenv("TRAIN_S_THIN_RC", "1").strip().lower() not in ("0", "false", "no")

# 클래스 가중치(CrossEntropyLoss weight)입니다. 정상 슬라이싱으로 불균형이 커지면 모델이 정상 쪽으로 치우쳐 놓침이 늘 수 있습니다.
# None이면 쓰지 않습니다(기본). 검증 없이 켜면 값에 따라 오탐이 크게 늘 수 있습니다.
# 학습 초반 검증 재현율이 0에 오래 머물면 역빈도의 제곱근(약 1:19면 4.3)부터 켜 봅니다.
# 환경변수 TRAIN_CLASS_WEIGHTS: 'auto'는 (1, sqrt(에포크당 정상 수 / 충돌 수))이며 balance를 켜면 솎은 뒤 수를 씁니다.
# 숫자 하나 '3.6'은 (1, 3.6), 두 개 '1,3.6'은 그대로입니다. 켜면 이름에 _cw<사고 배수>가 붙습니다.
# 검증의 균형 손실과 판단 지표는 가중치 없는 손실이라 다른 학습과 그대로 비교됩니다.
def _class_weights(v):
    v = (v or '').strip().lower()
    if not v:
        return None
    if v == 'auto':
        return 'auto'
    xs = [float(x) for x in v.split(',') if x.strip()]
    return (1.0, xs[0]) if len(xs) == 1 else (xs[0], xs[1])


TRAIN_CLASS_WEIGHTS = _class_weights(os.getenv("TRAIN_CLASS_WEIGHTS", ""))
TARGET_ID = 0
USE_AMP = True
# GPU 메모리 배치 channels_last_3d입니다(학습, 평가, 서비스 공통, GPU일 때만).
# X3D는 RTX 3080에서 끄는 편이 학습 약 2.3배, 검증 약 2.6배 빨랐고 결과는 같았습니다.
# GPU나 모델을 바꾸면 python -m model.speed_check로 다시 잽니다.
USE_CHANNELS_LAST = False

# ---------- 학습 속도(결과는 같습니다) ----------
# 검증 클립 저장본입니다. 검증은 증강 없이 매 에포크 같은 클립을 쓰므로, 처음 만든 클립(자르고 줄인 uint8)을 저장해 두고
# 다음 에포크부터 읽습니다. 검증 입력이 비트 단위로 같고 영상 풀기 시간이 사라집니다.
# 영상 파일, 시작 프레임, 박스, r, 크기, 클립 만드는 코드가 바뀌면 새로 만듭니다. 예전 저장본은 자동으로 지우지 않으며,
# 공간이 필요하면 폴더째 지워도 됩니다(다음 학습 때 다시 만듭니다). 디스크가 차면 저장 없이 계속합니다.
TRAIN_VAL_CACHE = os.getenv("TRAIN_VAL_CACHE", "1").strip().lower() not in ("0", "false", "no")
TRAIN_VAL_CACHE_DIR = _ROOT / "data" / ".clip_cache" / "val"
# 데이터 로더 작업자마다 쓰는 OpenCV 스레드 수(영상 풀기, 색 변환, 크기 줄이기)입니다. None이면 OpenCV 기본값(코어 수)입니다.
# 작업자가 많으면 스레드가 코어보다 많아져 서로 기다릴 수 있습니다. python -m model.speed_check로 기본과 1을 비교해
# 빠를 때만 1로 둡니다(환경변수 TRAIN_CV2_THREADS=1). 화소는 같습니다.
TRAIN_CV2_THREADS = int(os.getenv("TRAIN_CV2_THREADS", "0") or 0) or None
# 학습 로더 작업자 수입니다. None이면 min(8, CPU 코어)입니다. 검증 로더는 저장본을 읽어 가벼우므로 그대로 둡니다.
# 작업자를 늘릴 때는 TRAIN_CV2_THREADS=1과 함께 쓰고 speed_check로 확인한 뒤 켭니다.
# 작업자마다 미리 만들어 둘 배치는 전체 약 32개가 되게 줄입니다(train.loader_prefetch, 메모리 때문).
TRAIN_NUM_WORKERS = int(os.getenv("TRAIN_NUM_WORKERS", "0") or 0) or None

# ---------- 사전학습과 입력 정규화 ----------
# 백본은 pytorchvideo X3D-M이고, 학습할 때 Kinetics-400 사전학습 가중치로 초기화합니다.
# 사전학습은 초기값만 바꾸므로 모델 크기와 추론 속도는 같습니다.
# 사전학습을 쓰지 않으려면 학습 명령에 --scratch를 줍니다(환경변수 PRETRAINED=0도 같습니다).
# scratch면 백본도 분류 헤드와 같은 학습률로 학습하고(사전학습일 때는 백본 0.1배), 기본 학습률은 TRAIN_SCRATCH_LEARNING_RATE입니다.
PRETRAINED = os.getenv("PRETRAINED", "1").strip().lower() not in ("0", "false", "no")
# 결과물이 서로 덮어쓰이지 않도록 사전학습 여부를 파일 이름에 넣습니다(ptY, ptN).
PRETRAIN_TAG = "ptY" if PRETRAINED else "ptN"
# 입력 정규화 통계는 Kinetics-400 사전학습과 같은 값입니다(pytorchvideo의 video_mean, video_std).
# 학습과 추론이 반드시 같은 값을 써야 하므로 여기서 함께 관리합니다.
NORM_MEAN = (0.45, 0.45, 0.45)
NORM_STD = (0.225, 0.225, 0.225)

# ---------- 학습 전용 ----------
# 최종 가중치를 둘 폴더입니다(이 경로의 폴더만 쓰고 파일 이름은 쓰지 않습니다). 학습 중 best와 후보는
# 회차 폴더(outputs/trainlogs/<회차>/work_best.pth, work_cand_ep*.pth)에 저장하고, 학습이 끝나면 아래 규칙의 이름으로
# 이 폴더에 옮깁니다(train.py). 회차마다 따로라 같은 조건의 학습을 동시에 돌려도 섞이지 않습니다.
# hitandrun_[모델]_[YYMMDD_HHMM]_[N]ep_[earlyY|N]_[ptY|N]_[손실]
#     [YYMMDD_HHMM]: 학습 시작 시각(회차 폴더 이름의 시각), [N]ep: best 에포크,
#     [earlyY|N]: 조기 종료 여부, [손실]: best를 저장할 때의 판단 지표 값
TRAIN_BEST_MODEL_SAVE_PATH = (
    _ROOT / "weights" / f"hitandrun_{MODEL_NAME}_{PRETRAIN_TAG}_best.pth")
TRAIN_BATCH_SIZE = 15  # 논문(Hwang & Lee 2024)과 같은 배치 15입니다.
# 배치가 작으면 배치에 사고 클립이 하나도 없는 일이 잦아지고, 그런 배치에서는 사고 학습 신호가 없습니다.
TRAIN_NUM_EPOCHS = 100
TRAIN_SPLIT_RATIO = 0.8
TRAIN_EARLY_STOPPING_PATIENCE = 10
# 유예 기간(에포크)입니다. 처음 이만큼은 조기 종료를 세지 않고, 학습률을 줄이지 않고, best와 후보를 고르지 않습니다
# (에포크별 가중치와 학습 CSV는 그대로 남고, suggest도 이 기간은 추천하지 않습니다). 지나면 그때부터 새로 셉니다.
# scratch는 불균형 때문에 먼저 '전부 정상'을 배우며 판단 지표가 한동안 나빠지므로 이 기간을 둡니다.
# 사전학습 모델은 0입니다. 바꾸려면 환경변수 TRAIN_GRACE_EPOCHS를 줍니다(비우면 scratch는 TRAIN_SCRATCH_GRACE_EPOCHS, 사전학습은 0).
TRAIN_SCRATCH_GRACE_EPOCHS = 20
_grace_env = os.getenv("TRAIN_GRACE_EPOCHS", "").strip()
TRAIN_GRACE_EPOCHS = int(_grace_env) if _grace_env else None

# ---------- best 판단 지표와 후보 저장 ----------
# best 저장, 학습률 감소(ReduceLROnPlateau), 조기 종료가 모두 판단 지표 하나로 움직입니다.
# 판단 지표는 val_real의 클래스 균형 손실((충돌 평균 손실 + 정상 평균 손실) / 2)을 최근 TRAIN_MONITOR_SMOOTH 에포크 평균한 값입니다.
# 검증 클립 대부분이 정상이라 그냥 손실은 정상을 얼마나 확신하느냐로 정해지므로 균형 손실을 씁니다.
# 한 에포크 값은 흔들려 운 좋게 한 번 낮은 에포크가 뽑힐 수 있어 평균하며, 대신 약 1에포크 늦게 반응합니다.
TRAIN_MONITOR_SMOOTH = 3
# 후보는 판단 지표 상위 K개를 저장하되 서로 MIN_GAP 에포크 이상 떨어진 것만 남깁니다(붙어 있는 에포크는 사실상 같은 모델).
# 1위는 best와 같습니다. 최종 선택은 학습 뒤 검증 영상 서비스 판정(epoch_rank)과 평가로 합니다.
TRAIN_TOPK_CANDIDATES = 3
TRAIN_TOPK_MIN_GAP = 3
# 매 에포크 가중치를 회차 폴더 epochs/에 남깁니다(파일 하나 약 12MB).
# 학습이 끝나면 평가해 볼 에포크를 추천하고(model/epoch_suggest.py), epoch_rank로 검증 영상 서비스 판정 순위를 매깁니다.
# 학습 중이나 중단한 뒤에도: python -m model.main --mode suggest --run outputs/trainlogs/<회차 폴더>
TRAIN_SAVE_ALL_EPOCHS = True
TRAIN_SUGGEST_TOPK = 2
TRAIN_LEARNING_RATE = 0.00003  # X3D-M 미세조정 학습률(헤드 기준이며, 백본은 train.py에서 0.1배인 3e-6)
# scratch(무작위 초기화) 학습률입니다. 모든 파라미터에 같은 값(AdamW)을 쓰며, 검증된 값이 아니라 출발점입니다.
# 서버에서 1e-5, 1e-4, 1e-3을 짧게(약 10에포크) 돌려 val_real 균형 손실이 가장 잘 내려가는 값을 고릅니다(--scratch --lr 1e-3).
TRAIN_SCRATCH_LEARNING_RATE = 0.0001
# AdamW의 decoupled weight decay입니다. Adam에 weight_decay를 주면 L2 페널티가 적응적 학습률 때문에 파라미터마다 달라지므로
# 정규화가 목적이면 AdamW를 씁니다.
# decoupled weight decay는 학습률과 곱해지는데, 학습률이 작아(헤드 3e-5, 백본 3e-6) 0.01로는 정규화가 거의 걸리지 않아 0.1로 둡니다.
TRAIN_WEIGHT_DECAY = 0.1
# 학습과 평가 결과는 실행마다 '<YYMMDD_HHMM>_<모델>_<조건>' 회차 폴더를 새로 만들어 모읍니다(model/runlog.py).
# 학습: outputs/trainlogs/<회차>/에 학습 곡선 CSV, 그래프, 결과 json, 그때의 config 사본, 학습과 검증 영상 목록(split.json)을 둡니다.
# 가중치는 weights/에 같은 시각을 붙여 저장합니다.
TRAIN_LOG_DIR = _ROOT / "outputs" / "trainlogs"
# 평가: outputs/evallogs/<회차>/에 화면 전문 로그와 영상별 결과 CSV를 둡니다.
# 모델 비교용 요약은 outputs/evallogs/eval_summary.csv 한 파일에 계속 덧붙입니다.
EVAL_LOG_DIR = _ROOT / "outputs" / "evallogs"

# ---------- 웹 서비스(백엔드 Celery 워커) 전용 ----------
# 백엔드 prediction_job이 불러오는 서비스 가중치입니다. X3D-M 구조의 .pth여야 합니다.
# balance, 약한 저화질 증강, 상위 3개 풀링(top3)으로 학습한 회차(261121_1915)의 26에포크입니다.
# 판정 기준 0.45에서 평가 25개 중 21개를 맞혔습니다(사고 18/18, 정상 오탐 영상 4/7).
# top3 풀링이라 헤드 풀링은 가중치 이름을 따라갑니다(서버 .env의 HEAD_TEMPORAL_POOL은 비워 둡니다).
# 되돌릴 때 쓸 이전 서비스 가중치: hitandrun_x3d_261121_1915_ep039_ptY_bal_lqw_top3.pth
SERVICE_WEIGHTS_PATH = _ROOT / "weights" / "hitandrun_x3d_261121_1915_ep026_ptY_bal_lqw_top3.pth"
# 서버에서 코드를 고치지 않고 바꿀 때는 .env에 SERVICE_WEIGHTS=weights/<파일>.pth를 줍니다(상대경로는 프로젝트 루트 기준).
if os.getenv("SERVICE_WEIGHTS", "").strip():
    SERVICE_WEIGHTS_PATH = Path(os.environ["SERVICE_WEIGHTS"].strip())
    if not SERVICE_WEIGHTS_PATH.is_absolute():
        SERVICE_WEIGHTS_PATH = _ROOT / SERVICE_WEIGHTS_PATH

# ---------- 영상 하나 예측과 CAM 출력 전용 ----------
PREDICT_WEIGHTS_PATH = SERVICE_WEIGHTS_PATH
PREDICT_VIDEO_PATH = _ROOT / "data" / "eval" / "real01.mp4"
PREDICT_TXT_PATH = _ROOT / "data" / "eval" / "real01.txt"
PREDICT_OUTPUT_DIR = _ROOT / "data" / "predict_cam_result"
# 서비스 3D-CNN에 묶음을 몇 개씩 넣을지(배치 크기) 정합니다. 장치마다 빠른 값이 달라 따로 둡니다(device_utils.predict_batch_size).
# 판정 결과는 같습니다(소수점 끝자리만 조금 다를 수 있습니다).
# CPU(OpenVINO 등)는 묶음 하나만으로 코어를 다 써서 배치를 키울수록 오히려 느립니다.
# GPU는 한 번에 많이 넣을수록 효율이 올라 평가와 같은 8을 씁니다.
PREDICT_INFER_BATCH_SIZE_CPU = 1
PREDICT_INFER_BATCH_SIZE_GPU = 8
# 슬라이딩 윈도우 간격(프레임)입니다. 서비스는 GPU 추론을 가정하므로 모든 위치를 봅니다(1).
# 아래 PREDICT_MIN_EVENT_SPAN_FRAMES(35)는 간격 1 기준 값이므로, 간격을 바꾸면 함께 다시 봐야 합니다.
PREDICT_WINDOW_STRIDE = 1
# 클립에 판정(S/A)과 CAM을 그리는 프레임은 묶음 시작 + PREDICT_DISPLAY_OFFSET(0 ~ CLIP_LENGTH-1)입니다.
# 모델은 충돌 약 25프레임 전부터 시작하는 묶음을 처음 A로 판정하므로 25를 씁니다.
# 서비스 클립의 A 띠(빨간 박스, CAM)는 아래 시작, 끝 위치를 따로 쓰고, 이 값은 CAM을 고르는 위치,
# 짧은 반응의 대체, CLI 전체 영상(predict_full_video)에 씁니다. 이벤트 구간, DB 기록, 충격 시점 계산에는 영향이 없습니다.
PREDICT_DISPLAY_OFFSET = 25
# 서비스 클립의 A 띠는 (이벤트 안 첫 A 묶음 시작 + START)부터 (마지막 A 묶음 시작 + END)까지입니다.
# 첫 A 묶음은 충돌 시작 약 30프레임 전에서 시작하지만, 마지막 A 묶음은 충돌 끝 뒤에서 시작합니다(충돌 직후 흔들림이 담긴 묶음도 A).
# 그래서 양 끝에 같은 값을 쓰면 끝이 늦으므로 시작 +29, 끝 +0을 씁니다. 검증 영상에서 이 값이 충돌 구간과 가장 잘 맞았습니다.
# 반응이 짧아 띠가 PREDICT_DISPLAY_MIN_SEC보다 짧아지면(시작 > 끝 포함) +25/+25로 그려 잠깐 번쩍이고 끝나지 않게 합니다.
# 충돌 앞 접근이나 뒤 여파를 길게 A로 본 영상은 여전히 몇 초 어긋날 수 있습니다.
PREDICT_DISPLAY_START_OFFSET = 29
PREDICT_DISPLAY_END_OFFSET = 0
PREDICT_DISPLAY_MIN_SEC = 0.5
# 서비스 사고 클립에 CAM 히트맵을 합성할지 정합니다. 박스 전체가 아니라 판정 근거가 강한 곳(충돌 부위)에만 칠합니다
# (inference._draw_collision_cam). 차량 박스(사고 빨강, 정상 초록)와 아래 상태 띠(S/A)는 그대로 그립니다.
# 히트맵 재료는 판정과 같은 추론에서 함께 나와(infer_runtime의 with_cam) 켜도 분석 속도는 거의 같습니다.
# CLI 전체 영상 출력(predict_full_video)은 이 설정과 무관하게 박스 전체 CAM을 그립니다.
PREDICT_CLIP_CAM_OVERLAY = True
# 히트맵을 칠하는 곳은 박스 안 칸별 'A - S' 로짓이 가장 강한 값의 이 비율 이상인 곳입니다(그 아래는 원본 그대로).
# 낮추면 넓게, 높이면 가장 강한 곳만 좁게 칠합니다. 0.7이면 충돌 지점에 작은 고리 모양으로 칠해집니다.
PREDICT_CLIP_CAM_MIN = 0.7
# 가장 강한 곳(가운데)의 히트맵 불투명도(0~1)입니다. 기준선에서 0으로 시작해 안쪽으로 갈수록 진해집니다.
PREDICT_CLIP_CAM_ALPHA = 0.9
# 기준선에서 가운데까지 진해지는 빠르기입니다. 1이면 고르게, 작을수록 안쪽으로 빨리 진해져 고리가 또렷해집니다.
PREDICT_CLIP_CAM_GAMMA = 0.6
# 사고 클립 앞뒤로 붙이는 여유(프레임)입니다. 사고 판정 구간 앞뒤로 이만큼 더 잘라 보여 줍니다(30fps 기준 1초).
PREDICT_CLIP_PAD_FRAMES = 30

# ---------- 2단계 탐색: 모델로 거친 탐색 뒤 의심 구간만 촘촘히 ----------
# 긴 영상의 모든 묶음(간격 1)을 3D-CNN에 넣으면 느리므로 이렇게 합니다.
#     1) PREDICT_COARSE_STEP 프레임 간격의 묶음만 먼저 모델로 봅니다(전체의 1/10).
#     2) P(A) > PREDICT_COARSE_PROB인 묶음 주변 PREDICT_COARSE_PAD 프레임 안의 묶음을 모두 봅니다.
#     3) 끝까지 보지 않은 묶음은 충돌이 아닌 것으로 봅니다.
# 광학흐름으로 고르면 움직임이 작은 미세 충돌이 걸러지므로 모델로 고릅니다. 그래서 모델이 잡을 수 있는 충돌이 탐색 때문에 빠지지 않습니다.
# 연속 A 묶음이 10개 이상인 충돌은 거친 탐색 묶음을 반드시 하나 품으므로 놓치지 않습니다.
# 긴 영상에서 실제로 몇 %를 보는지는 서버 로그의 '[2단계] 거친 탐색' 줄로 확인합니다.
PREDICT_USE_COARSE_SCAN = True
PREDICT_COARSE_STEP = 10       # 거친 탐색 간격(프레임, 30fps 기준이며 PREDICT_FPS_SCALED_POSTPROC가 fps에 맞춥니다)
# 이 P(A)를 넘는 거친 묶음 주변을 촘촘히 봅니다. 약한 신호도 줍도록 A_THRESHOLD보다 낮게 둡니다.
# 기본은 min(0.3, 0.6 * 판정 기준)이라 판정 기준을 0.5 아래로 내리면 함께 내려갑니다. 직접 정하려면 환경변수 PREDICT_COARSE_PROB를 줍니다.
PREDICT_COARSE_PROB = float(os.getenv("PREDICT_COARSE_PROB", "") or min(0.3, 0.6 * A_THRESHOLD))
PREDICT_COARSE_PAD = CLIP_LENGTH  # 의심 묶음 주변 이 범위(프레임) 안의 묶음을 모두 평가합니다.
PREDICT_FLOW_SAMPLE_STEP = 2   # 충격 시점 계산에 쓰는 광학흐름 프레임 간격(속도용)

# ---------- 이벤트 후처리(분할 방지와 깜빡임 제거) ----------
# 모델 판정이 A, S, A로 깜빡이면 한 충돌이 여러 이벤트로 쪼개지므로 병합한 뒤 길이 필터를 겁니다.
#     1) 간격이 가까운 이벤트는 같은 사고로 보고 병합합니다.
#     2) 그래도 너무 짧은(단발 깜빡임) 이벤트는 지웁니다.
# 병합 간격은 초로 정하고 영상 fps로 프레임으로 바꿉니다. 프레임으로 고정하면 fps에 따라 실제 시간이 크게 달라지기 때문입니다.
PREDICT_EVENT_MERGE_GAP_SEC = 3.0
# 최소 길이입니다. 이벤트 길이는 28 + (A로 판정된 연속 묶음 수)입니다(간격 1 기준).
# 묶음이 30프레임이라 A 묶음이 하나만 있어도 29프레임이므로, 29 이하로 두면 필터를 끈 것과 같습니다.
# 이벤트 길이는 충돌이 실제로 이어진 시간이 아닙니다. 짧은 충돌도 그 장면을 담는 묶음이 30개라 모델이 잡으면 여러 묶음이 연속으로 A가 됩니다.
# 진짜 충돌은 연속 A 묶음이 수십 개 이상이고 단발 오탐은 몇 개뿐이라, 약하게 잡힌 미세 충돌을 놓치지 않도록 여유를 둔 35(연속 7개)를 씁니다.
# 연속 묶음 수 부분(35 - 28 = 7)은 30fps 기준이라 아래 PREDICT_FPS_SCALED_POSTPROC가 fps에 맞춥니다.
PREDICT_MIN_EVENT_SPAN_FRAMES = 35
# 거친 탐색 간격과 최소 길이를 영상 fps에 맞출지 정합니다(inference.fps_scaled_event_params).
# PREDICT_COARSE_STEP(10)과 최소 길이의 연속 A 묶음 수(7)는 30fps 기준이라, 그대로 두면 10fps 영상에서는 거친 탐색 간격이 1초가 되고
# 모델이 반응한 충돌도 연속 묶음이 모자라 지워집니다. 켜면 둘을 fps/30배로 바꿉니다(최소 길이의 28은 묶음이 늘 30프레임이라 그대로).
#     10fps는 간격 3, 연속 2묶음(최소 길이 30), 15fps는 5, 4묶음(32), 24~25fps는 8, 6묶음(34), 60fps는 20, 14묶음(42)입니다.
# 병합 간격은 이미 초 단위입니다. 끄면 프레임 고정과 같습니다.
PREDICT_FPS_SCALED_POSTPROC = True
PREDICT_POSTPROC_REF_FPS = 30.0    # 위 프레임 값들의 기준 fps

# ---------- 추론 실행 방식(서비스와 평가 공통, infer_runtime.py) ----------
# 서버 컴퓨터 사양이 정해져 있지 않아, 그 컴퓨터에서 쓸 수 있는 방식을 처음 한 번 자가 검사해 가장 빠른 것을 고릅니다.
# PyTorch와 결과(P(A))가 같은 방식만 통과합니다(FP32 그대로).
#     auto(기본), torch, openvino, onnxruntime 중에서 고릅니다(환경변수 INFER_BACKEND로도 지정할 수 있습니다).
# 모델이 GPU(CUDA)에 있으면 auto는 PyTorch GPU를 씁니다. Intel CPU에서는 OpenVINO가 PyTorch보다 약 3~4배 빠릅니다.
# openvino나 onnxruntime이 설치되지 않았거나 변환, 검사에 실패하면 다음 방식(마지막은 PyTorch)으로 넘어갑니다.
# 검사 결과와 변환 파일은 INFER_CACHE_DIR에 저장하고, 가중치, 패키지 버전, CPU와 GPU, OS가 같으면 다음부터는
# 속도를 다시 재지 않고 워커를 띄울 때마다 묶음 하나로 결과만 빠르게 확인합니다.
# 처음 한 번의 전체 검사를 미리 해 두려면 python -m model.main --mode check-backend를 실행합니다.
INFER_BACKEND = os.getenv("INFER_BACKEND", "auto").strip().lower()
# GPU FP16(autocast)입니다. Tensor Core가 있는 NVIDIA GPU(연산 능력 7.0 이상)에서만 켜지고 자가 검사도 거칩니다
# (그 밖의 GPU와 CPU는 FP32 그대로). RTX 3080에서 P(A) 차이가 판정을 바꾸지 않았고 평가 지표도 같았으며 서비스 경로가 약 1.2배 빨랐습니다.
# 다른 GPU로 옮기면 python -m model.fp16_check로 다시 확인합니다. 끄기: 환경변수 INFER_FP16=0.
INFER_FP16 = os.getenv("INFER_FP16", "1").strip().lower() in ("1", "true", "yes")
INFER_CHECK_MAX_PROB_DIFF = 1e-3   # 자가 검사에서 FP32 방식들의 P(A) 허용 차이
INFER_FP16_MAX_PROB_DIFF = 1e-2    # 자가 검사에서 FP16의 P(A) 허용 차이
INFER_CACHE_DIR = _ROOT / "weights" / ".infer_cache"   # 변환 파일(ONNX)과 자가 검사 결과

# ---------- 실제 영상 정확도 평가 전용 ----------
# 평가할 가중치입니다. 여러 학습 결과를 비교할 때는 환경변수로 고릅니다.
#     EVAL_WEIGHTS=weights/<가중치>.pth python -m model.main --mode eval
EVAL_WEIGHTS_PATH = (Path(os.environ["EVAL_WEIGHTS"]) if os.getenv("EVAL_WEIGHTS")
                     else SERVICE_WEIGHTS_PATH)
EVAL_FOLDER_PATH = _ROOT / "data" / "eval"
# 서비스 경로 평가에서 검출로 인정하는 시작 시점 허용 오차(초)입니다.
# 이벤트가 레이블 start_f에서 이 값 안에 시작해야 검출로 셉니다. 이벤트가 나오더라도 엉뚱한 곳이면 검출이 아닙니다.
EVAL_HIT_TOLERANCE_SEC = 1.0
EVAL_INFER_BATCH_SIZE = 8 # 평가 추론 배치 크기
EVAL_WINDOW_STRIDE = 1  # 기본 1(올리면 빨라지지만 정확도가 조금 떨어집니다)
