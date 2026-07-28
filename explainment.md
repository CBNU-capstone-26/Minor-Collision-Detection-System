# Hit-and-Run Detection Model — 구현 설명서

> 주차장 내 물피도주 사고를 CCTV 영상에서 인식하는 3D-CNN 기반 모델의 전체 구성과 학습 과정을 설명합니다.

---

## 목차

1. [모델 아키텍처](#1-모델-아키텍처)
2. [데이터 전처리](#2-데이터-전처리)
3. [학습 과정](#3-학습-과정)
4. [추론 및 CAM 시각화](#4-추론-및-cam-시각화)
5. [성능 평가](#5-성능-평가)
6. [하이퍼파라미터 통합표](#6-하이퍼파라미터-통합표)
7. [학습 안정화 — 추가/변경/보류 기법](#7-학습-안정화--추가변경보류-기법)
8. [S3D 전이학습 전환 & 데이터 증강 보강](#8-s3d-전이학습-전환--데이터-증강-보강)

---

## 1. 모델 아키텍처

> 파일: `model/hitandrun_model.py`

### 1-1. 전체 구조

I3D(Inflated 3D ConvNet) 기반의 GoogLeNet 구조를 채택합니다.  
2D 이미지 분류 네트워크의 Conv2d·Pool2d를 시간 축으로 팽창(inflate)시켜 **공간 특징과 시간 특징을 동시에** 학습합니다.

```
입력 (B, 3, 30, 224, 224)
  │
  ├── Conv1 (7×7×7, stride 2×2×2) → (B, 64, 15, 112, 112)
  ├── MaxPool1 (1×3×3, stride 1×2×2)  → (B, 64, 15, 56, 56)
  ├── Conv2 (1×1×1 → 3×3×3)           → (B, 192, 15, 56, 56)
  ├── MaxPool2 (1×3×3, stride 1×2×2)  → (B, 192, 15, 28, 28)
  │
  ├── [Inception Block 1]
  │     Inception3a → Inception3b → MaxPool3 (stride 2×2×2)
  │     (B, 480, 8, 14, 14)
  │
  ├── [Inception Block 2]
  │     Inception4a → 4b → 4c → 4d → 4e → MaxPool4 (stride 1×2×2)
  │     (B, 832, 8, 7, 7)
  │
  ├── [Inception Block 3]
  │     Inception5a → Inception5b     → (B, 1024, 8, 7, 7)
  │
  ├── AdaptiveAvgPool3d((1,1,1))       → (B, 1024, 1, 1, 1)
  ├── Dropout(p=0.4)
  └── head_conv: Conv3d(1024 → 2, 1×1×1) → squeeze → (B, 2)
```

### 1-2. InceptionModule3D

GoogLeNet의 Inception 모듈을 3D로 확장한 구조입니다.  
**4개의 병렬 경로**를 통해 서로 다른 시공간 스케일의 특징을 동시에 추출하고, 채널 축으로 연결(concat)합니다.

```
입력
 ├── Branch 1: Conv3d(1×1×1)                          → n1x1 채널
 ├── Branch 2: Conv3d(1×1×1) → Conv3d(3×3×3, pad=1)  → n3x3 채널
 ├── Branch 3: Conv3d(1×1×1) → Conv3d(3×3×3, pad=1)  → n5x5 채널  ← 원 논문의 5×5를 3×3으로 대체
 └── Branch 4: MaxPool3d(3×3×3) → Conv3d(1×1×1)      → pool_proj 채널
출력: 위 4개 채널 합산 (concat)
```

**채널 수 진행표:**

| 레이어 | 입력 채널 | Branch 출력 (1×1 / 3×3 / 5→3×3 / pool) | 출력 채널 |
|---|---|---|---|
| inception3a | 192 | 64 / 128 / 32 / 32 | 256 |
| inception3b | 256 | 128 / 192 / 96 / 64 | 480 |
| inception4a | 480 | 192 / 208 / 48 / 64 | 512 |
| inception4b | 512 | 160 / 224 / 64 / 64 | 512 |
| inception4c | 512 | 128 / 256 / 64 / 64 | 512 |
| inception4d | 512 | 112 / 288 / 64 / 64 | 528 |
| inception4e | 528 | 256 / 320 / 128 / 128 | 832 |
| inception5a | 832 | 256 / 320 / 128 / 128 | 832 |
| inception5b | 832 | 384 / 384 / 128 / 128 | 1024 |

### 1-3. head_conv 설계 (Linear 대신 Conv3d)

분류 헤드를 `nn.Linear` 대신 `nn.Conv3d(1024, num_classes, kernel_size=1)`로 구성합니다.

- `AdaptiveAvgPool3d` → `Dropout` → `Conv3d(1×1×1)` → `squeeze()` 순서로 처리
- `inception5b`의 피처맵 공간 구조가 flatten 없이 보존됨
- CAM 가중치(`head_conv.weight`)와 피처맵을 직접 곱해 **히트맵 생성이 가능**

### 1-4. 장점 / 단점

| | 내용 |
|---|---|
| **장점** | 시공간 특징을 단일 forward pass에서 동시 학습 |
| | Inception 병렬 분기로 다양한 수용 영역(receptive field) 확보 |
| | head_conv 구조로 CAM 시각화 지원 — 모델 해석 가능성 제공 |
| | 1×1 Conv로 채널 병목을 만들어 연산량 절감 |
| **단점** | 3D Conv 특성상 메모리·연산량이 2D CNN 대비 3~5배 |
| | 사전 학습(pre-trained) 가중치 미사용으로 수렴 속도 느림 |
| | 파라미터 수 약 2,500만 개 — 소규모 데이터셋에서 과적합 위험 |

---

## 2. 데이터 전처리

> 파일: `model/dataset.py`

### 2-1. 어노테이션 파싱

각 영상과 쌍을 이루는 `.txt` 파일에서 차량 정보와 이벤트 레이블을 읽습니다.

```
# txt 파일 포맷 예시
car,0,120,80,300,220      → 차량 ID=0, bbox (x1,y1,x2,y2)
car,1,450,100,620,260     → 차량 ID=1, bbox
A,0,45                    → Accident, 타겟 차량 ID=0, 충돌 시작 프레임=45
S,0,0                     → Safe (정상)
```

- `A` → label=1 (충돌), `S` → label=0 (정상)
- 타겟 차량의 bbox와 충돌 시작 프레임(start_f)을 메타데이터로 저장

### 2-2. 정사각형 크롭 + 패딩 (`_crop_and_pad`)

타겟 차량을 화면 중앙에 놓고, 정사각형 영역으로 잘라냅니다.

```
r_value = wc / wv    (wc: 크롭 폭, wv: 차량 bbox 폭)

1. 차량 bbox 중심점 (cx, cy) 계산
2. max(bbox_width, bbox_height) × r_value → 정사각형 한 변 길이
3. 이미지 경계를 벗어나는 영역은 검은색(0)으로 패딩
4. (224, 224)로 리사이즈 (cv2.INTER_LINEAR)
```

**r_value 효과:**  
- `r=1.0` → 차량 bbox 크기와 동일한 크롭 (충돌 여부에 집중)  
- `r>1.0` → 주변 환경 포함 (타 차량 궤적 정보 추가)  
- 논문 최고 성능 결과: `r=1.0` (주변보다 충돌 자체에 집중)

### 2-3. 프레임 패딩

영상이 `clip_length(30)` 프레임보다 짧은 경우 **마지막 프레임을 반복 복사**해 길이를 맞춥니다.

```python
while len(frames) < clip_length:
    frames.append(frames[-1])   # 마지막 프레임 복제
```

### 2-4. 데이터 증강 (온라인, 학습 시에만 적용)

**모든 증강 파라미터는 한 영상의 30프레임 전체에 동일하게 적용**합니다.  
(프레임마다 다른 변환을 주면 시간적 일관성이 깨짐)

| 기법 | 적용 확률 | 범위 | 시뮬레이션 대상 |
|---|---|---|---|
| 수평 반전 (Horizontal Flip) | 50% | 좌우 반전 | 카메라 설치 방향 차이 |
| 밝기 (Brightness) | 50% (공통) | 0.6 ~ 1.4 | 지하주차장(어두움) ↔ 외부(밝음) |
| 대비 (Contrast) | 50% (공통) | 0.7 ~ 1.3 | 흐린 날(저대비) ↔ 직사광(고대비) |
| 채도 (Saturation) | 50% (공통) | 0.7 ~ 1.3 | 카메라 기종별 색상 포화도 차이 |
| 색조 (Hue) | 50% (공통) | -0.1 ~ +0.1 | 화이트밸런스·조명 색온도 차이 |

> 색상 증강 설계 이유: 실제 주차장 CCTV는 학습 데이터(RC카 촬영)와 조명 환경·날씨·카메라 색감이 다를 수 있어 이 도메인 갭을 줄이기 위함.

### 2-5. 정규화

```
[0, 255] → / 255.0 → [0.0, 1.0]
→ (pixel - mean) / std
   mean = [0.485, 0.456, 0.406]   (ImageNet RGB 평균)
   std  = [0.229, 0.224, 0.225]   (ImageNet RGB 표준편차)
```

최종 텐서 형태: `(C=3, T=30, H=224, W=224)` → DataLoader 배치 후 `(B, C, T, H, W)`

### 2-6. 장점 / 단점

| | 내용 |
|---|---|
| **장점** | 차량 bbox 기반 크롭으로 불필요한 배경 정보 제거 |
| | 색상 증강으로 실제 CCTV 환경과의 도메인 갭 완화 |
| | 온라인 증강으로 매 에포크마다 다른 변형 생성 → 사실상 데이터 증폭 효과 |
| **단점** | 단일 차량 타겟만 처리 — 다중 차량 동시 감시 불가 |
| | r_value가 고정값 — 영상마다 차량 크기가 다르면 최적값이 달라질 수 있음 |
| | 학습 시 증강만 적용, 추론 시 미적용 → 학습/추론 전처리 파이프라인 차이 존재 |

---

## 3. 학습 과정

> 파일: `model/train.py`

### 3-1. 데이터 분할

```
전체 데이터셋 → random_split(train_split_ratio=0.8)
 ├── Train: 80%  (shuffle=True)
 └── Val:   20%  (shuffle=False)
```

동일한 `HitAndRunDataset`을 공유하므로 증강은 Train/Val 구분 없이 `__getitem__`에서 발생.  
단, 추론 시에는 `model.eval()` + `torch.inference_mode()`로 Dropout이 비활성화됩니다.

### 3-2. 옵티마이저

```python
optimizer = optim.Adam(model.parameters(), lr=1e-5)
```

- Adam: 모멘텀과 적응적 학습률을 결합 — 3D CNN 같은 깊은 네트워크에 안정적
- lr=1e-5: 사전 학습 없이 처음부터 학습하므로 작은 학습률로 안정적 수렴 유도

### 3-3. 손실 함수

```python
criterion = nn.CrossEntropyLoss()
```

- 이진 분류(S/A)에 적합한 Softmax + NLL Loss 결합
- 내부적으로 Softmax를 포함하므로 모델 출력은 raw logits

### 3-4. EarlyStopping

```
patience=10: 검증 손실이 10 에포크 동안 개선 없으면 학습 중단
delta=0: 아주 미세한 개선도 인정
path: 가장 좋은 검증 손실일 때 모델 가중치 저장
```

- 과적합 방지 + 불필요한 학습 시간 절감
- 학습 종료 후 저장된 최적 가중치를 자동 로드하여 반환

### 3-5. AMP (자동 혼합 정밀도)

CUDA/ROCm 환경에서 자동 활성화됩니다.

```python
# 순전파: FP16 (빠른 Tensor Core 연산)
with torch.amp.autocast("cuda"):
    outputs = model(inputs)
    loss = criterion(outputs, labels)

# 역전파: GradScaler로 수치 안정성 확보
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

- VRAM 사용량 ~50% 감소
- Tensor Core 활용으로 연산 속도 ~2배 향상
- CPU 환경에서는 자동 비활성화

### 3-6. DataLoader 전략

```python
# CUDA: 멀티워커 병렬 전처리
num_workers = min(8, os.cpu_count())
prefetch_factor = 4       # GPU 연산 중 미리 준비할 배치 수
persistent_workers = True  # 워커 프로세스 재사용
pin_memory = True          # CPU→GPU 전송 속도 향상

# CPU/ROCm: 단일 프로세스 (멀티프로세싱 충돌 방지)
num_workers = 0
```

### 3-7. 장점 / 단점

| | 내용 |
|---|---|
| **장점** | Adam + 낮은 lr 조합으로 3D CNN 학습 안정적 수렴 |
| | EarlyStopping으로 자동 최적 모델 선택 및 과적합 방지 |
| | AMP로 메모리 효율화 — 제한된 VRAM에서 더 큰 배치 가능 |
| | DataLoader 멀티워커로 I/O 병목 제거 |
| **단점** | 학습률 스케줄러 미사용 — 후반 수렴 속도 저하 가능 |
| | 클래스 불균형(A:S 비율) 처리 없음 — 정상 클래스에 편향 가능성 |
| | 배치 정규화(BatchNorm) 미사용 — 학습 안정성 일부 희생 |
| | random_split이 영상 단위 분할만 보장 (시간적 정보 누출 없음) |

---

## 4. 추론 및 CAM 시각화

> 파일: `model/predict_cam.py`

### 4-1. 슬라이딩 윈도우 추론

긴 영상을 고정 길이(30프레임) 클립으로 잘라서 순차적으로 추론합니다.

```
전체 영상 (N 프레임)
  │
  ├── 윈도우 0:    프레임 [0 ~ 29]    → 예측 결과 (S 또는 A)
  ├── 윈도우 1:    프레임 [1 ~ 30]    → 예측 결과
  ├── 윈도우 2:    프레임 [2 ~ 31]    → 예측 결과
  │   ... (stride=1)
  └── 윈도우 N-29: 프레임 [N-30 ~ N-1] → 예측 결과
```

- `stride=1`: 모든 프레임에 대한 최대 밀도 추론 (논문 기준)
- 배치 추론(`infer_batch_size=16`)으로 GPU 활용률 최대화

### 4-2. CAM (Class Activation Map) 생성

`inception5b` 레이어에 forward hook을 걸어 피처맵을 캡처하고,  
분류 헤드(`head_conv.weight`)의 가중치와 곱해 히트맵을 생성합니다.

```
수식:
  CAM(h,w) = ReLU( Σ_c [ weight[pred_class, c] × feature_map[c, :, h, w] ] )
  cam_2d   = mean over temporal dimension T

  weight        : head_conv.weight[pred_class]   shape (1024, 1, 1, 1)
  feature_map   : inception5b 출력              shape (1024, T', H', W')
  cam           :                               shape (T', H', W')
  cam_2d        :                               shape (H', W')
```

- `ReLU`로 음수(해당 클래스에 반하는 영역) 제거
- 0~255 정규화 후 JET 색상맵 적용 → 빨강=고활성 영역
- 원본 영상과 60:40 비율로 합성 (충돌 클래스일 때만 CAM 오버레이)

### 4-3. 다중 이벤트 감지 (S→A→S 상태 머신)

```
상태:  S  S  S  A  A  A  A  S  S  S  A  A  S
           
전환 감지:     ↑ S→A        ↑ A→S      ↑ S→A
이벤트:        └── 이벤트 1 ──┘         └── 이벤트 2 ──→

반환값:
  events = [
      {'start_frame': 3,  'end_frame': 6},   # 이벤트 1
      {'start_frame': 10, 'end_frame': ...},  # 이벤트 2
  ]
```

- `start_frame`: S→A 전환이 일어난 슬라이딩 윈도우의 **시작 프레임 인덱스**
- `end_frame`: A→S 전환 직전 프레임 인덱스 (영상 끝까지 A이면 마지막 프레임)
- 영상 좌상단에 현재 클래스 상태(S=초록 / A=빨강) 실시간 표시

### 4-4. 멀티스레드 VideoWriter

추론 루프와 디스크 쓰기를 별도 스레드로 분리해 병렬 실행합니다.

```
[추론 스레드]  → frame queue (maxsize=64) → [Writer 스레드] → mp4 파일
```

- 추론이 쓰기보다 빠를 때 큐가 버퍼 역할
- 예외 발생 시 큐에 `None` (종료 시그널)을 넣어 스레드 안전하게 종료

### 4-5. 장점 / 단점

| | 내용 |
|---|---|
| **장점** | 임의 길이 영상 처리 가능 (슬라이딩 윈도우) |
| | CAM으로 충돌 지점 시각화 — 오탐 검토 및 모델 신뢰도 제공 |
| | 다중 이벤트 자동 감지 및 시작·종료 프레임 반환 |
| | 멀티스레드 I/O로 처리 속도 향상 |
| **단점** | stride=1이면 N-29개 윈도우 추론 → 긴 영상에서 느림 |
| | 예측 노이즈(한 윈도우만 A/S 반전)가 이벤트를 잘못 분리할 수 있음 |
| | CAM의 공간 해상도는 inception5b 출력 크기에 제한됨 |
| | 단일 타겟 차량만 처리 (타겟 ID 하나 지정 필요) |

---

## 5. 성능 평가

> 파일: `model/evaluate.py`

### 5-1. GT(정답) 레이블 판정

파일명 규칙으로 정답을 자동 결정합니다.

```
파일명 형식: {name}_{XA or XS}_{...}.mp4
              ↑ 두 번째 토큰의 두 번째 문자

예: video_1A_front.mp4  →  label=1 (Accident)
    video_2S_side.mp4   →  label=0 (Safe)
```

### 5-2. Any-Window-1 전략 (High-Recall 판정)

```python
for batch in windows:
    outputs = model(clips)
    if (outputs.argmax(dim=1) == 1).any():
        predicted_label = 1   # 한 번이라도 A 예측 → Accident
        break                 # Early exit
```

- **한 윈도우라도 A를 예측하면 영상 전체를 Accident로 판정**
- 물피도주 탐지에서 False Negative(미탐)를 최소화하는 보수적 전략
- 오탐(False Positive)이 늘어나는 단점이 있지만, 탐지 목적 상 미탐이 더 큰 손실

### 5-3. 장점 / 단점

| | 내용 |
|---|---|
| **장점** | High-Recall 전략으로 충돌을 놓칠 가능성 최소화 |
| | Early exit으로 Accident 영상의 평가 속도 향상 |
| | 오답 영상을 파일명으로 추적하는 오답 노트 기능 |
| **단점** | 배경 잡음에도 민감 — 오탐률(False Positive Rate) 상승 가능 |
| | 다수결(Majority Vote) 대신 단일 예측으로 판정 — 더 공격적 |
| | 파일명 규칙에 의존 → 이름 형식이 다른 데이터에 범용성 없음 |

---

## 6. 하이퍼파라미터 통합표

> 파일: `model/config.py`

| 범주 | 파라미터 | 값 | 설명 |
|---|---|---|---|
| **모델** | `MODEL_NUM_CLASSES` | 2 | S(정상) / A(충돌) |
| | `CLIP_LENGTH` | 30 | 입력 클립 프레임 수 |
| | `RESIZE` | (224, 224) | 프레임 해상도 (W, H) |
| | `R_VALUE` | 1.0 | 차량 bbox 크롭 배율 |
| **학습** | `TRAIN_BATCH_SIZE` | 15 | 학습 배치 크기 |
| | `TRAIN_NUM_EPOCHS` | 100 | 최대 학습 에포크 수 |
| | `TRAIN_LEARNING_RATE` | 0.00001 | Adam 학습률 |
| | `TRAIN_SPLIT_RATIO` | 0.8 | 학습/검증 분할 비율 |
| | `TRAIN_EARLY_STOPPING_PATIENCE` | 10 | 조기 종료 대기 에포크 수 |
| | `USE_AMP` | True | 자동 혼합 정밀도 (CUDA only) |
| | `USE_CHANNELS_LAST` | True | Channels Last 포맷 (NVIDIA CUDA only) |
| **추론** | `PREDICT_INFER_BATCH_SIZE` | 16 | 추론 배치 크기 |
| | `PREDICT_WINDOW_STRIDE` | 1 | 슬라이딩 윈도우 이동 간격 |
| **평가** | `EVAL_NUM_SAMPLES` | 10 | 랜덤 평가 영상 수 (None=전체) |
| | `EVAL_INFER_BATCH_SIZE` | 16 | 평가 추론 배치 크기 |
| | `EVAL_WINDOW_STRIDE` | 1 | 슬라이딩 윈도우 이동 간격 |
| **디바이스** | `DEVICE_TYPE` | "cuda" | "cuda" 또는 "cpu" |

---

## 7. 학습 안정화 — 추가/변경/보류 기법

> **배경:** 학습 시 **손실값이 진동하고 수렴이 안 되는** 문제가 있었다(감지가 쉬운 영상에서도 충돌 미감지). 원인 진단 후, **모델 아키텍처·손실 함수·기본 학습률을 바꾸지 않는 "안전한" 안정화 기법**만 선별 적용했다. 즉 *모델 성능을 떨어뜨리지 않거나 영향이 미미한* 변경만 반영했다.

### 7-1. 검증(val) 세트 증강 제거 — **[변경/버그수정]**

- **문제:** `HitAndRunDataset.__getitem__`이 train/val 구분 없이 **항상** `_apply_augmentation`(랜덤 hflip·brightness/contrast/saturation/hue jitter)을 적용했다. `train.py`가 `random_split(동일 데이터셋)`으로 나눴기 때문에 **검증 세트도 매 epoch 무작위로 바뀌었다.**
  - → **val loss가 epoch마다 진동** (같은 영상인데 입력이 매번 달라짐)
  - → EarlyStopping·best-model 선택이 **노이즈 기반**이라, 실제로 좋지 않은 가중치가 저장될 수 있었음
- **변경:**
  - `dataset.py`: `HitAndRunDataset(augment=True)` 플래그 추가. `__getitem__`은 `self.augment`일 때만 증강.
  - `train.py`: 학습용(`augment=True`)·검증용(`augment=False`) 데이터셋을 **각각 생성**해, **동일 인덱스(seed=42 고정)** 로 `Subset` 분할. → val은 결정적, train만 증강.
- **효과:** val loss 안정화, best-model 선택 신뢰도 향상. (※ 본 문서 *2-4*에 "학습 시에만 적용"이라 명시돼 있었으나 실제 코드가 val에도 적용하던 불일치를 바로잡은 것.)
- **성능 영향:** 없음(오히려 정상화). 모델 입력 분포·구조 불변.

### 7-2. Gradient Clipping — **[추가]**

- **위치:** `train.py` 학습 루프, `backward()` 후 `optimizer.step()` 전.
  ```python
  torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)  # grad_clip_norm = 1.0
  ```
  - AMP 경로에서는 정확한 클리핑을 위해 `scaler.unscale_(optimizer)`로 스케일을 먼저 해제한 뒤 클리핑 → `scaler.step()`.
- **목적:** 그래디언트 노름에 상한을 둬 **폭주(exploding gradient)·진동을 억제**.
- **성능 영향:** 미미. 정상적인 업데이트엔 영향이 거의 없고, **비정상적으로 큰 그래디언트만** 잘라낸다. 모델 구조·손실 불변. (필요 시 `grad_clip_norm` 값 1.0→0.5로 더 강하게 조정 가능)

### 7-3. ReduceLROnPlateau LR 스케줄러 — **[추가]**

- **위치:** `train.py`.
  ```python
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
  ...
  scheduler.step(avg_val_loss)   # 매 epoch
  ```
  - epoch 로그에 현재 LR도 함께 출력.
- **목적:** **val loss가 정체(patience=3 epoch)** 되면 LR을 절반(factor=0.5)으로 낮춰, 최솟값 근처에서 더 미세하게 수렴하도록 보조.
- **성능 영향:** 없음~긍정적. **기본 학습률은 그대로** 두고, 정체 시에만 LR을 낮추므로 성능을 떨어뜨리는 경로가 없다.

### 7-4. 적용하지 않은(보류) 기법 — **[보류]**

다음은 **모델 성능/결과를 바꿀 수 있어** 의도적으로 적용하지 않았다(필요 시 별도 실험):

| 기법 | 보류 이유 |
|---|---|
| 기본 학습률 변경 (예: 1e-5 → 1e-4) | 학습 동역학을 직접 바꿈 → 결과 달라짐(튜닝 영역) |
| BatchNorm → GroupNorm | 아키텍처 변경 → 기존 가중치와 불일치, 결과 달라짐 |
| 배치 크기 변경 | BN 통계·일반화에 영향 → 결과 달라짐 |
| Reduction 인셉션(다운샘플 인셉션) 도입 | 아키텍처 변경 + 재학습 필요, 본 문제(수렴 불안정)의 직접 원인 아님 |

> **요약:** 7-1(val 증강 제거)로 진동의 직접 원인을 제거하고, 7-2·7-3으로 최적화를 안정화했다. 세 가지 모두 **모델 구조·손실·기본 LR 불변** 으로, 성능 저하 위험 없이 학습 안정성만 높이는 변경이다.

---

## 8. S3D 전이학습 전환 & 데이터 증강 보강

> **배경:** "성능 저하 없이 경량화하고 오히려 성능을 높일 방법"을 검토한 결과, 커스텀 I3D(12.3M)를 **torchvision S3D + Kinetics-400 사전학습**으로 교체했다. 동시에 데이터 증강을 "다양한 주차장 환경 일반화" 관점에서 재검토·보강했다. (아키텍처 변경이므로 **재학습 필수** — 기존 `hitandrun_model_best.pth`는 새 구조와 호환되지 않는다.)

### 8-1. 백본 교체: 커스텀 I3D → S3D (Kinetics-400 사전학습) — **[변경]**

- **S3D란:** I3D의 3D conv(k×k×k)를 **공간(1×k×k) + 시간(k×1×1) 분리 컨볼루션**으로 인수분해한 구조 (Xie et al., *Rethinking Spatiotemporal Feature Learning*, ECCV 2018). 논문에서 I3D 대비 **가볍고 빠르면서 정확도는 동등 이상**임이 입증됐다.
- **파라미터:** 12,289,314 → **7,912,098 (약 36% 경량)**. CPU 추론 속도는 실측상 기존과 비슷한 수준(레이어 수가 늘어 FLOPs 감소가 CPU에선 상쇄됨) — 경량화 이득은 주로 메모리/저장 크기.
- **사전학습에 대한 판단 재검토:** "사전학습은 모델을 무겁게 한다"는 이전 판단은 오해 — 사전학습은 파라미터 **초기값**만 바꾸며 크기·속도는 동일하다. 미세한 차량 떨림 감지에서도 **하위 레이어의 모션 프리미티브(시간 그래디언트·방향성 모션 필터)는 그대로 전이**되고, 상위 레이어는 미세조정으로 과제에 특화된다. 소규모 데이터(현 상황)에서는 사전학습 이득이 가장 크다.
- **서비스 인터페이스 보존:** 클래스명 `HitAndRun3DCNN`, 분류 헤드 `head_conv`(1×1×1 Conv3d, CAM 가중치), CAM hook 대상 `inception5b`(→ S3D 마지막 인셉션 블록을 가리키는 **property 별칭**)를 유지해 `predict_cam.py`·백엔드 워커가 **코드 수정 없이** 동작한다.
- **생성 인자:** `HitAndRun3DCNN(num_classes, pretrained=False)` — 기본 False(서비스 워커는 학습된 state_dict를 로드하므로 다운로드 불필요), 학습 시에만 `config.PRETRAINED=True`로 초기화.

### 8-2. 입력 정규화 통계 변경 — **[변경]**

- ImageNet 통계(0.485/0.456/0.406 …) → **Kinetics-400 통계(0.43216/0.394666/0.37645, std 0.22803/0.22145/0.216989)**. 사전학습과 동일한 정규화를 써야 전이 효율이 최대화된다.
- 3곳에 중복 하드코딩돼 있던 것을 **`config.NORM_MEAN / NORM_STD`로 일원화** (`dataset.py`, `predict_cam.py`, `evaluate.py`가 공유).

### 8-3. 미세조정 학습률 전략 — **[변경]**

- `TRAIN_LEARNING_RATE = 1e-4` (헤드 기준). **백본(features)은 ×0.1(1e-5), 새로 초기화된 헤드(head_conv)는 1e-4** 의 파라미터 그룹으로 분리 — 사전학습 특징은 보수적으로 보존하고 헤드는 빠르게 적응시키는 표준 미세조정 관행.
- ReduceLROnPlateau(7-3)는 두 그룹 모두 비례 감소.

### 8-4. 데이터 증강 검토 결과 및 보강 — **[검토+추가]**

**기존(유지):** hflip(p=0.5), 색상 지터(밝기 0.6–1.4 / 대비 0.7–1.3 / 채도 0.7–1.3 / 색조 ±0.1) — 조명·날씨·카메라 색감 대응으로 적절. 클립 내 모든 프레임 동일 변환(시간적 일관성) 원칙도 올바름.

**부족했던 부분 → 추가 (train 전용):**

| 추가 증강 | 시뮬레이션 대상 | 구현 |
|---|---|---|
| **시간 오프셋 지터** | 학습 클립이 항상 "충돌 시작=윈도 첫 프레임"인 **위치 편향** — 추론(슬라이딩 윈도)에선 충돌이 윈도 어디에나 걸림 | `start_f - randint(0,10)` (0~10프레임 앞당김, 충돌은 윈도 내 유지) |
| **bbox 지터** | 서비스에서 사용자가 **마우스로 대충 그린 박스** vs 학습의 정밀 GT 박스 분포 불일치 | 중심 ±5% 이동 + 크기 0.9~1.15배 |
| **그레이스케일 (p=0.15)** | **야간 적외선(IR) CCTV** — 사실상 무채색 영상 | `rgb_to_grayscale(3ch)` |
| **가우시안 블러 (p=0.2)** | 저화질 CCTV 초점 흐림·압축 열화 | `gaussian_blur(k=3)` |
| **센서 노이즈 (p=0.2)** | 야간 게인 상승 노이즈 (σ 2~8, **프레임별 재샘플링** — 실제 노이즈 특성) | 가우시안 노이즈 가산 |

**검토했지만 보류:**

| 보류 증강 | 사유 |
|---|---|
| 카메라 흔들림 합성(정상 클립에 전역 모션 추가) | "전역 흔들림 ≠ 사고" 학습에 유용하나 레이블 의미에 개입 → 별도 실험으로 |
| 시간 리샘플링(fps 변화 시뮬레이션) | 충돌 동역학의 속도 자체가 판별 신호일 수 있어 신중해야 함 |
| 시간 역재생 | 충돌은 방향성 있는 사건 — 역재생은 비현실적 분포 |
| Cutout/부분 가림 | 효과 불확실, 우선순위 낮음 |

> **요약:** 사전학습 S3D 전이로 "경량화(−36%) + 소규모 데이터 일반화"를 동시에 달성하고, 증강은 **추론 환경과의 분포 불일치**(윈도 오프셋, 손그림 bbox)와 **CCTV 도메인 특성**(야간 IR, 저화질)을 메우는 방향으로 보강했다. Colab 노트북(`hitandrun_colab.ipynb`)도 동일 코드로 재생성됨.
