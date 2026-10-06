# 프로젝트 워크플로우 자료

확인 기준: 2026-09-21. 이 자료는 저장소의 구현과 현재 세션에서 적용한 모델을 설명한다. 학습 실행 이력이나 모델 성능을 새로 측정한 결과는 아니다.

## 파일

- `project_workflow.png`: 흰색 배경의 2400×1900 발표용 이미지
- `create_workflow.py`: 문구와 배치를 수정해 이미지를 다시 생성하는 원본

## 핵심 구분

| 범위 | 코드·설정에서 확인된 내용 |
|---|---|
| 저장소 학습 모델 | `model/train.py`가 `HitAndRun3DCNN` 생성. `model/hitandrun_model.py`의 커스텀 3D Inception이며 사전학습을 사용하지 않음 |
| 웹에 적용한 모델 | `model/slowfast_service.py`의 PyTorchVideo SlowFast R50. `backend/app/prediction_job.py`에서 체크포인트 키에 따라 모델 선택 |
| 적용 가중치 | `weights_2026-09-11/slowfast_r50_2026-09-11_best.pth`. 이번 세션 워커 실행 환경변수로 지정. 설정 파일의 기본 가중치 경로를 바꾼 것은 아님 |
| SlowFast 학습 조건 | 이 체크포인트에는 state_dict가 있으며 학습 로그·하이퍼파라미터·성능 기록은 확인되지 않음. 아래 3D Inception 학습 설정을 SlowFast의 학습 이력으로 해석하면 안 됨 |

## 학습 흐름의 근거

- `model/dataset.py`: MP4와 같은 이름의 TXT에서 BBox, A/S 클래스, 시작 프레임을 읽음. ROI crop/pad, 224×224 리사이즈, 30프레임, RGB 정규화.
- 학습 증강: 시간·BBox 지터, 좌우 반전, 색상 변화, 회색조, 블러, 노이즈. 검증 데이터에는 증강하지 않음.
- `model/train.py`: 도메인/방향/시나리오별 층화 분할, 분할 seed 42, 그룹별 학습 비율 0.8. 그룹별 정수 반올림 때문에 실제 전체 비율은 정확히 80:20이 아닐 수 있음.
- `model/config.py`: batch 8, 최대 100 epoch, 학습률 0.00003, early stopping patience 15.
- `model/train.py`: CrossEntropyLoss, Adam, ReduceLROnPlateau(factor=0.5, patience=3), gradient clipping 1.0. CUDA 사용 시 AMP.
- 검증은 RC/실제 영상을 나누어 loss와 accuracy 산출. 실제 영상 검증 손실이 있으면 이를 기준으로 best model을 저장하고, 없으면 RC 검증 손실 사용.
- 기본 `data/train/`에서 MP4 0개, `label/`에서 TXT 935개 확인. 935는 파일 수로 실제 학습 표본 수가 아님. 외부 학습 서버나 다른 환경변수 경로의 데이터 수는 조사하지 않음.

## 성능 표시의 근거와 한계

- `model/evaluate.py`는 영상 단위 TP, TN, FP, FN, Accuracy, Recall, Precision, F1, False alarm을 계산하는 코드임.
- `model/main.py`의 기존 eval 진입점은 3D Inception을 생성함. 현재 SlowFast 웹 후처리 파이프라인의 성능을 검증한 결과가 아님.
- 기본 `data/eval/`가 없고, 저장소에서 SlowFast 정량 평가 보고서나 epoch별 학습 로그를 확인하지 못했으므로 수치는 **미확인**으로 표시.
- `outputs/*summary.json`은 다른 `TemporalCollisionCNN`의 단일 영상 예측 요약이며, 정답 주석으로 heatmap 표시가 제한된 결과. 그 확률을 현재 SlowFast의 정확도로 사용할 수 없음.
- 가중치 파일명의 loss는 정확도, 재현율, F1을 대신하지 않음. 실제 성능 비교에는 동일 평가셋, 체크포인트, 전처리와 이벤트 후처리 설정의 기록이 필요함.

## 웹 분석 흐름의 근거

- `backend/app/routers/videos.py`: YOLO11x 차량 검출. 사용자 BBox를 분석에 활용.
- `model/predict_cam.py`: ROI 입력의 슬라이딩 윈도 분석, Softmax 충돌 확률, 움직임·확률·지속시간 기반 구간 필터, CAM 클립 생성.
- `model/config.py`: 서비스 입력 30프레임, 224×224, 추론 batch 2, 윈도 stride 5, CPU 추론.
- `model/slowfast_service.py`: 입력을 Fast 32프레임으로 재샘플링하고 Slow 8프레임을 추출. 두 경로에서 특징 융합 후 2304차원 특징을 2개 클래스에 투영. CAM은 Res5 특징의 시간 평균과 분류 가중치를 사용.
- `backend/app/prediction_job.py`: Celery 작업, 분석 상태와 이벤트 DB 저장, 생성된 클립 경로 반환.

## 재생성

Pillow와 macOS AppleSDGothicNeo 폰트가 있는 Python에서 `create_workflow.py`를 실행하면 같은 폴더에 PNG를 생성한다.

## 사용자 제공 성능 반영본

- 최신 간소화 자료: `slowfast_visual_summary.png` (2400×1640).
- 편집·재생성 원본: `create_visual_summary.py`.
- 입력 수치: `user_provided_metrics.csv`. 사용자가 대화에서 제공한 검증 정확도와 사고 F1을 그대로 시각화했으며, 독립적으로 재평가한 수치는 아님.
- S3D: 정확도 90.9%, F1 0.75. X3D-S: 정확도 81.8%, F1 0.50. SlowFast-R50: 정확도 100%, F1 1.00.
- 검증 표본 수와 평가 조건은 제공되지 않음. 개별 예측값 없이 산점도나 학습 곡선을 추정하지 않고, 제공된 집계 지표의 막대그래프를 사용함.
- 학습 과정은 SlowFast의 두 경로를 설명하는 개념도이며, 기존 3D Inception 학습 코드나 설정이 SlowFast 학습에 사용되었다는 의미가 아님.
- 초기 `project_workflow.png`는 수치 제공 전 저장소 조사 기록으로 보존.
