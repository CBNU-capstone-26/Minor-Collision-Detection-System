# 비사고 영상 자동 어노테이션

논문의 비사고 클래스 `S`처럼, 주행 차량 주변의 인접 차량 2대를 `car 0`, `car 1`로 기록하는 파이프라인이다. 자동 탐지 결과는 초안이며, 기존 수동 GUI에서 검수한 뒤 최종 파일을 만든다.

## 설치

저장소 루트에서 실행한다.

```bash
cd "/Users/manuelpark/Documents/대학교 프로그래밍/캡스톤디자인/Minor-Collision-Detection-System"
source .venv/bin/activate
python -m pip install -r "Non-Accident Pre-Processing/requirements.txt"
brew install ffmpeg
```

## 자동 초안 생성

```bash
cd "Non-Accident Pre-Processing"
source ../.venv/bin/activate

python -m src.nonaccident_pipeline inventory \
  --source-root ../Non-Accident \
  --output work/inventory.json

python -m src.nonaccident_pipeline detect \
  --inventory work/inventory.json \
  --output work/drafts.json \
  --model yolo11n.pt

python -m src.nonaccident_pipeline prepare-review \
  --drafts work/drafts.json \
  --output work/nonaccident_annotations.json
```

`detect`는 Ultralytics YOLO와 ByteTrack으로 여러 프레임의 승용차를 추적한다. 기본 탐지 클래스는 COCO의 `car`(`class 2`)만 사용하며, 신뢰도 `0.5` 미만의 탐지는 제외한다. 이동량이 큰 주행 차량은 제외하고, 그 주변에 있으면서 정지 상태인 차량 2대를 기준 차량으로 제안한다.

가림 대응도 자동으로 적용된다. 기준 박스는 두 차량이 동시에 검출된 프레임 중에서 선택하며, 이동 차량의 박스와 크게 겹치는 프레임은 대표 프레임 후보에서 감점한다. 짧게만 나타난 추적 ID도 기준 차량 후보에서 제외한다. 따라서 이동 차량이 앞을 지나가는 순간의 헤드라이트·차체 일부 박스가 기준 박스로 선택될 가능성을 줄인다.

코드나 탐지 조건을 변경한 뒤에는 기존 `drafts.json`을 재사용하지 말고 `detect`부터 다시 실행한다.

```bash
python -m src.nonaccident_pipeline detect \
  --inventory work/inventory.json \
  --output work/drafts.json \
  --model yolo11n.pt
```

## GUI 검수

```bash
python -m src.app \
  --mode non-accident \
  --source-root ../Non-Accident \
  --annotations work/nonaccident_annotations.json \
  --output-root output
```

비사고 모드에서는 사고 이벤트와 시작·종료 프레임을 입력하지 않는다. 자동 박스를 수정하고, 기준 차량이 정확히 두 대인지 확인한 뒤 `기준 차량 2대 확정`과 `저장`을 누른다. 이동 차량은 최종 TXT에 포함하지 않는다.

## 최종 내보내기

```bash
python -m src.nonaccident_pipeline export \
  --annotations work/nonaccident_annotations.json \
  --output-root output
```

`confirmed` 상태인 영상만 처리한다. `needs_review`와 `excluded`는 건너뛴다.

TXT는 논문 형식에 맞춰 `A` 레코드 없이 두 줄만 생성한다.

```text
car,0,x1,y1,x2,y2
car,1,x1,y1,x2,y2
```

결과는 `output/normal/txt/`와 `output/normal/visualized/`에 저장된다. 검수 MP4는 원본의 전체 길이·해상도·FPS·프레임 수를 유지한다.

## 주의

- 원본 `Non-Accident/*.mp4`는 수정하지 않는다.
- `yolo11n.pt`는 첫 탐지 실행 때 자동으로 다운로드될 수 있다.
- 자동 결과는 최종 확정본이 아니므로 GUI 검수를 거쳐야 한다.
