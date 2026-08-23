# 수동 바운딩 박스 GUI

이 도구는 논문 데이터셋의 메타데이터 형식에 맞춰 차량의 고정 바운딩 박스와 사고 프레임 범위를 직접 입력하는 macOS용 GUI다.

원본 `Accident/학습용`, `Accident/테스트용` 영상은 읽기만 하며 수정하지 않는다.

## 논문 형식

차량 박스는 프레임 번호가 없는 고정 박스다. 사고 차량은 `car 0` 또는 `car 1`로 고정되지 않고 `A` 레코드의 차량 ID로 지정한다.

```text
car,0,x1,y1,x2,y2
car,1,x1,y1,x2,y2
A,0,start_frame,end_frame
```

GUI에서는 차량이 가장 명확하게 보이는 프레임을 선택해 박스를 입력한다. 차량별 기준 프레임은 작업 JSON에만 저장되고, 논문 호환 TXT에는 좌표만 저장된다.

## 설치

터미널에서 저장소 루트로 이동한다.

```bash
cd "/Users/manuelpark/Documents/대학교 프로그래밍/캡스톤디자인/Minor-Collision-Detection-System"
python3 -m venv .venv
source ../.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r Pre-Processing/requirements.txt
```

검수용 H.264 MP4를 만들려면 `ffmpeg`도 필요하다.

```bash
brew install ffmpeg
```

## 실행

`Pre-Processing` 폴더에서 실행한다.

```bash
cd "/Users/manuelpark/Documents/대학교 프로그래밍/캡스톤디자인/Minor-Collision-Detection-System/Pre-Processing"
source .venv/bin/activate
python -m src.app \
  --source-root ../Accident \
  --annotations work/annotations.json \
  --output-root output
```

`src`를 모듈로 실행해야 상대 import가 정상적으로 동작한다. GUI가 열리면 `학습용`과 `테스트용`의 MP4가 왼쪽 목록에 표시된다. `폴더 열기`로 다른 폴더를 선택할 수도 있다.

## 작업 순서

1. 왼쪽 영상 목록에서 영상을 선택한다.
2. 차량이 가장 잘 보이는 프레임으로 이동한다.
3. 영상 위에서 마우스로 차량을 드래그해 박스를 만든다.
4. 박스를 클릭하면 선택되고, 드래그하면 이동한다. 오른쪽 아래 모서리를 드래그하면 크기가 바뀐다.
5. 오른쪽 `기준 차량 박스` 목록에서 입력된 ID와 좌표를 확인한다.
6. `사고 차량 ID`를 선택한다.
7. 시작·종료 프레임을 입력한다.
8. `사고 이벤트 추가`를 누른 뒤 `검수 완료`를 누른다.
9. `저장`을 눌러 작업을 JSON에 저장한다.
10. `TXT + 검수 MP4 생성`을 눌러 결과를 만든다.

단축키:

- `←` / `→`: 프레임 이동
- `Space`: 재생·정지
- `Delete`: 선택한 박스 삭제
- `S`: 저장

프로그램을 닫았다가 다시 실행해도 `Pre-Processing/work/annotations.json`에서 작업을 복구한다.

## 출력

```text
Pre-Processing/
├── work/annotations.json
└── output/
    ├── learning/txt/
    ├── learning/visualized/
    ├── testing/txt/
    ├── testing/visualized/
    └── normal/txt/
```

영상 하나의 첫 사고 이벤트는 원본 영상명과 같은 이름으로 저장된다. 같은 영상의 두 번째 이벤트부터 `__a2`, `__a3`가 붙는다.

검수용 MP4는 원본 영상 전체 길이·해상도·FPS·프레임 수를 유지한다. 모든 기준 차량 박스를 표시하고, 사고 차량은 주황색으로 강조하며 현재 프레임 번호와 사고 구간을 표시한다.

## 테스트

저장소 루트에서 실행한다.

```bash
python3 -m unittest discover -s Pre-Processing/tests -v
python3 -m py_compile Pre-Processing/src/*.py
```

GUI를 실행하지 않고도 TXT 형식, 좌표·프레임 검증, JSON 저장·복구, 다중 이벤트 파일명 규칙을 테스트할 수 있다.
