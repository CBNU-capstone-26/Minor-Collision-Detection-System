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
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r Pre-Processing/requirements.txt
```

검수용 H.264 MP4를 만들려면 `ffmpeg`도 필요하다.

```bash
brew install ffmpeg
```

JSON 포맷터는 `Pre-Processing/.prettierrc.json`에 정의되어 있으며, 들여쓰기 4칸·공백 사용·줄바꿈 LF를 사용한다. 프로그램의 `save_annotations()`도 같은 4칸 들여쓰기 규칙으로 저장한다. Prettier CLI를 설치한 환경에서는 다음 명령으로 기존 JSON을 수동 정리할 수 있다.

```bash
npx prettier --config .prettierrc.json --write work/annotations.json
```

## 실행

`Pre-Processing` 폴더에서 실행한다.

```bash
cd "/Users/manuelpark/Documents/대학교 프로그래밍/캡스톤디자인/Minor-Collision-Detection-System/Pre-Processing"
source ../.venv/bin/activate
python -m src.app \
  --source-root ../Accident \
  --annotations work/annotations.json \
  --output-root output
```

`src`를 모듈로 실행해야 상대 import가 정상적으로 동작한다. GUI가 열리면 `학습용`과 `테스트용`의 MP4가 왼쪽 목록에 표시된다. `폴더 열기`로 다른 폴더를 선택할 수도 있다.

영상 목록 위의 정렬 메뉴에서 `번호순`, `파일명순`, `검수 상태순`을 선택할 수 있다. 번호순은 파일명의 숫자를 자연스러운 숫자 순서로 정렬하고, 검수 상태순은 `미작성 → 작업 중 → 완료 → 제외` 순서로 배치한다.

## 작업 순서

1. 왼쪽 영상 목록에서 영상을 선택한다.
2. 영상 아래 재생 바를 드래그하거나 좌우 버튼·방향키로 차량이 가장 잘 보이는 프레임으로 이동한다.
3. 영상 위에서 마우스로 차량을 드래그해 박스를 만든다.
4. 박스를 클릭하면 선택된다. 박스 안을 드래그하면 이동한다.
5. 선택된 박스에는 8개의 조절점이 표시된다. 네 변 중앙 점은 한 방향만 조절하고, 네 모서리 점은 원래 가로·세로 비율을 유지하며 크기를 조절한다.
6. 오른쪽 `기준 차량 박스` 목록에서 입력된 ID와 좌표를 확인한다. `선택 박스 삭제` 버튼 또는 목록 우클릭 메뉴로 박스를 삭제할 수 있다.
   박스를 새로 만들거나 이동·크기 조절할 때 좌표는 항상 영상 경계 안으로 제한된다.
7. `영상 현황`에서 `작업 중`, `완료`, `제외` 중 하나를 선택한다.
8. `사고 차량 ID`를 선택한다.
9. 시작·종료 프레임을 입력한다.
10. `사고 이벤트 추가`를 누른 뒤 `검수 완료`를 누른다.
11. `저장`을 눌러 작업과 영상별 현황을 JSON에 저장한다.
12. `TXT + 검수 MP4 생성`을 눌러 결과를 만든다.

## 이벤트 관리

오른쪽 `이벤트 목록`에서 현재 영상의 이벤트를 선택할 수 있다. 이벤트를 선택하면 사고 차량 ID, 시작·종료 프레임, 이벤트 상태와 메모를 수정할 수 있다.

- `사고 이벤트 추가`: 현재 프레임과 선택된 차량으로 새 이벤트를 만든다.
- `선택 이벤트 저장`: 선택한 이벤트의 수정 내용을 JSON에 저장한다.
- `검수 완료`: 선택한 이벤트를 `confirmed`로 저장한다. 여러 이벤트가 있으면 모든 이벤트가 완료된 경우에만 영상 현황도 `완료`가 된다.
- `선택 이벤트 삭제`: 확인 창에서 승인한 뒤 이벤트를 삭제하고 JSON에 저장한다.

이벤트가 여러 개이면 `영상명-a1`, `영상명-a2`처럼 목록에 유지되며, 내보내기 결과도 이벤트별로 생성된다.

단축키:

- `←` / `→`: 프레임 이동
- `←` / `→` 길게 누르기: 키 반복 입력에 따라 프레임 연속 이동
- `Space`: 재생·정지
- `Delete`: 선택한 박스 삭제
- `ESC`: 박스 선택 해제
- `Cmd/Ctrl+Z`: 마지막 박스 작업 실행 취소
- `Cmd/Ctrl+Shift+Z` 또는 `Ctrl+Y`: 실행 취소한 박스 작업 다시 실행
- `S`: 저장

영상 아래의 `이전`·`다음` 버튼도 길게 누르면 약간의 지연 후 자동 반복되어 프레임이 연속으로 이동한다.

프로그램을 닫았다가 다시 실행해도 `Pre-Processing/work/annotations.json`에서 작업을 복구한다.

각 영상의 현황은 JSON의 `status` 필드에 저장된다. 상태를 바꾼 뒤에는 반드시 `저장`을 눌러야 다음 실행 때도 유지된다. 박스를 만들거나 이벤트를 추가하면 자동으로 `작업 중`이 되고, `검수 완료`를 누르면 `완료`가 된다.

오른쪽 메뉴는 별도 스크롤 영역이다. 창 높이가 부족하면 오른쪽 패널 위에서 마우스 휠을 사용해 이벤트 관리·저장·내보내기 버튼까지 내려갈 수 있다.

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
