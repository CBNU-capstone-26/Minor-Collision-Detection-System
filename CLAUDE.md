# CLAUDE.md

이 파일은 현재 위치의 repository 안에서 동작하는 코드들의 Claude Code (claude.ai/code)의 guidance를 제공합니다.

> 상세 구축 계획·의사결정은 루트 `IMPLEMENTATION_PLAN.md` 참고.

---

## 프로젝트 컨텍스트

주차장 내 물피도주(차량 접촉/스크래치 후 도주) 사고를 CCTV 영상으로 인식하는 웹 서비스.
사용자가 **영상을 업로드 → 재생 화면에서 사고 차량을 bbox로 직접 지정 → AI가 사고 의심 구간을 탐지 → 해당 구간만 CAM 오버레이 클립으로 생성**하여 웹에서 리뷰한다.

**핵심 데이터 흐름 (E2E):**
```
[UI] 업로드(+녹화일자) ─▶ [FastAPI] storage/uploads 저장 + 메타추출 ─▶ MySQL videos
[UI] 사고차량 드래그(bbox) + '사고감지 실행'
        └─bbox를 원본 해상도로 환산─▶ [FastAPI] AnalysisTask 생성 ─▶ Celery enqueue(Redis)
[Celery 워커] 모델 로드(1회) ─▶ 슬라이딩윈도우 추론 ─▶ 사고구간만 CAM 클립 생성(storage/clips)
        └─▶ MySQL crash_events 저장(시작/종료 프레임·초, 확률, 클립경로)
[UI] 원본 영상 재생 + 타임라인 마커 + 이벤트 ▶버튼 ─▶ 팝업에서 CAM 클립 재생
```

---

## 기술 스택

- **Frontend**: React 19 + Vite, react-router-dom (SPA, URL 라우팅)
- **Backend**: FastAPI, SQLAlchemy 2.x
- **DB**: MySQL 8 (docker)
- **비동기 작업**: Celery + Redis (broker/result backend)
- **ML/DL**: PyTorch, 3D-CNN(I3D 기반 GoogLeNet), OpenCV, 3D-CAM Overlay
- **저장소**: 로컬 파일시스템 `storage/` (GPU/스토리지 서버 대체)

---

## 폴더 구조

```
capstone-26/
├── model/                      ← ML 패키지 (학습·추론·평가)
│   ├── __init__.py            ← sys.path 설정 (외부 임포트 지원)
│   ├── config.py              ← 모든 파라미터 (여기만 수정)
│   ├── dataset.py             ← HitAndRunDataset
│   ├── hitandrun_model.py     ← HitAndRun3DCNN (torchvision S3D 백본 + head_conv)
│   ├── train.py               ← EarlyStopping, train_model
│   ├── predict_cam.py         ← predict_hit_and_run_final(CLI용 전체영상) /
│   │                            predict_events_and_clips(서비스용 구간 클립)
│   ├── evaluate.py            ← evaluate_folder_accuracy
│   ├── device_utils.py        ← get_device(device_type), is_cuda_like
│   └── main.py                ← CLI 진입점
├── backend/
│   ├── app/
│   │   ├── main.py            ← FastAPI 앱, CORS, 라우터 등록, 시작 시 테이블 생성
│   │   ├── settings.py        ← DB/Redis URL, storage 경로 (BASE_DIR 기준)
│   │   ├── db_connection.py   ← engine, SessionLocal, Base, get_db
│   │   ├── db_models.py       ← User/Video/AnalysisTask/CrashEvent (SQLAlchemy)
│   │   ├── api_schemas.py     ← Pydantic 요청/응답
│   │   ├── auth_guard.py      ← 간소 토큰 인증(hash_password, get_current_user)
│   │   ├── worker.py          ← Celery 인스턴스 (celery -A app.worker)
│   │   ├── prediction_job.py  ← run_prediction_task (모델 지연로딩 + 클립생성)
│   │   └── routers/
│   │       ├── auth.py        ← /api/auth/signup, /login
│   │       ├── videos.py      ← /api/videos (업로드/목록/상세/stream)
│   │       └── analysis.py    ← /api/videos/{id}/analyze, /tasks/{id}, /events/{id}/clip
│   └── requirements.txt
├── ui/                         ← React + Vite (src/App.jsx, src/api.js)
├── storage/                    ← 로컬 저장소 (gitignore 권장)
│   ├── uploads/               ← 업로드 원본 mp4  ({uuid}{ext})
│   └── clips/                 ← 사고구간 CAM 클립 ({uuid}_event{N}.mp4)
├── data/                       ← 학습/평가 데이터 (mp4 + txt 쌍)
├── weights/                    ← 학습된 가중치 (hitandrun_model_best.pth)
├── docker-compose.yml          ← mysql + redis
└── IMPLEMENTATION_PLAN.md      ← 상세 구축 계획
```

---

## 실행 방법 (로컬, CPU 추론)

GPU/대용량 스토리지 서버가 없는 환경을 가정한다. **루트 `./venv`는 Python 3.14라 torch/opencv 불가** → ML 실행은 별도 Python 3.11 환경(`./venv311`, uv로 생성) 사용.

```bash
# 0) (최초 1회) Python 3.11 환경 구성 — uv 사용 (uv venv엔 pip이 없으니 uv pip 사용)
#    팀원 재현용 고정 버전 파일: requirements-service.txt (검증 완료)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.11 venv311
uv pip install --python venv311 -r requirements-service.txt
uv pip install --python venv311 --index-url https://download.pytorch.org/whl/cpu torch==2.12.0 torchvision==0.27.0

# 1) DB·Redis
docker compose up -d db redis

# 2) 백엔드(웹) — backend/ 에서 실행
cd backend && ../venv311/bin/python -m uvicorn app.main:app --port 8000

# 3) Celery 워커 — backend/ 에서 실행
cd backend && ../venv311/bin/celery -A app.worker worker --concurrency=1 --loglevel=info

# 4) UI (vite 프록시로 /api → :8000)
cd ui && npm install && npm run dev
```

**CAM 영상 저장 위치**: `storage/clips/{uuid}_event{N}.mp4`. DB `crash_events.cam_heatmap_path`엔 `clips/...` 상대경로 저장, `GET /api/events/{event_id}/clip`로 서빙. 업로드 원본은 `storage/uploads/`.

---

## ML 컴포넌트 (`model/`)

### CLI 실행
```bash
python -m model.main --mode train     # 학습
python -m model.main --mode predict   # 단일 영상 CAM 예측(전체영상 출력)
python -m model.main --mode eval      # 폴더 정확도 평가
```

### 설정 (`model/config.py`)
모든 경로·하이퍼파라미터는 여기서만 수정. 경로는 `_ROOT`(프로젝트 루트) 기준 상대경로 자동 계산.

**디바이스는 학습/추론 분리:**
```python
TRAIN_DEVICE_TYPE = "cuda"   # 학습은 GPU 기본
INFER_DEVICE_TYPE = "cpu"    # 예측·평가(서비스 워커)는 CPU
```
`device_utils.get_device(device_type)`가 인자로 디바이스 타입을 받는다. CPU 추론 속도를 위해 `PREDICT_WINDOW_STRIDE`는 15로 상향(GPU면 1로 낮춰 정확도↑).

### 모델 (`hitandrun_model.py`)
**torchvision S3D** 백본(분리 컨볼루션 I3D, ≈7.9M 파라미터) + `1×1×1 Conv3d` 분류 헤드(`head_conv`, CAM 가중치 겸용). 학습 시 `config.PRETRAINED=True`로 **Kinetics-400 사전학습** 초기화 후 미세조정(백본 LR ×0.1). CAM hook 대상 `inception5b`는 S3D 마지막 인셉션 블록을 가리키는 **property 별칭**이라 `predict_cam.py`가 수정 없이 동작.
> ⚠️ 가중치는 **S3D 구조**(state_dict 키 `features.*`, `head_conv.*`)여야 로드된다. 구 I3D 가중치와 호환되지 않음. 입력 정규화는 `config.NORM_MEAN/STD`(Kinetics-400 통계)로 일원화 — 학습·추론이 반드시 같은 값을 써야 한다.

### 추론·CAM (`predict_cam.py`)
두 진입점:
- **`predict_hit_and_run_final(...)`**: CLI용. 전체 길이 영상에 CAM 합성 출력. txt(차량 bbox) 파일 필요.
- **`predict_events_and_clips(model, video_path, bbox, output_dir, ...)`**: **서비스용**. bbox를 직접 인자로 받아(txt 불필요) 슬라이딩 윈도우 추론 → **사고 의심 구간만** 짧은 CAM 오버레이 클립으로 렌더링. 반환: `[{start_frame,end_frame,start_sec,end_sec,crash_prob,clip_path}]`. 전체 영상 재생성을 피해 멀티-아워 영상에도 효율적.

`inception5b` forward hook으로 feature map을 받아 `head_conv.weight[pred_class]`로 가중합 → CAM 히트맵 생성. `activation` dict는 단일 스레드 추론 전용.

### 해상도/fps 유연성
- **해상도**: bbox 주변을 정사각형 크롭 후 224×224 리사이즈 → 어떤 해상도든 흡수. UI가 화면 좌표를 원본 해상도로 환산해 전달.
- **fps**: `timestamp_sec`·클립 fps에 영상 실측 fps 사용. 모델은 항상 30프레임 윈도우를 보므로 fps에 따라 윈도우의 실시간 길이가 달라지지만(24fps=1.25초), 동작에 큰 문제 없음. 학습(30fps) 환경에 더 맞추려면 시간 리샘플링 옵션 추가 가능.

---

## 백엔드 (`backend/app/`)

### DB 모델 (`db_models.py`)
`users → videos → analysis_tasks → crash_events` (crash_events는 video_id도 보유, 빠른 조회용 반정규화).
- **User**: username/password_hash/name/email/role
- **Video**: video_path(storage 상대경로)/recording_date/width/height/fps/total_frames
- **AnalysisTask**: bbox(xmin..xmax)/celery_task_id/status(PENDING→PROCESSING→SUCCESS/FAILURE)/error_message
- **CrashEvent**: 사고구간 1건 = 시작/종료 frame·sec + crash_prob + cam_heatmap_path(클립경로)

> DB 스키마 변경 시 `create_all`은 기존 테이블을 ALTER 하지 않는다. 개발 중 컬럼 추가 후 충돌하면 `Base.metadata.drop_all` 후 `create_all`로 재생성(데이터 없을 때).

### 주요 엔드포인트
| 메서드 | 경로 | 설명 |
|---|---|---|
| POST | `/api/auth/signup` · `/login` | 회원가입 / 로그인(간소 토큰 `tok_{id}`) |
| POST | `/api/videos` | multipart 업로드 + recording_date, cv2로 메타추출 |
| GET | `/api/videos[?days=]` · `/{id}` | 목록 / 상세(+events) |
| GET | `/api/videos/{id}/stream` | 원본 mp4 스트리밍(Range, 인증 미적용 — `<video>`용) |
| POST | `/api/videos/{id}/analyze` | bbox로 AnalysisTask 생성 + Celery enqueue |
| GET | `/api/tasks/{id}` | 분석 상태 + 완료 시 events |
| GET | `/api/events/{id}/clip` | 사고구간 CAM 클립 스트리밍(인증 미적용) |

### 모델 통합 (`prediction_job.py` + `worker.py`)
torch/opencv/모델 임포트는 **태스크 내부에서 지연 로딩** → 웹 프로세스는 ML 의존성 없이도 `.delay()` 호출 가능. 워커는 모델을 프로세스당 1회 로드 후 재사용. 인증/보안은 MVP 수준(추후 강화).

---

## 프론트엔드 (`ui/`)

`react-router-dom` 기반 URL 라우팅: `/login`, `/signup`, `/videos`(목록), `/videos/:videoId`(재생), `/analytics`(통계). 미인증 시 `/login` 가드.

- **API 클라이언트**: `src/api.js` — fetch 래퍼 + 토큰(localStorage) + `normalizeVideo`(API 응답을 UI 형태로 정규화).
- **재생 화면 핵심**: 실제 `<video>`로 원본 재생. '사고 차량 지정하기'로 **단일 bbox** 드래그(라벨 없음). '사고감지 실행' → 확인 팝오버 → bbox를 원본 해상도로 환산해 `/analyze` 호출 → 태스크 폴링 → 완료 시 이벤트 갱신. bbox 미지정 시 경고 토스트.
- **CAM 클립 조회**: 좌측 이벤트 목록의 ▶ 버튼 → 팝업 모달에서 `/api/events/{id}/clip` 재생.
- vite proxy: `/api → http://localhost:8000`.

---

## 알려진 환경 제약
- 루트 `./venv`(Python 3.14)에는 torch/opencv/celery 휠이 없다. ML 실행은 `./venv311`(uv, Python 3.11) 사용.
- 실 인프라(GPU 서버·대용량 스토리지) 없음 → CPU 추론 + 로컬 `storage/`로 대체. 실시간 CCTV 스트림은 향후 확장.
