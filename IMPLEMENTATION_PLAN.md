# 물피도주 자동감지 웹 서비스 — E2E MVP 구축 계획

## Context (배경)

`model/`(I3D 3D-CNN 추론 + CAM), `weights/hitandrun_model_best.pth`, `ui/`(React+Vite, 현재 mock 데이터)는 각각 완성돼 있으나 서로 연결돼 있지 않다. 목표는 **업로드 → 사고차량 bbox 지정 → AI 분석 → 사고구간 CAM 클립 조회**까지 동작하는 수직 슬라이스(E2E MVP 골격)를 FastAPI + MySQL + Redis + Celery로 구축하는 것이다.

### 확정된 제약/결정사항
- **bbox 소스**: 사용자가 영상 재생 화면에서 마우스 드래그로 대상 차량을 직접 지정(단일 bbox). 자동 detector 없음.
- **영상 입력**: 녹화 파일 업로드. 업로드 시 **녹화일자(달력 팝업)** 지정. (실시간 스트림은 향후 확장)
- **AI 출력**: 사고 의심 *구간(start~end 프레임/초)* + 확률 반환. 유형 분류 없음.
- **CAM 표시 방식**: 전체 길이 결과 영상을 재생성하지 **않는다**(멀티-아워 비효율). AI가 반환한 **사고 구간만** 짧은 CAM 오버레이 클립으로 생성 → 사용자는 **원본 영상**을 보고, 좌측 이벤트 목록의 **재생 버튼** → **팝업창**에서 해당 구간 CAM 클립 재생.
- **인프라 제약**: GPU·대용량 스토리지 서버 없음 → **CPU 추론 + 로컬 파일시스템 저장**. MySQL·Redis는 기존 `docker-compose.yml`로 로컬 구동.
- **디바이스 분리**: 학습은 `cuda` 기본 유지, 추론(예측·평가)은 `cpu` 별도 설정.
- **영상 재생**: 실제 `<video>` 태그. bbox 좌표를 원본 해상도로 환산해 전달.
- **인증**: 간소 수준(보안 미고려). 회원가입 페이지 신규 추가.
- **폴더 구조**: CLAUDE.md 명세 outdated. 최적 구조로 재설계.

---

## 1. 폴더 구조 (재설계) + 백엔드 파일명(역할 기반 직관적 명명)

루트의 `server.py`/`database.py`/`db_schema.py`는 `backend/app/`로 흡수·정리 후 삭제.

```
capstone-26/
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI 앱, CORS, 라우터 등록, 시작 시 테이블 생성
│   │   ├── settings.py        # 설정(DB URL, REDIS URL, STORAGE 경로)
│   │   ├── db_connection.py   # engine, SessionLocal, get_db 의존성
│   │   ├── db_models.py       # SQLAlchemy 모델 (기존 db_schema.py 이전 + 보완)
│   │   ├── api_schemas.py     # Pydantic 요청/응답 스키마
│   │   ├── auth_guard.py      # 현재 사용자(간소 토큰) 의존성
│   │   ├── worker.py          # Celery 인스턴스 (broker/backend = redis)
│   │   ├── prediction_job.py  # AI 추론 Celery 태스크 (모델 래핑 + 사고구간 클립 생성)
│   │   └── routers/
│   │       ├── auth.py        # 회원가입 / 로그인
│   │       ├── videos.py      # 업로드 / 목록 / 상세 / 스트리밍
│   │       └── analysis.py    # 분석 요청 / 태스크 상태 / 이벤트·클립 조회
│   └── requirements.txt       # fastapi, uvicorn, sqlalchemy, pymysql, celery, redis, python-multipart, opencv-python
├── model/                     # 기존 유지 (config 디바이스 분리만)
├── ui/                        # 라우팅 + bbox 수정 + 회원가입 + 업로드 + 클립 팝업 + API 연동
├── storage/                   # GPU/스토리지 서버 대체 — 로컬 저장소
│   ├── uploads/               # 업로드 원본 mp4
│   └── clips/                 # 사고구간 CAM 오버레이 클립 (이벤트별)
├── weights/  ·  data/  ·  docker-compose.yml(mysql+redis 유지)
```

---

## 2. 백엔드 — FastAPI

### 2.1 DB 모델 보완 (`backend/app/db_models.py`, 기존 `db_schema.py` 기반)
- **User**: 회원가입을 위해 `name`, `email` 컬럼 추가(현재 username/password_hash/role만 존재).
- **Video**: 변경 없음. `recording_date`에 업로드 시 받은 녹화일자 저장.
- **AnalysisTask**: 변경 없음(bbox, status, celery_task_id). 전체 결과 영상은 만들지 않으므로 result 컬럼 불필요.
- **CrashEvent**(이벤트별 = 사고구간별):
  - `timestamp_sec`, `frame_number` → 구간 **시작** 시점.
  - **추가**: `end_frame_number`, `end_timestamp_sec` — 구간 종료(클립 길이 표시용).
  - `crash_prob` — 해당 구간 대표 신뢰도(추론 시 함께 반환, NOT NULL 가능).
  - `cam_heatmap_path` → **해당 사고구간 CAM 클립 파일 경로**로 사용(이름 그대로 활용, `storage/clips/`).

### 2.2 API 엔드포인트
| 메서드 | 경로 | 설명 |
|---|---|---|
| POST | `/api/auth/signup` | `{username,name,email,password}` → 사용자 생성 |
| POST | `/api/auth/login` | `{username,password}` → `{token,user}` (간소 토큰) |
| GET | `/api/videos?days=7` | 기간 필터 영상 목록 |
| POST | `/api/videos` | multipart 업로드 + `recording_date` → `storage/uploads/` 저장, `cv2`로 width/height/fps/total_frames 추출 후 Video 생성 |
| GET | `/api/videos/{id}` | 영상 상세 + crash_events |
| GET | `/api/videos/{id}/stream` | **원본** mp4 스트리밍(Range 지원) — `<video>` 재생용 |
| POST | `/api/videos/{id}/analyze` | `{bbox_xmin,ymin,xmax,ymax}` → AnalysisTask 생성 + Celery enqueue → `{task_id,celery_task_id}` |
| GET | `/api/tasks/{task_id}` | 분석 상태(PENDING/PROCESSING/SUCCESS/FAILURE) + 완료 시 events |
| GET | `/api/videos/{id}/events` | 타임라인 마커 + 이벤트 목록 |
| GET | `/api/events/{event_id}/clip` | **사고구간 CAM 클립** 스트리밍 (팝업 재생용) |
| GET | `/api/analytics?days=` | (선택) 통계 집계 |

### 2.3 모델 통합 (`backend/app/prediction_job.py` + `worker.py`)
- **Celery**: broker/result backend = `redis://localhost:6379`. 워커 init 시 모델 **1회 로드** 후 재사용.
- **bbox 전달**: AnalysisTask bbox로 임시 txt(`car,0,{xmin},{ymin},{xmax},{ymax}`) 생성해 전달(또는 predict가 bboxes dict를 직접 받도록 소폭 리팩터). `target_id=0`.
- **사고구간 클립 생성(핵심 변경)**: 기존 `predict_hit_and_run_final`(model/predict_cam.py:75)은 *전체 길이* 영상을 출력하므로, 이를 **구간-한정 클립** 방식으로 적응한다:
  1. 슬라이딩 윈도우 추론 → events(`{start_frame, end_frame}` + 구간 대표 confidence) 산출. (기존 이벤트 추적 로직 재사용.)
  2. **각 이벤트 구간 `[start_frame, end_frame]`에 대해서만** CAM 오버레이 프레임을 렌더링해 짧은 클립을 `storage/clips/`에 저장. (기존 CAM 계산·크롭·오버레이 헬퍼 `_crop_square_and_pad`, CAM 히트맵 합성 로직 재사용 — 출력 범위만 구간으로 제한.)
  - 전체 길이 영상 재생성은 하지 않음 → 멀티-아워 영상도 효율적.
- **태스크 흐름**: status=PROCESSING → 추론·클립 생성 → 이벤트별 CrashEvent 저장(start/end frame·sec, crash_prob, cam_heatmap_path=클립경로) → status=SUCCESS. 예외 시 FAILURE + error_message.

### 2.4 디바이스 분리 (`model/config.py` + `model/device_utils.py`)
- `model/config.py`: `DEVICE_TYPE` 단일값을 **`TRAIN_DEVICE_TYPE="cuda"`**(학습 기본 유지) / **`INFER_DEVICE_TYPE="cpu"`**(예측·평가)로 분리.
- `device_utils.get_device()`가 device_type 인자를 받도록 소폭 리팩터. `train.py`는 train 디바이스, `predict_cam.py`/`evaluate.py`는 infer 디바이스 사용.
- **CPU 추론 성능**: stride=1은 느리므로 `PREDICT_WINDOW_STRIDE`를 ~15로 상향(짧은 영상 권장).

### 2.5 로컬 저장소
- `settings.py`에 `STORAGE_DIR`. 업로드 `storage/uploads/{uuid}_{filename}`, 클립 `storage/clips/{event}_{...}.mp4`. DB엔 상대경로. FastAPI가 서빙.

---

## 3. 프론트엔드 — `ui/src/App.jsx` (+ 페이지 분리)

> 현재 1,449줄 단일 파일. 라우팅 도입과 함께 페이지 컴포넌트로 분리.

### 3.1 라우팅 도입 (`react-router-dom` 추가)
| 경로 | 페이지 |
|---|---|
| `/login` | 로그인 |
| `/signup` | **회원가입 (신규)** |
| `/videos` | 영상 목록 (현 home-view) |
| `/videos/:videoId` | 영상 재생 (현 watch-view) |
| `/analytics` | 통계 (현 AnalyticsView) |
- 공통 헤더 `Layout` + `<Outlet/>`. 미인증 시 `/login` 가드. `App.jsx`를 `pages/*` + `components/Layout`로 분리.

### 3.2 회원가입 + 로그인 (신규)
- User 스키마 기반 폼(username, name, email, password+확인) → `POST /api/auth/signup`.
- 기존 `LoginPage`(App.jsx:93)에 "회원가입" 버튼 추가 → `/signup`.

### 3.3 업로드 + 녹화일자(달력 팝업)
- 영상 목록 페이지에 **업로드 버튼** → 팝업 모달: 파일 선택 + **달력 date picker로 녹화일자 지정** → `POST /api/videos`(multipart + recording_date).

### 3.4 bbox 단일화 + 'car N' 라벨 제거 (재생 페이지)
- **단일 유지**: onMouseUp(App.jsx:1052)·onMouseLeave(App.jsx:1067)의 append를 **`setBboxList([box])`**(교체)로 변경 → 항상 1개.
- **라벨 제거**: 그려진 박스 `<span className="bbox-label">car {idx}</span>`(App.jsx:1104) 삭제. 미리보기 `car{nextIdx}` 텍스트(App.jsx:1132-1134)도 제거/좌표만.

### 3.5 '사고감지 실행' 버튼 + 팝오버 + 토스트 (신규)
- '사고 차량 지정하기' 버튼(App.jsx:854) **옆에** '사고감지 실행' 버튼 추가.
  - `bboxList` 비었으면 → **경고 토스트** `'사고를 감지할 차량을 드래그하여 주십시오.'`
  - 있으면 → **확인 팝오버** `"현재 선택하신 차량의 사고예상 구간을 탐지하시겠습니까?"` + `[실행] [뒤로가기]`.
  - `[실행]` → bbox 원본 해상도 환산(3.7) → `POST /api/videos/:id/analyze` → `GET /api/tasks/:id` 폴링 → SUCCESS 시 events refetch.
- 토스트/팝오버/모달은 외부 라이브러리 없이 자체 컴포넌트.

### 3.6 이벤트 목록 재생 버튼 → CAM 클립 팝업 (신규)
- 재생 페이지는 **원본 영상**을 재생.
- 좌측 감지 이벤트 목록(App.jsx:904) 각 행에 **재생 버튼** 추가 → 클릭 시 **팝업 모달**에서 해당 이벤트의 `GET /api/events/{event_id}/clip`(사고구간 CAM 클립) 재생.
- 타임라인 마커(App.jsx:1147)는 crash_events 기반 유지.

### 3.7 실제 `<video>` 연동 + 좌표 환산 + API 연동
- mock div(App.jsx:1023)를 `<video src="/api/videos/:id/stream">`로 교체. bbox 오버레이를 video 위 절대배치.
- video `videoWidth/videoHeight`(원본) ↔ 표시크기(`clientWidth/Height`)로 bbox **원본 픽셀 좌표 환산** 후 전송.
- `src/api.js` fetch 래퍼 추가, `mockVideos`(App.jsx:5) 제거 → `GET /api/videos`. `vite.config.js`에 `/api → http://localhost:8000` 프록시 추가.

---

## 4. 실행/구동 (로컬, GPU·스토리지 서버 대체)
- DB·Redis: `docker compose up -d db redis`.
- 백엔드: `cd backend && uvicorn app.main:app --reload` (시작 시 테이블 자동 생성).
- 워커: `cd backend && celery -A app.worker worker --loglevel=info` (CPU 추론).
- UI: `cd ui && npm install && npm run dev`.

---

## 5. 검증 (End-to-End)
1. `docker compose up -d db redis` → 백엔드·워커·UI 기동.
2. `/signup` 계정 생성 → `/login`.
3. `/videos`에서 **짧은** 테스트 mp4 + 녹화일자(달력) 업로드 → 목록 카드 노출.
4. 카드 클릭 → `/videos/:id`에서 실제 원본 영상 재생 확인.
5. '사고 차량 지정하기' → 드래그. **여러 번 그려도 박스 1개**, 'car0' 라벨 없음 확인.
6. bbox 없이 '사고감지 실행' → 경고 토스트. bbox 후 → 팝오버 → '실행'.
7. 태스크 PROCESSING→SUCCESS 전이, 타임라인 마커 + 좌측 이벤트 목록 생성 확인.
8. 이벤트 목록의 **재생 버튼** → 팝업에서 사고구간 CAM 클립 재생 확인. `storage/clips/`에 클립 생성, `crash_events` 행(시작/종료/clip 경로) 확인.

### 주의/튜닝
- CPU 추론은 느림 → 데모는 짧은 영상 + `PREDICT_WINDOW_STRIDE` 상향. 학습 디바이스는 `cuda` 유지.
- bbox 좌표 스케일링 정확도가 모델 결과 품질을 좌우 → video width/height 기반 환산 검증 필수.
