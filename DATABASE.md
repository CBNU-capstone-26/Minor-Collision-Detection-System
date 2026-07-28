# DB 스키마

---

## 테이블 관계도

```
users
 └── videos          (user_uid → users.uid)
      └── analysis_tasks  (video_id → videos.video_id)
           └── crash_events    (task_id  → analysis_tasks.task_id)
                               (video_id → videos.video_id)  ← 반정규화
```

---

## 1. `users` — 사용자 테이블

| 컬럼명 | 타입 | 제약 | 설명 |
|---|---|---|---|
| `uid` | VARCHAR(100) | PK | 로그인 아이디 겸 고유 식별자 |
| `name` | VARCHAR(100) | NOT NULL | 사용자 실명 |
| `password_hash` | VARCHAR(255) | NOT NULL | 암호화된 비밀번호 |
| `email` | VARCHAR(255) | UNIQUE | 이메일 주소 |
| `role` | VARCHAR(50) | NOT NULL | `USER` (일반) / `ADMIN` (CCTV 관리자) |
| `created_at` | DATETIME | NOT NULL | 계정 생성 일시 |

---

## 2. `videos` — 영상 테이블

| 컬럼명 | 타입 | 제약 | 설명 |
|---|---|---|---|
| `video_id` | INT | PK, AUTO_INCREMENT | 비디오 고유 번호 |
| `user_uid` | VARCHAR(100) | FK → users.uid | 업로드한 사용자 |
| `video_name` | VARCHAR(255) | NOT NULL | 원본 파일명 |
| `video_path` | VARCHAR(500) | NOT NULL | 서버 내 저장 경로 |
| `recording_date` | DATE | | CCTV 녹화 일자 |
| `width` | INT | | 가로 픽셀 수 |
| `height` | INT | | 세로 픽셀 수 |
| `fps` | FLOAT | | 초당 프레임 수 |
| `total_frames` | INT | | 총 프레임 수 |
| `created_at` | DATETIME | NOT NULL | 업로드 일시 |

---

## 3. `analysis_tasks` — 분석 작업 테이블

| 컬럼명 | 타입 | 제약 | 설명 |
|---|---|---|---|
| `task_id` | INT | PK, AUTO_INCREMENT | 분석 작업 고유 번호 |
| `video_id` | INT | FK → videos.video_id | 분석 대상 비디오 |
| `celery_task_id` | VARCHAR(255) | | 비동기 AI 작업 접수 번호 |
| `bbox_xmin` | INT | NOT NULL | 차량 좌상단 X 좌표 |
| `bbox_ymin` | INT | NOT NULL | 차량 좌상단 Y 좌표 |
| `bbox_xmax` | INT | NOT NULL | 차량 우하단 X 좌표 |
| `bbox_ymax` | INT | NOT NULL | 차량 우하단 Y 좌표 |
| `status` | VARCHAR(50) | NOT NULL | `PENDING` / `PROCESSING` / `SUCCESS` / `FAILURE` |
| `error_message` | TEXT | | 에러 원인 (정상 시 NULL) |
| `created_at` | DATETIME | NOT NULL | 분석 요청 일시 |

---

## 4. `crash_events` — 충돌 의심 이벤트 테이블

| 컬럼명 | 타입 | 제약 | 설명 |
|---|---|---|---|
| `event_id` | INT | PK, AUTO_INCREMENT | 이벤트 고유 번호 |
| `task_id` | INT | FK → analysis_tasks.task_id | 해당 분석 작업 |
| `video_id` | INT | FK → videos.video_id | 원본 비디오 (빠른 조회용) |
| `timestamp_sec` | FLOAT | NOT NULL | 충돌 의심 시작 시점 (초) |
| `frame_number` | INT | NOT NULL | 충돌 의심 시작 프레임 번호 |
| `crash_prob` | FLOAT | | AI 예측 충돌 확률 |
