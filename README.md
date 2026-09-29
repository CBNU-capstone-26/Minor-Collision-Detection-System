# 팀원용 Supabase·Cloudflare R2 사용 안내

이 문서는 이 프로젝트의 다른 팀원이 본인의 계정을 사용해 기존 팀 Supabase 프로젝트와 Cloudflare R2 버킷에 접근하기 위한 안내서입니다.

팀원은 각자의 Supabase·Cloudflare 계정을 사용하고, 프로젝트 관리자가 보낸 초대를 수락한 뒤 자신의 환경 설정 파일에 필요한 값을 입력합니다.

> 주의: Supabase와 Cloudflare의 관리 권한을 받는 것과 웹 애플리케이션에서 다른 사용자의 영상을 보는 것은 별개의 문제입니다. 이 문서는 서비스 관리 화면과 저장소에 접근하는 절차를 설명합니다.

## 1. 시작하기 전에

팀원은 다음을 준비합니다.

1. 본인의 이메일 주소
2. 본인의 Supabase 계정
3. 본인의 Cloudflare 계정
4. Git과 프로젝트 저장소 접근 권한
5. Python, Docker, Node.js가 설치된 개발 환경

프로젝트 관리자에게 다음 두 가지 초대를 받아야 합니다.

- Supabase 조직 초대
- Cloudflare 계정 멤버 초대

초대를 받지 못했다면 본인의 Supabase·Cloudflare 가입 이메일을 프로젝트 관리자에게 전달합니다.

## 2. Supabase 초대 수락하기

### 2.1 이메일 초대 수락

1. Supabase 초대 이메일을 엽니다.
2. 이메일 안의 Accept invitation 버튼을 클릭합니다.
3. 이미 Supabase 계정이 있으면 로그인합니다.
4. 계정이 없으면 초대받은 이메일 주소로 Supabase 계정을 생성합니다.
5. Supabase Dashboard에 접속합니다.
6. 화면 상단의 조직 선택 메뉴를 엽니다.
7. 초대한 조직을 선택합니다.
8. Projects 목록에 이 프로젝트가 표시되는지 확인합니다.

프로젝트가 표시되지 않는다면 다음을 확인합니다.

- 초대받은 이메일과 로그인한 Supabase 이메일이 같은지 확인합니다.
- 개인 조직이 아니라 초대받은 팀 조직을 선택했는지 확인합니다.
- 초대 이메일이 만료되지 않았는지 확인합니다.
- 프로젝트 관리자에게 초대를 다시 보내 달라고 요청합니다.

### 2.2 Supabase 권한 확인

본인과 같은 수준의 Supabase 권한을 받은 경우 다음 경로에서 역할을 확인할 수 있습니다.

1. Supabase Dashboard에서 초대받은 조직을 선택합니다.
2. Organization Settings를 엽니다.
3. Team 또는 Members를 선택합니다.
4. 본인의 이메일을 찾습니다.
5. 역할이 Owner인지 확인합니다.

Owner는 데이터베이스, 프로젝트 설정, API 키, 멤버 관리 등 조직과 프로젝트에 대한 광범위한 권한을 가집니다. 프로젝트 관리자에게 필요한 권한을 받은 것이 맞는지 확인합니다.

## 3. Supabase DB 연결 정보 입력하기

이 프로젝트의 백엔드는 Supabase PostgreSQL에 직접 연결합니다. 따라서 Dashboard에 로그인할 수 있는 것과 별도로 로컬 백엔드가 사용할 DATABASE_URL이 필요합니다.

### 3.1 연결 문자열 복사

1. Supabase Dashboard에 로그인합니다.
2. 초대받은 조직을 선택합니다.
3. Projects 목록에서 이 프로젝트를 클릭합니다.
4. 프로젝트 화면 상단의 Connect 버튼을 클릭합니다.
5. Session pooler 또는 Shared Pooler를 선택합니다.
6. 화면에 표시된 연결 문자열을 복사합니다.
7. 문자열의 [YOUR-PASSWORD] 부분을 데이터베이스 비밀번호로 바꿉니다.

현재 프로젝트는 Shared Pooler 연결을 사용하므로 화면에 표시된 Host, Port, User 값을 그대로 사용합니다. 값을 추측해서 직접 조합하지 않는 것이 좋습니다.

연결 문자열은 다음과 같은 형태입니다.

    postgresql://postgres.<PROJECT_REF>:<DB_PASSWORD>@<POOLER_HOST>:5432/postgres?sslmode=require

DB_PASSWORD는 Supabase 로그인 비밀번호와 다를 수 있습니다. 프로젝트 관리자가 전달한 데이터베이스 비밀번호를 사용해야 합니다.

### 3.2 비밀번호 특수문자 처리

비밀번호에 다음 문자가 포함되어 있으면 URL 인코딩해야 합니다.

    @  #  ?  :  /  공백

예를 들어 비밀번호가 다음과 같다면:

    my@password#2026

연결 문자열에는 다음처럼 입력합니다.

    my%40password%232026

인코딩하지 않으면 인증 실패 또는 연결 문자열 파싱 오류가 발생할 수 있습니다.

### 3.3 로컬 환경 파일 만들기

프로젝트를 클론한 뒤 프로젝트 루트에서 실행합니다.

    git clone <저장소_URL>
    cd Minor-Collision-Detection-System
    cp backend/.env.example backend/.env

backend/.env 파일에서 DATABASE_URL을 본인이 복사한 Supabase 연결 문자열로 바꿉니다.

최소한 다음 값이 필요합니다.

    DATABASE_URL=postgresql://postgres.<PROJECT_REF>:<URL_ENCODED_PASSWORD>@<POOLER_HOST>:5432/postgres?sslmode=require
    REDIS_URL=redis://127.0.0.1:6379/0

backend/.env는 개인 파일이므로 Git에 커밋하거나 팀 채팅에 전체 내용을 올리지 않습니다.

## 4. Cloudflare 초대 수락하기

### 4.1 이메일 초대 수락

1. Cloudflare 초대 이메일을 엽니다.
2. Accept invitation 버튼을 클릭합니다.
3. 본인의 Cloudflare 계정으로 로그인합니다.
4. 계정이 없으면 초대받은 이메일 주소로 Cloudflare 계정을 생성합니다.
5. Cloudflare Dashboard에 접속합니다.
6. 계정 선택 메뉴에서 초대받은 Cloudflare 계정을 선택합니다.

### 4.2 Cloudflare 멤버 권한 확인

1. Cloudflare Dashboard에서 초대받은 계정을 선택합니다.
2. Manage Account를 엽니다.
3. Members 메뉴로 이동합니다.
4. 본인의 이메일을 선택합니다.
5. 역할을 확인합니다.

본인과 같은 수준의 Cloudflare 관리 권한을 받은 경우 역할은 Super Administrator입니다.

Super Administrator는 R2 버킷, 계정 설정, 멤버 관리, API 토큰, 결제 등 광범위한 기능에 접근할 수 있습니다. 이 권한이 보이지 않거나 R2 메뉴가 표시되지 않으면 프로젝트 관리자에게 권한 설정을 확인해 달라고 요청합니다.

## 5. 본인 R2 API 토큰 만들기

Cloudflare Dashboard에 들어갈 수 있어도 애플리케이션이 R2에 파일을 업로드하려면 별도의 S3 API 자격증명이 필요합니다. 다른 팀원의 토큰을 복사하지 말고 본인의 토큰을 만듭니다.

### 5.1 R2 토큰 생성 메뉴

1. Cloudflare Dashboard에서 초대받은 계정을 선택합니다.
2. 왼쪽 메뉴에서 Storage & databases를 엽니다.
3. R2를 선택합니다.
4. Overview 화면으로 이동합니다.
5. Account Details 영역을 찾습니다.
6. Manage 또는 Manage R2 API Tokens를 클릭합니다.
7. Create API token을 클릭합니다.

### 5.2 토큰 권한 선택

1. 토큰 이름을 입력합니다.
   - 예: minor-collision-dev-hong
2. 파일 업로드와 삭제가 필요하면 Object Read & Write를 선택합니다.
3. 파일 확인과 다운로드만 필요하면 Object Read 또는 Object Read Only를 선택합니다.
4. Apply to specific buckets only를 선택합니다.
5. 이 프로젝트에서 사용하는 R2 버킷 하나만 선택합니다.
6. 다른 버킷이나 전체 계정 권한은 선택하지 않습니다.
7. Create API Token을 클릭합니다.

### 5.3 생성 결과 저장

생성 결과 화면에서 다음 값을 확인하고 안전하게 복사합니다.

- Access Key ID
- Secret Access Key
- Account ID
- S3 Endpoint

Secret Access Key는 생성 직후에만 표시될 수 있습니다. 화면을 닫기 전에 비밀번호 관리 도구에 저장합니다.

R2 Endpoint는 보통 다음 형태입니다.

    https://<ACCOUNT_ID>.r2.cloudflarestorage.com

Secret Access Key를 분실하면 기존 토큰을 다시 확인할 수 없을 수 있으므로 기존 토큰을 폐기하고 새 토큰을 생성해야 합니다.

## 6. R2 설정을 로컬 환경에 입력하기

본인의 backend/.env에 다음 항목을 입력합니다.

    STORAGE_BACKEND=s3
    S3_ENDPOINT_URL=https://<ACCOUNT_ID>.r2.cloudflarestorage.com
    S3_BUCKET=<기존_프로젝트_R2_버킷_이름>
    S3_ACCESS_KEY_ID=<본인_R2_ACCESS_KEY_ID>
    S3_SECRET_ACCESS_KEY=<본인_R2_SECRET_ACCESS_KEY>
    S3_REGION=auto
    S3_PRESIGNED_URL_EXPIRES=3600

각 항목은 다음 위치에서 확인합니다.

| 환경변수 | 확인 위치 |
| --- | --- |
| S3_ENDPOINT_URL | R2 Overview 또는 API 토큰 생성 결과 |
| S3_BUCKET | R2 → Overview의 버킷 목록 |
| S3_ACCESS_KEY_ID | 방금 생성한 R2 API 토큰 결과 |
| S3_SECRET_ACCESS_KEY | 방금 생성한 R2 API 토큰 결과 |
| S3_REGION | 일반 R2 버킷이면 auto |

S3_BUCKET은 새 버킷을 임의로 만들지 말고 팀에서 사용하는 기존 프로젝트 버킷의 이름을 입력해야 기존 영상과 클립에 접근할 수 있습니다.

## 7. 로컬 개발 환경 실행

Supabase와 R2를 사용하더라도 이 프로젝트의 분석 작업 큐에는 Redis가 필요합니다.

프로젝트 루트에서 실행합니다.

    docker compose up -d redis

상태를 확인합니다.

    docker compose ps

FastAPI 백엔드를 실행합니다.

    PYTHONPATH=backend venv311/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

새 터미널에서 Celery worker를 실행합니다.

    PYTHONPATH=backend venv311/bin/celery -A app.worker worker --loglevel=info --concurrency=1

API 문서가 열리는지 확인합니다.

    http://localhost:8000/docs

## 8. 연결 확인 순서

### 8.1 Supabase 확인

1. FastAPI 백엔드를 실행합니다.
2. 터미널에 데이터베이스 인증 오류가 없는지 확인합니다.
3. Supabase Dashboard에서 Table Editor를 엽니다.
4. 다음 테이블이 보이는지 확인합니다.
   - users
   - videos
   - analysis_tasks
   - crash_events

### 8.2 R2 확인

1. Cloudflare Dashboard에서 Storage & databases → R2 → Overview를 엽니다.
2. 기존 프로젝트 버킷을 클릭합니다.
3. Objects 또는 파일 목록에서 기존 영상 객체를 확인합니다.
4. 애플리케이션에서 테스트 영상을 업로드합니다.
5. R2 버킷의 객체 목록을 새로고침합니다.
6. 새 영상 객체와 썸네일이 생성되었는지 확인합니다.

### 8.3 사고 클립 확인

1. 애플리케이션에서 영상 분석을 실행합니다.
2. Celery worker 터미널에 작업 수신 로그가 나타나는지 확인합니다.
3. 분석이 완료될 때까지 기다립니다.
4. R2 버킷에서 사고 분석 클립 객체가 생성되었는지 확인합니다.
5. 웹 UI에서 사고 이벤트와 클립이 재생되는지 확인합니다.

## 9. 문제 해결

### Supabase 프로젝트가 보이지 않음

- 올바른 조직을 선택했는지 확인합니다.
- 초대 이메일과 현재 로그인 이메일이 같은지 확인합니다.
- 초대가 만료되지 않았는지 확인합니다.
- 프로젝트 관리자에게 Supabase 초대를 다시 보내 달라고 요청합니다.

### Supabase DB 연결 실패

- Supabase 프로젝트의 Connect 화면에서 연결 문자열을 다시 복사합니다.
- Shared Pooler의 Host와 User를 직접 추측하지 않습니다.
- 사용자 이름이 postgres.<PROJECT_REF> 형식인지 확인합니다.
- 비밀번호 특수문자를 URL 인코딩합니다.
- backend/.env 수정 후 FastAPI를 재시작합니다.
- DB 비밀번호와 Supabase 로그인 비밀번호를 혼동하지 않았는지 확인합니다.

### R2 버킷이 보이지 않음

- 올바른 Cloudflare 계정을 선택했는지 확인합니다.
- Storage & databases → R2 → Overview로 이동했는지 확인합니다.
- R2 멤버 초대가 수락되었는지 확인합니다.
- Super Administrator 또는 R2에 필요한 권한이 있는지 관리자에게 확인합니다.

### R2 AccessDenied 또는 NoSuchBucket 오류

- S3_BUCKET이 기존 프로젝트 버킷 이름과 정확히 같은지 확인합니다.
- 토큰 권한이 해당 버킷에 적용되어 있는지 확인합니다.
- Object Read 권한만 있는 토큰으로 업로드하고 있지 않은지 확인합니다.
- Access Key ID와 Secret Access Key를 서로 바꾸지 않았는지 확인합니다.
- Endpoint에 Account ID가 포함되어 있는지 확인합니다.
- backend/.env 수정 후 FastAPI와 Celery worker를 모두 재시작합니다.

### 기존 영상이 웹 UI에 보이지 않음

Supabase와 R2에 접근할 수 있어도 현재 웹 애플리케이션은 로그인한 사용자별로 영상을 분리합니다. 기존 영상의 업로드 사용자와 현재 로그인한 사용자가 다르면 영상이 보이지 않을 수 있습니다.

이 경우 서비스 권한 문제가 아니라 애플리케이션의 user_id 접근 제어 문제입니다. 팀 전체가 같은 영상을 웹 UI에서 사용하려면 workspace 또는 팀 공유 권한 기능을 별도로 구현해야 합니다.

## 10. 보안 규칙

- backend/.env 전체를 GitHub에 올리지 않습니다.
- Supabase 연결 문자열을 공개 이슈나 팀 채팅에 올리지 않습니다.
- R2 Secret Access Key를 다른 팀원에게 전달하지 않습니다.
- 팀원이 나가면 Supabase 멤버 권한, Cloudflare 멤버 권한, R2 API 토큰을 각각 폐기합니다.
- R2 토큰은 프로젝트 버킷 하나에만 적용합니다.
- 운영 데이터와 개발 데이터를 같은 버킷에서 사용하지 않는 것이 좋습니다.
- 의심스러운 토큰 노출이 있으면 Cloudflare에서 해당 토큰을 즉시 폐기하고 새로 생성합니다.

## 11. 공식 문서

- Supabase 접근 제어: https://supabase.com/docs/guides/platform/access-control
- Supabase DB 연결: https://supabase.com/docs/guides/database/connecting-to-postgres
- Cloudflare 멤버 관리: https://developers.cloudflare.com/fundamentals/manage-members/manage/
- Cloudflare 역할: https://developers.cloudflare.com/fundamentals/manage-members/roles/
- Cloudflare R2 API 토큰: https://developers.cloudflare.com/r2/api/tokens/

