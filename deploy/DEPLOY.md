# SIOT 서버 배포 (시연용)

화면·백엔드·분석 워커를 실행하고 Supabase/Azure PostgreSQL과 Redis URL, Cloudflare R2(S3 호환) 또는 로컬 저장소를 환경변수로 선택한다.

```
브라우저 ──HTTPS──▶ nginx (80·443)
                     ├─ 화면 파일 (/var/www/siot ← ui/dist)
                     └─ /api ──▶ 백엔드 uvicorn (127.0.0.1:8000) ──┐
                                                                  ├─▶ PostgreSQL (Supabase/Azure 또는 MySQL)
                                 분석 워커 Celery ─────────────────┘
                                 (Redis URL, 영상·클립: R2 또는 storage/)
```

- 대상: **Ubuntu 22.04 또는 24.04** — Oracle Cloud(ARM, CPU 추론) / AWS 등(x86, NVIDIA GPU 추론) 모두 같은 순서
- 서버에 넣는 파일: 이 폴더(`deploy/`)의 nginx·systemd 설정을 `install.sh` 가 설치한다
- 개발용 `dev.sh` 는 그대로 쓰면 된다(서버 설정은 `.env` 가 있을 때만 적용)
- 보안: 모든 API 가 로그인을 확인한다(영상·썸네일·클립은 로그인 때 받는 HttpOnly 쿠키로, 본인 것만).
  비밀번호는 PBKDF2(소금+60만 회)로 저장하고, nginx 가 로그인·가입을 같은 IP 분당 10번으로 제한한다
- 영상 보관: 업로드한 지 90일(`VIDEO_RETENTION_DAYS`)이 지나면 영상과 사고 클립·원본 클립·썸네일·분석 기록을
  모두 지운다(백엔드가 켜질 때 한 번, 그 뒤 1시간마다). 목록 카드에 '○일 뒤 만료'로 보인다

---

## 0. 서버 만들 때

- **디스크**: 영상이 쌓이므로 100GB 이상
- **포트**: 22(SSH), 80, 443 을 연다
  - Oracle: VCN 보안 목록(Security List)에 80·443 인바운드 추가 **+ 서버 안 방화벽도** 열어야 한다(Oracle Ubuntu 이미지는 SSH 말고 다 막혀 있다)
    ```bash
    sudo iptables -I INPUT -p tcp -m multiport --dports 80,443 -m state --state NEW -j ACCEPT
    sudo netfilter-persistent save
    ```
  - AWS: 보안 그룹 인바운드에 22(내 IP), 80, 443
- **도메인**(HTTPS 용): DuckDNS 같은 무료 주소에 서버 공인 IP 를 연결해 둔다. 없으면 `http://<IP>` 로만 접속(분석 완료 브라우저 알림은 HTTPS 에서만 된다)

## 1. 기본 프로그램 (sudo)

```bash
sudo apt update
sudo apt install -y git curl rsync nginx certbot python3-certbot-nginx ffmpeg libgl1 libglib2.0-0
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER          # 다시 로그인하면 sudo 없이 docker

# 화면 빌드용 Node 22 — 서버에서 빌드할 때만. (로컬에서 npm run build 한 ui/dist 를 복사해도 된다)
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt install -y nodejs
```

GPU 서버만: NVIDIA 드라이버가 있어야 한다 — `nvidia-smi` 가 GPU 를 보여 주면 된다
(AWS Deep Learning AMI 는 이미 설치돼 있음. 아니면 `sudo ubuntu-drivers install` 후 재부팅).

## 2. 코드와 Python 환경

```bash
git clone <저장소 주소> ~/capstone-26 && cd ~/capstone-26
git checkout <배포할 브랜치>

curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
uv venv --python 3.11 venv311

# torch 를 먼저 — 서버에 맞는 줄 하나만
uv pip install --python venv311 --index-url https://download.pytorch.org/whl/cpu   torch==2.12.0 torchvision==0.27.0   # CPU 서버(Oracle ARM 등)
uv pip install --python venv311 --index-url https://download.pytorch.org/whl/cu126 torch==2.12.0 torchvision==0.27.0   # NVIDIA GPU (nvidia-smi 의 CUDA Version 12.6 이상. 13.0 이상이면 cu130 도 됨)

uv pip install --python venv311 -r requirements.txt
```

torch 를 먼저 깔아야 `requirements.txt` 가 다른 torch(수 GB)를 한 번 더 받지 않는다.
2026-10-18 확인: Ubuntu 22.04 이상이면 ARM·x86 모두 필요한 패키지가 다 있다(torch 2.12.0 은 cu128 판이 없다).

## 3. 가중치 (로컬 PC → 서버)

git 에 없는 파일이라 직접 복사한다.

```bash
# 로컬 PC 의 capstone-26 폴더에서
scp weights/hitandrun_x3d_260922_33ep_earlyY_ptY_0.1999.pth <계정>@<서버>:~/capstone-26/weights/
scp yolo11x.pt <계정>@<서버>:~/capstone-26/        # 없으면 첫 차량 탐지 때 자동으로 받는다
```

다른 가중치를 쓰려면 파일을 복사하고 `.env` 의 `SERVICE_WEIGHTS` 에 경로를 적는다.

## 4. 설정 파일 `.env`

```bash
cp deploy/env.example .env
nano .env
```

| 값 | 넣을 것 |
|---|---|
| `DOMAIN` | 도메인(없으면 비움) |
| `DATABASE_URL` | Supabase/Azure PostgreSQL URL. 비우면 loopback 전용 로컬 MySQL을 사용한다 |
| `REDIS_URL` | 로컬 Redis 또는 Azure Cache URL. Azure는 `rediss://`와 TLS 포트를 사용한다 |
| `STORAGE_BACKEND` | `s3`(Cloudflare R2/S3) 또는 `local`. R2는 endpoint·bucket·두 키를 모두 입력한다 |
| `AUTH_SECRET_KEY` | `openssl rand -hex 32` 결과 |

URL에 비밀번호를 넣을 때 `@`, `#`, `/`, `:` 같은 문자는 percent-encode 해야 한다. 로컬 MySQL·Redis 컨테이너는 loopback 주소에만 포트를 연다.

## 5. 설치 — 처음에도, 코드 갱신 후에도 이것 하나

```bash
sudo ./deploy/install.sh
```

하는 일: 화면 빌드 → `/var/www/siot` 복사 → Redis(docker) 켜기 → 백엔드·워커를 서비스로 등록·재시작 →
nginx 설정·재시작 → 백엔드 응답 확인. 끝나면 접속 주소를 알려 준다.
설치 전에 만들어질 파일만 보고 싶으면 `./deploy/install.sh --dry-run /tmp/siot-check`.

분석 실행 방식 미리 검사(안 해도 첫 분석 때 저절로 한다 — 미리 하면 첫 분석이 빨라진다):

```bash
set -a; . ./.env; set +a
venv311/bin/python -m model.main --mode check-backend
```

## 6. HTTPS (도메인이 있을 때)

```bash
sudo certbot certonly --nginx -d <도메인> --deploy-hook "systemctl reload nginx"
sudo ./deploy/install.sh        # 인증서를 찾아 HTTPS 설정으로 바꾼다(http 는 https 로 넘김)
```

인증서 갱신은 certbot 이 자동으로 한다.

## 7. 확인

- 서버 밖(휴대폰 LTE 등)에서 접속 → 회원가입 → 업로드 → 차량 지정 → 분석 → 클립 재생
- Windows(Chrome·Edge), Mac(Safari) 에서 영상 재생
- `sudo reboot` 후에도 저절로 다시 뜨는지

```bash
systemctl status siot-backend siot-worker nginx
journalctl -u siot-backend -f        # 백엔드 로그
journalctl -u siot-worker -f         # 분석 워커 로그
sudo tail -f /var/log/nginx/error.log
```

## 8. 코드 갱신

```bash
cd ~/capstone-26 && git pull
uv pip install --python venv311 -r requirements.txt     # requirements.txt 가 바뀌었을 때만
sudo ./deploy/install.sh
```

진행 중이던 분석은 재시작 때 끊기고 '실패'로 바뀐다 — 화면에서 다시 실행하면 된다.

## 문제 해결

| 증상 | 확인할 것 |
|---|---|
| 접속이 안 됨 | 0단계 포트(Oracle 은 보안 목록 + 서버 iptables 둘 다), `systemctl status nginx` |
| 502 Bad Gateway | 백엔드가 꺼짐 → `journalctl -u siot-backend -n 50` |
| PostgreSQL 연결 실패 | DATABASE_URL의 비밀번호 인코딩·SSL 설정, Supabase/Azure 방화벽과 허용 IP 확인 |
| R2 업로드·재생 실패 | STORAGE_BACKEND=s3, endpoint·bucket·Access Key·Secret Key 확인 |
| 분석이 '대기'에서 안 넘어감 | `journalctl -u siot-worker -n 50` |
| 업로드 413 | nginx 설정이 안 들어감 → `sudo ./deploy/install.sh` 다시. Cloudflare 등을 거치면 무료 플랜은 100MB 제한 |
| `libGL.so.1` 오류 | 1단계 `libgl1 libglib2.0-0` 설치 |
| 분석이 느림 | CPU 서버는 느리다. `check-backend` 결과(고른 실행 방식)와 `journalctl -u siot-worker` 의 속도 확인 |
| 로그인이 '요청이 너무 많습니다' | 같은 IP 에서 로그인·가입을 몰아서 시도함 — 1분쯤 기다리면 풀린다(siot-http/https.conf 의 rate) |
| 썸네일·영상이 안 나옴(401) | 영상 쿠키가 없음 — 다시 로그인. HTTPS 로 받은 쿠키는 HTTP 주소에선 안 실린다 |

## 시연 뒤 보완할 것

- DB 백업(`docker exec ... mysqldump`)과 `storage/` 백업
- 업로드 크기 제한이 없다(로그인한 사용자가 디스크를 채울 수 있음) · 회원가입이 누구에게나 열려 있다
