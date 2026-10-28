#!/usr/bin/env bash
# SIOT 서버 설치·갱신 — Ubuntu 서버에서 코드를 받은(또는 git pull 한) 뒤 실행한다. 여러 번 실행해도 된다.
#
#   sudo ./deploy/install.sh                설치·갱신: 화면 빌드 → Redis → 백엔드·워커 → nginx
#   ./deploy/install.sh --dry-run <폴더>     설치하지 않고 만들어질 설정 파일만 <폴더>에 쓴다(확인용)
#
# 준비물(deploy/DEPLOY.md 1~4단계): venv311, 가중치 파일, 프로젝트 루트 .env, nginx, docker
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_USER="${SUDO_USER:-$(id -un)}"
DRY=""
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY="${2:?--dry-run 뒤에 결과를 쓸 폴더를 주세요}"
fi

log()  { printf '\033[1;36m[install]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[install]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[install]\033[0m %s\n' "$*" >&2; exit 1; }

# .env 에서 값 하나 읽기 (KEY=VALUE 줄, 마지막 것)
env_get() {
  [[ -f "$REPO/.env" ]] || return 0
  sed -n "s/^$1=//p" "$REPO/.env" | tail -n 1 | tr -d '\r'
}

env_first() {
  local name value
  for name in "$@"; do
    value="$(env_get "$name")"
    if [[ -n "$value" ]]; then
      printf '%s' "$value"
      return 0
    fi
  done
  return 0
}

DOMAIN="$(env_get DOMAIN)"
SERVER_NAME="${DOMAIN:-_}"   # 도메인이 없으면 IP 로 들어오는 요청을 모두 받는다
HTTPS=""
if [[ -n "$DOMAIN" && -f "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" ]]; then
  HTTPS=1
fi
NGINX_TEMPLATE="$REPO/deploy/nginx/siot-http.conf"
[[ -n "$HTTPS" ]] && NGINX_TEMPLATE="$REPO/deploy/nginx/siot-https.conf"

render() {
  sed -e "s|__REPO__|$REPO|g" -e "s|__USER__|$RUN_USER|g" -e "s|__DOMAIN__|$SERVER_NAME|g" "$1"
}

# ── 확인용: 파일만 만들어 보고 끝 ──────────────────────────────────────
if [[ -n "$DRY" ]]; then
  mkdir -p "$DRY"
  render "$REPO/deploy/systemd/siot-backend.service" > "$DRY/siot-backend.service"
  render "$REPO/deploy/systemd/siot-worker.service" > "$DRY/siot-worker.service"
  render "$NGINX_TEMPLATE" > "$DRY/siot.conf"
  cp "$REPO/deploy/nginx/siot-common.conf" "$DRY/siot-common.conf"
  log "만들어질 파일을 $DRY 에 썼습니다 (실행 계정 $RUN_USER, 도메인 $SERVER_NAME, HTTPS ${HTTPS:-없음})"
  exit 0
fi

# ── 준비물 확인 ───────────────────────────────────────────────────────
[[ $EUID -eq 0 ]] || die "sudo 로 실행하세요: sudo ./deploy/install.sh"
[[ "$RUN_USER" != "root" ]] || die "root 계정이 아니라 평소 계정에서 sudo 로 실행하세요(서비스가 그 계정으로 돕니다)"
[[ -x "$REPO/venv311/bin/python" ]] || die "venv311 이 없습니다 — deploy/DEPLOY.md 2단계"
[[ -f "$REPO/.env" ]] || die ".env 가 없습니다 — cp deploy/env.example .env 후 값을 채우세요"
DATABASE_URL="$(env_first DATABASE_URL AZURE_POSTGRESQL_CONNECTIONSTRING)"
LOCAL_DATABASE="0"
if [[ -z "$DATABASE_URL" ]]; then
  LOCAL_DATABASE="1"
  warn "DATABASE_URL 이 비어 있어 loopback 전용 로컬 MySQL을 사용합니다"
fi
REDIS_URL="$(env_get REDIS_URL)"
LOCAL_REDIS="1"
if [[ -n "$REDIS_URL" ]]; then
  case "$REDIS_URL" in
    redis://*localhost:*|redis://*127.0.0.1:*|rediss://*localhost:*|rediss://*127.0.0.1:*) ;;
    *) LOCAL_REDIS="0" ;;
  esac
else
  REDIS_HOST="$(env_first REDIS_HOST AZURE_REDIS_HOST)"
  case "$REDIS_HOST" in
    ""|localhost|127.0.0.1|::1) ;;
    *) LOCAL_REDIS="0" ;;
  esac
fi
[[ -n "$(env_get AUTH_SECRET_KEY)" ]] || warn "AUTH_SECRET_KEY 가 비어 있어 backend/.auth_secret 에 키를 만들어 씁니다"
STORAGE_BACKEND="$(env_get STORAGE_BACKEND)"
STORAGE_BACKEND="${STORAGE_BACKEND:-local}"
if [[ "$STORAGE_BACKEND" == "s3" ]]; then
  [[ -n "$(env_first S3_ENDPOINT_URL R2_ENDPOINT_URL CLOUDFLARE_R2_ACCOUNT_ID)" ]] || die ".env 에 R2 endpoint 또는 CLOUDFLARE_R2_ACCOUNT_ID 를 채우세요"
  [[ -n "$(env_first S3_BUCKET R2_BUCKET CLOUDFLARE_R2_BUCKET)" ]] || die ".env 에 S3_BUCKET 또는 CLOUDFLARE_R2_BUCKET 을 채우세요"
  [[ -n "$(env_first S3_ACCESS_KEY_ID R2_ACCESS_KEY_ID CLOUDFLARE_R2_ACCESS_KEY_ID)" ]] || die ".env 에 R2 access key 를 채우세요"
  [[ -n "$(env_first S3_SECRET_ACCESS_KEY R2_SECRET_ACCESS_KEY CLOUDFLARE_R2_SECRET_ACCESS_KEY)" ]] || die ".env 에 R2 secret key 를 채우세요"
elif [[ "$STORAGE_BACKEND" != "local" ]]; then
  die ".env 의 STORAGE_BACKEND 는 local 또는 s3 이어야 합니다"
fi
command -v nginx >/dev/null || die "nginx 가 없습니다 — deploy/DEPLOY.md 1단계"
if [[ "$LOCAL_DATABASE" == "1" || "$LOCAL_REDIS" == "1" ]]; then
  command -v docker >/dev/null || die "로컬 DB/Redis를 쓰려면 docker가 필요합니다 — deploy/DEPLOY.md 1단계"
fi

WEIGHTS="$(sudo -u "$RUN_USER" env SERVICE_WEIGHTS="$(env_get SERVICE_WEIGHTS)" \
  "$REPO/venv311/bin/python" -c "import sys; sys.path.insert(0, '$REPO/model'); import config; print(config.SERVICE_WEIGHTS_PATH)")"
[[ -f "$WEIGHTS" ]] || die "서비스 가중치가 없습니다: $WEIGHTS — 로컬에서 복사하세요(deploy/DEPLOY.md 3단계)"
[[ -f "$REPO/yolo11x.pt" ]] || warn "yolo11x.pt 가 없어 첫 차량 탐지 때 자동으로 내려받습니다(약 110MB)"

chown "$RUN_USER" "$REPO/.env"
chmod 600 "$REPO/.env"   # 비밀번호가 들어 있다

# ── 1) 화면 빌드 → /var/www/siot ───────────────────────────────────────
NODE_MAJOR="$(sudo -u "$RUN_USER" bash -lc 'node -v 2>/dev/null' | sed -n 's/^v\([0-9]*\).*/\1/p')"
if [[ -n "$NODE_MAJOR" && "$NODE_MAJOR" -ge 20 ]]; then
  log "화면 빌드 (node v$NODE_MAJOR)"
  sudo -u "$RUN_USER" bash -lc "
    set -e; cd '$REPO/ui'
    # package-lock.json 이 바뀌었거나 처음이면 패키지를 새로 설치한다
    if [[ ! -f node_modules/.package-lock.json || package-lock.json -nt node_modules/.package-lock.json ]]; then
      npm ci --no-audit --no-fund
    fi
    npm run build"
elif [[ -f "$REPO/ui/dist/index.html" ]]; then
  warn "node 20 이상이 없어 이미 있는 ui/dist 를 그대로 씁니다(로컬에서 빌드해 복사한 것인지 확인하세요)"
else
  die "화면 빌드 파일(ui/dist)이 없고 node 20 이상도 없습니다 — deploy/DEPLOY.md 1단계(Node) 또는 로컬 빌드 복사"
fi
install -d -m 755 /var/www/siot
if command -v rsync >/dev/null; then
  rsync -a --delete "$REPO/ui/dist/" /var/www/siot/
else
  find /var/www/siot -mindepth 1 -delete
  cp -r "$REPO/ui/dist/." /var/www/siot/
fi
chmod -R a+rX /var/www/siot

# ── 2) 영상 저장 폴더 (다른 디스크로 옮겼으면 권한을 맞춘다) ─────────────
STORAGE="$(env_get STORAGE_DIR)"
if [[ -n "$STORAGE" ]]; then
  [[ "$STORAGE" = /* ]] || STORAGE="$REPO/$STORAGE"
  install -d -o "$RUN_USER" "$STORAGE"
fi

# ── 3) 선택한 로컬 DB·Redis (docker) ──────────────────────────────────
LOCAL_SERVICES=()
if [[ "$LOCAL_DATABASE" == "1" ]]; then
  LOCAL_SERVICES+=(db)
fi
if [[ "$LOCAL_REDIS" == "1" ]]; then
  LOCAL_SERVICES+=(redis)
fi
if (( ${#LOCAL_SERVICES[@]} > 0 )); then
  log "로컬 서비스 기동: ${LOCAL_SERVICES[*]} (loopback 전용)"
  docker compose --project-directory "$REPO" -f "$REPO/docker-compose.yml" up -d "${LOCAL_SERVICES[@]}"
else
  log "외부 PostgreSQL·Redis 설정을 사용합니다"
fi

# ── 4) 백엔드·워커 서비스 ──────────────────────────────────────────────
log "백엔드·워커 서비스 등록·재시작 (실행 계정 $RUN_USER)"
for unit in siot-backend siot-worker; do
  render "$REPO/deploy/systemd/$unit.service" > "/etc/systemd/system/$unit.service"
done
systemctl daemon-reload
systemctl enable siot-backend siot-worker >/dev/null
systemctl restart siot-backend siot-worker

# ── 5) nginx ──────────────────────────────────────────────────────────
log "nginx 설정 (${HTTPS:+HTTPS, }도메인 $SERVER_NAME)"
install -m 644 "$REPO/deploy/nginx/siot-common.conf" /etc/nginx/snippets/siot-common.conf
render "$NGINX_TEMPLATE" > /etc/nginx/sites-available/siot.conf
ln -sf /etc/nginx/sites-available/siot.conf /etc/nginx/sites-enabled/siot.conf
if [[ -L /etc/nginx/sites-enabled/default ]]; then
  rm /etc/nginx/sites-enabled/default   # 기본 환영 페이지 끄기(원본은 sites-available/default 에 그대로)
  log "nginx 기본 환영 페이지를 껐습니다"
fi
nginx -t
systemctl enable nginx >/dev/null
systemctl reload nginx 2>/dev/null || systemctl restart nginx

# ── 6) 확인 ───────────────────────────────────────────────────────────
log "백엔드가 뜨기를 기다립니다(최대 90초)"
for _ in $(seq 1 45); do
  if curl -fsS http://127.0.0.1:8000/ >/dev/null 2>&1; then
    if [[ -n "$HTTPS" ]]; then URL="https://$DOMAIN"; else URL="http://${DOMAIN:-<서버 공인 IP>}"; fi
    log "완료 — 접속 주소: $URL"
    [[ -n "$DOMAIN" && -z "$HTTPS" ]] && log "HTTPS 는 deploy/DEPLOY.md 6단계(certbot) 후 이 스크립트를 한 번 더 실행하면 켜집니다"
    exit 0
  fi
  sleep 2
done
die "백엔드가 응답하지 않습니다 — journalctl -u siot-backend -n 50 으로 오류를 확인하세요"
