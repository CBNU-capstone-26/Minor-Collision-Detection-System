#!/usr/bin/env bash
# 개발용 서비스 한 번에 켜기/끄기 — tmux 세션 하나에 백엔드·워커·UI를 칸 3개로 띄운다.
#
# 데이터베이스는 backend/.env의 Supabase 설정을 사용한다. 이 스크립트는 로컬 SQL을
# 기동하거나 관리하지 않으며, Celery 브로커/결과 저장소인 Redis만 확인·기동한다.
#
#   ./dev.sh            서비스 시작(이미 떠 있으면 그 화면으로 들어감)
#   ./dev.sh stop       서비스 모두 종료
#   ./dev.sh restart    종료 후 다시 시작
#   ./dev.sh status     상태 확인
#   ./dev.sh attach     떠 있는 화면으로 들어가기
#
# tmux 종료 방법:
#   ./dev.sh stop               이 스크립트가 만든 서비스 세션을 정상 종료
#   tmux kill-session -t capstone  capstone 세션을 직접 강제 종료
#   tmux kill-server            실행 중인 모든 tmux 세션을 종료
#
# 코드를 저장하면 자동으로 반영된다(껐다 켤 필요 없음):
#   · 백엔드  : backend/app 의 .py 저장 → uvicorn --reload 가 재시작
#   · 워커    : backend/app · model 의 .py 저장 → watchfiles 가 재시작
#              ⚠️ 분석이 돌던 중이면 그 분석은 중단된다('서버 재시작으로 중단' → 다시 실행)
#   · UI     : Vite 가 저장 즉시 반영
#
# tmux 조작: Ctrl+b 후 방향키 = 칸 이동 · Ctrl+b d = 화면에서 나오기(서비스는 계속 돔)
#            Ctrl+b [ = 로그 위로 스크롤(q 로 종료)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="capstone"
PY="$ROOT/venv311/bin"
BACKEND_PORT=8000
UI_PORT=5173

log() { printf '\033[1;36m[dev]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[dev]\033[0m %s\n' "$*" >&2; }

port_in_use() {
  # macOS에는 ss가 기본 설치되어 있지 않으므로 lsof를 먼저 쓴다.
  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:$1 -sTCP:LISTEN >/dev/null 2>&1
  elif command -v ss >/dev/null 2>&1; then
    ss -ltn "sport = :$1" 2>/dev/null | grep -q LISTEN
  else
    return 1
  fi
}

ensure_redis() {
  if port_in_use 6379; then
    log "Redis 확인 — localhost:6379 사용 중"
    return 0
  fi

  command -v docker >/dev/null 2>&1 || {
    err "Redis가 꺼져 있고 Docker를 찾을 수 없습니다. Redis를 6379 포트에서 먼저 실행하세요."
    return 1
  }

  log "Redis 기동"
  if ! (cd "$ROOT" && docker compose up -d redis); then
    err "Redis 기동 실패 — Docker Desktop이 켜져 있는지 확인하세요."
    return 1
  fi

  for _ in $(seq 1 20); do
    port_in_use 6379 && return 0
    sleep 0.5
  done
  err "Redis가 localhost:6379에서 준비되지 않았습니다."
  return 1
}

# 워커 감싸개: watchfiles 가 재시작하려고 보내는 SIGINT 를 Celery 의 즉시 종료(SIGQUIT)로
# 바꿔 전달한다. SIGINT 를 그대로 받으면 Celery 는 진행 중 분석이 끝날 때까지 기다리는
# '정상 종료'를 해서, 코드를 고쳐도 반영이 늦고 강제 종료 시 하위 프로세스가 남을 수 있다.
run_worker() {
  cd "$ROOT/backend"
  "$PY/celery" -A app.worker worker --concurrency=1 --loglevel=info &
  local pid=$!
  trap 'kill -QUIT "$pid" 2>/dev/null || true' INT TERM
  while kill -0 "$pid" 2>/dev/null; do
    wait "$pid" || true
  done
}

start() {
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    log "이미 실행 중입니다 — 화면으로 들어갑니다 (나오기: Ctrl+b d)"
    exec tmux attach -t "$SESSION"
  fi

  # 준비물 확인
  [[ -x "$PY/python" ]] || { err "venv311 이 없습니다. CLAUDE.md 의 '실행 방법 0)' 으로 먼저 만드세요."; exit 1; }
  for p in "$BACKEND_PORT" "$UI_PORT"; do
    if port_in_use "$p"; then
      err "포트 $p 를 이미 다른 프로그램이 쓰고 있습니다(예전에 따로 띄운 서버?). 먼저 끄고 다시 실행하세요."
      exit 1
    fi
  done
  if [[ ! -d "$ROOT/ui/node_modules" ]]; then
    log "ui/node_modules 가 없어 npm install 을 먼저 실행합니다"
    (cd "$ROOT/ui" && npm install)
  fi

  ensure_redis

  log "tmux 세션 '$SESSION' 에 백엔드·워커·UI 를 띄웁니다"
  tmux new-session -d -s "$SESSION" -n services -c "$ROOT/backend" \
    "exec \"$PY/python\" -m uvicorn app.main:app --port $BACKEND_PORT --reload --reload-dir app"
  # 프로그램이 죽어도 칸을 닫지 않아 오류 로그를 볼 수 있게
  tmux set-option -t "$SESSION:services" remain-on-exit on
  tmux set-option -t "$SESSION:services" pane-border-status top
  tmux set-option -t "$SESSION" mouse on
  tmux select-pane -t "$SESSION:services.0" -T "백엔드 :$BACKEND_PORT (자동 재시작)"

  tmux split-window -v -t "$SESSION:services" -c "$ROOT/backend" \
    "exec \"$PY/watchfiles\" --filter python --sigint-timeout 15 '\"$ROOT/dev.sh\" _worker' app ../model"
  tmux select-pane -t "$SESSION:services.1" -T "Celery 워커 (자동 재시작)"

  tmux split-window -v -t "$SESSION:services" -c "$ROOT/ui" "npm run dev -- --port $UI_PORT --strictPort"
  tmux select-pane -t "$SESSION:services.2" -T "UI :$UI_PORT"
  tmux select-layout -t "$SESSION:services" even-vertical

  log "실행 완료 — UI: http://localhost:$UI_PORT · 백엔드: http://localhost:$BACKEND_PORT/docs"
  log "끄기: ./dev.sh stop · 화면에서 나오기: Ctrl+b d"
  if [[ -t 1 ]]; then
    exec tmux attach -t "$SESSION"
  fi
}

stop() {
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    log "실행 중인 서비스가 없습니다"
    return 0
  fi
  log "서비스 종료 중…"
  # 각 칸에 Ctrl+C 를 보내 정상 종료를 시도하고, 끝나길 잠깐 기다린 뒤 세션을 닫는다
  for pane in $(tmux list-panes -t "$SESSION" -F '#{pane_id}'); do
    tmux send-keys -t "$pane" C-c 2>/dev/null || true
  done
  for _ in $(seq 1 20); do
    alive=$(tmux list-panes -t "$SESSION" -F '#{pane_dead}' 2>/dev/null | grep -c 0 || true)
    [[ "$alive" == "0" ]] && break
    sleep 0.5
  done
  tmux kill-session -t "$SESSION" 2>/dev/null || true
  log "종료했습니다 (Supabase/R2와 Redis는 계속 사용 가능합니다)"
}

status() {
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    log "tmux 세션 '$SESSION' 실행 중"
    tmux list-panes -t "$SESSION" -F '  · #{pane_title} — #{?pane_dead,종료됨,동작 중}'
  else
    log "tmux 세션 없음(서비스 꺼짐)"
  fi
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$BACKEND_PORT/docs" || true)
  log "백엔드 응답: ${code:-없음}"
  if port_in_use 6379; then
    log "Redis 응답: localhost:6379 수신 중"
  else
    log "Redis 응답: 수신하지 않음 (Celery 작업은 실행할 수 없음)"
  fi
}

case "${1:-start}" in
  start)   start ;;
  stop)    stop ;;
  restart) stop; start ;;
  status)  status ;;
  attach)  exec tmux attach -t "$SESSION" ;;
  _worker) run_worker ;;   # 내부용: watchfiles 가 워커를 띄울 때 쓴다
  *) echo "사용법: ./dev.sh [start|stop|restart|status|attach]"; exit 1 ;;
esac
