#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 QUEUE_SCRIPT" >&2
  exit 2
fi

QUEUE_SCRIPT="$1"
WAIT_PIDS="${WAIT_PIDS:-}"
POLL_SECONDS="${POLL_SECONDS:-300}"

if [[ ! -f "${QUEUE_SCRIPT}" ]]; then
  echo "queue script not found: ${QUEUE_SCRIPT}" >&2
  exit 2
fi

cd "$(dirname "$0")/.."

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2
}

is_pid_alive() {
  local pid="$1"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

if [[ -n "${WAIT_PIDS}" ]]; then
  log "waiting for PIDs: ${WAIT_PIDS}"
  while true; do
    alive=()
    for pid in ${WAIT_PIDS}; do
      if is_pid_alive "${pid}"; then
        alive+=("${pid}")
      fi
    done
    if [[ ${#alive[@]} -eq 0 ]]; then
      log "all waited PIDs have exited"
      break
    fi
    log "still running: ${alive[*]}"
    sleep "${POLL_SECONDS}"
  done
fi

log "starting queue script: ${QUEUE_SCRIPT}"
bash "${QUEUE_SCRIPT}"
log "queue script finished: ${QUEUE_SCRIPT}"
