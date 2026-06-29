#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${ROOT_DIR}/scripts/common.sh"

HOST="${LASSOARE_BIND_HOST:-127.0.0.1}"
PORT="${LASSOARE_PORT:-15114}"
DATA_DIR="${LASSOARE_DATA_DIR:-$DEFAULT_DATA_DIR}"

usage() {
  cat <<'EOF'
Usage: ./start.sh [options]

Options:
  --host ADDRESS   Bind address (default: 127.0.0.1).
  --port PORT      Service port (default: 15114).
  --data-dir PATH  Persistent data directory.
  -h, --help       Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      [[ $# -ge 2 ]] || die "--host requires a value."
      HOST="$2"
      shift 2
      ;;
    --port)
      [[ $# -ge 2 ]] || die "--port requires a value."
      PORT="$2"
      shift 2
      ;;
    --data-dir)
      [[ $# -ge 2 ]] || die "--data-dir requires a value."
      DATA_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown option: $1"
      ;;
  esac
done

CONFIG_PATH="${DATA_DIR}/runtime.env"
[[ -f "$CONFIG_PATH" ]] || die "Runtime config not found. Run ./install.sh first."
source "$CONFIG_PATH"
export LASSOARE_DATA_DIR="$DATA_DIR"

cd "$ROOT_DIR"
exec "$LASSOARE_MAIN_PYTHON" -m uvicorn app.main:app --host "$HOST" --port "$PORT"
