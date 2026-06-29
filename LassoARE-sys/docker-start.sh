#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${ROOT_DIR}/scripts/common.sh"

PROFILE="auto"

usage() {
  cat <<'EOF'
Usage: ./docker-start.sh [options]

Options:
  --profile auto|cpu|cuda  Select image profile (default: auto).
  -h, --help               Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      [[ $# -ge 2 ]] || die "--profile requires a value."
      PROFILE="$2"
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

[[ "$PROFILE" =~ ^(auto|cpu|cuda)$ ]] || die "Profile must be auto, cpu, or cuda."
command -v docker >/dev/null 2>&1 || die "Docker is not installed."
docker compose version >/dev/null 2>&1 || die "Docker Compose v2 is required."

if [[ "$PROFILE" == "auto" ]]; then
  PROFILE="$(detect_profile)"
  if [[ "$PROFILE" == "cuda" ]] \
    && ! docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -qi nvidia; then
    log "NVIDIA Container Toolkit was not detected; using the CPU image."
    PROFILE="cpu"
  fi
fi

compose_files=(-f "${ROOT_DIR}/compose.yaml")
if [[ "$PROFILE" == "cuda" ]]; then
  [[ "$(detect_profile)" == "cuda" ]] \
    || die "CUDA Docker requires an NVIDIA GPU and driver >= ${CUDA_MIN_DRIVER}."
  docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -qi nvidia \
    || die "CUDA Docker requires NVIDIA Container Toolkit."
  compose_files+=(-f "${ROOT_DIR}/compose.cuda.yaml")
fi

log "Building and starting the ${PROFILE} Docker profile."
docker compose "${compose_files[@]}" up --build --detach
log "LassoARE is starting at http://${LASSOARE_BIND_HOST:-127.0.0.1}:${LASSOARE_PORT:-15114}"
