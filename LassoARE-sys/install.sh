#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${ROOT_DIR}/scripts/common.sh"

PROFILE="auto"
START_AFTER_INSTALL=1
DATA_DIR="${LASSOARE_DATA_DIR:-$DEFAULT_DATA_DIR}"
SMALL_SAMPLE_URL="${LASSOARE_SMALL_SAMPLE_URL:-}"

usage() {
  cat <<'EOF'
Usage: ./install.sh [options]

Options:
  --profile auto|cpu|cuda  Select runtime profile (default: auto).
  --data-dir PATH          Persistent data directory.
  --no-start               Install and validate without starting the service.
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
    --data-dir)
      [[ $# -ge 2 ]] || die "--data-dir requires a value."
      DATA_DIR="$2"
      shift 2
      ;;
    --no-start)
      START_AFTER_INSTALL=0
      shift
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
if [[ "$PROFILE" == "auto" ]]; then
  PROFILE="$(detect_profile)"
  log "Auto-detected profile: ${PROFILE}."
fi
if [[ "$PROFILE" == "cuda" ]]; then
  detected_profile="$(detect_profile)"
  [[ "$detected_profile" == "cuda" ]] || die \
    "CUDA profile requires an NVIDIA GPU and driver >= ${CUDA_MIN_DRIVER}."
fi

select_environment_manager
mkdir -p "$DATA_DIR" "${DATA_DIR}/samples"
CONDARC_PATH="${DATA_DIR}/installer.condarc"
printf 'channels: []\nchannel_priority: strict\n' > "$CONDARC_PATH"
export CONDARC="$CONDARC_PATH"

apply_environment \
  "lassoare_main" \
  "${ROOT_DIR}/environments/lassoare-main-${PROFILE}.yml"
MAIN_PYTHON="$(environment_python lassoare_main)"
TORCH_PROFILE="$([[ "$PROFILE" == "cuda" ]] && printf 'cu128' || printf 'cpu')"
log "Installing PyTorch profile packages: torch-${TORCH_PROFILE}.txt"
"$MAIN_PYTHON" -m pip install -r "${ROOT_DIR}/environments/torch-${TORCH_PROFILE}.txt"
RSC_PYTHON=""
if [[ "$PROFILE" == "cuda" ]]; then
  apply_environment "lassoare_rsc" "${ROOT_DIR}/environments/lassoare-rsc.yml"
  RSC_PYTHON="$(environment_python lassoare_rsc)"
fi

CONFIG_PATH="${DATA_DIR}/runtime.env"
{
  printf 'export LASSOARE_PROFILE=%q\n' "$PROFILE"
  printf 'export LASSOARE_MAIN_PYTHON=%q\n' "$MAIN_PYTHON"
  printf 'export LASSOARE_RSC_PYTHON=%q\n' "$RSC_PYTHON"
  printf 'export LASSOARE_DATA_DIR=%q\n' "$DATA_DIR"
  printf 'export LASSOARE_SAMPLE_DIR=%q\n' "${DATA_DIR}/samples"
  printf 'export LASSOARE_SMALL_SAMPLE_URL=%q\n' "$SMALL_SAMPLE_URL"
} > "$CONFIG_PATH"

log "Validating lassoare_main."
(
  cd "$ROOT_DIR"
  "$MAIN_PYTHON" -c \
    'import anndata, fastapi, scanpy, sklearn, torch; import backend.pairpotlpa; print(torch.__version__)'
)
if [[ "$PROFILE" == "cuda" ]]; then
  log "Validating lassoare_rsc."
  (
    cd "$ROOT_DIR"
    "$RSC_PYTHON" -c \
      'import cupy, cuml, cugraph, rapids_singlecell; print(cupy.cuda.runtime.runtimeGetVersion())'
  )
fi

if [[ -f "${ROOT_DIR}/sc_sampled.h5ad" ]]; then
  cp -f "${ROOT_DIR}/sc_sampled.h5ad" "${DATA_DIR}/samples/sc_sampled.h5ad"
fi

source "$CONFIG_PATH"
(
  cd "$ROOT_DIR"
  "$MAIN_PYTHON" -m app.samples --prepare-configured
)

log "Installation completed. Runtime config: ${CONFIG_PATH}"
if [[ "$START_AFTER_INSTALL" -eq 1 ]]; then
  exec "${ROOT_DIR}/start.sh" --data-dir "$DATA_DIR"
fi
