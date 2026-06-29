#!/usr/bin/env bash
set -euo pipefail

export LASSOARE_PROFILE="${LASSOARE_PROFILE:-cpu}"
export LASSOARE_MAIN_PYTHON="/opt/micromamba/envs/lassoare_main/bin/python"
export LASSOARE_DATA_DIR="${LASSOARE_DATA_DIR:-/data}"
export LASSOARE_SAMPLE_DIR="${LASSOARE_SAMPLE_DIR:-${LASSOARE_DATA_DIR}/samples}"

if [[ "$LASSOARE_PROFILE" == "cuda" ]]; then
  export LASSOARE_RSC_PYTHON="/opt/micromamba/envs/lassoare_rsc/bin/python"
else
  unset LASSOARE_RSC_PYTHON || true
fi

mkdir -p "$LASSOARE_DATA_DIR" "$LASSOARE_SAMPLE_DIR"
cd /opt/lassoare

exec "$LASSOARE_MAIN_PYTHON" -m uvicorn app.main:app \
  --host 0.0.0.0 \
  --port "${LASSOARE_PORT:-15114}"
