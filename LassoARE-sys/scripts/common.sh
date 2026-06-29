#!/usr/bin/env bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MICROMAMBA_VERSION="2.6.2-1"
CUDA_MIN_DRIVER="570.26"
DEFAULT_DATA_DIR="${HOME}/.local/share/lassoare"

log() {
  printf '[lassoare] %s\n' "$*"
}

die() {
  printf '[lassoare] ERROR: %s\n' "$*" >&2
  exit 1
}

version_at_least() {
  local current="$1"
  local minimum="$2"
  [[ "$(printf '%s\n%s\n' "$minimum" "$current" | sort -V | head -n1)" == "$minimum" ]]
}

detect_profile() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf 'cpu\n'
    return
  fi
  local driver
  driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1)"
  if [[ -n "$driver" ]] && version_at_least "$driver" "$CUDA_MIN_DRIVER"; then
    printf 'cuda\n'
  else
    printf 'cpu\n'
  fi
}

verify_bare_sha256() {
  local artifact="$1"
  local checksum_file="$2"
  local expected actual
  expected="$(tr -d '[:space:]' < "$checksum_file")"
  actual="$(sha256sum "$artifact" | awk '{print $1}')"
  [[ "$expected" =~ ^[0-9a-fA-F]{64}$ ]] || die "Invalid SHA-256 file: ${checksum_file}"
  [[ "$actual" == "$expected" ]] || die "SHA-256 verification failed: ${artifact}"
}

bootstrap_micromamba() {
  local install_dir="${HOME}/.local/share/lassoare/bin"
  local binary="${install_dir}/micromamba"
  local base_url="https://github.com/mamba-org/micromamba-releases/releases/download/${MICROMAMBA_VERSION}"
  local temp_dir
  temp_dir="$(mktemp -d)"

  mkdir -p "$install_dir"
  log "Downloading micromamba ${MICROMAMBA_VERSION}." >&2
  curl -fsSL "${base_url}/micromamba-linux-64" -o "${temp_dir}/micromamba-linux-64"
  curl -fsSL "${base_url}/micromamba-linux-64.sha256" -o "${temp_dir}/micromamba-linux-64.sha256"
  verify_bare_sha256 \
    "${temp_dir}/micromamba-linux-64" \
    "${temp_dir}/micromamba-linux-64.sha256"
  install -m 0755 "${temp_dir}/micromamba-linux-64" "$binary"
  rm -rf "$temp_dir"
  printf '%s\n' "$binary"
}

select_environment_manager() {
  if command -v micromamba >/dev/null 2>&1; then
    ENV_MANAGER="$(command -v micromamba)"
  elif command -v mamba >/dev/null 2>&1; then
    ENV_MANAGER="$(command -v mamba)"
  elif command -v conda >/dev/null 2>&1; then
    ENV_MANAGER="$(command -v conda)"
  else
    command -v curl >/dev/null 2>&1 || die "curl is required to install micromamba."
    ENV_MANAGER="$(bootstrap_micromamba)"
  fi

  if [[ "$(basename "$ENV_MANAGER")" == "micromamba" ]]; then
    export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${HOME}/.local/share/lassoare/micromamba}"
  fi
  log "Using environment manager: ${ENV_MANAGER}"
}

environment_exists() {
  local name="$1"
  "$ENV_MANAGER" run -n "$name" python -V >/dev/null 2>&1
}

apply_environment() {
  local name="$1"
  local manifest="$2"
  if environment_exists "$name"; then
    log "Updating ${name}."
    "$ENV_MANAGER" env update -n "$name" -f "$manifest" -y
  else
    log "Creating ${name}."
    "$ENV_MANAGER" env create -n "$name" -f "$manifest" -y
  fi
}

environment_python() {
  local name="$1"
  "$ENV_MANAGER" run -n "$name" python -c 'import sys; print(sys.executable)'
}
