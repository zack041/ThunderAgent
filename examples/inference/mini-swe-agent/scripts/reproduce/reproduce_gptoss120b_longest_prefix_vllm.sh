#!/usr/bin/env bash
set -euo pipefail

# Reproduce the GPT-OSS-120B mini-swe-agent SWE-bench run using vanilla vLLM
# with automatic prefix caching and LongestPrefixHitScheduler:
#
#   mini-swe-agent -> vLLM (longest-prefix-hit-first)
#
# GPT-OSS-120B uses its native MXFP4 weight quantization and the default KV
# cache dtype. Run this script from the ThunderAgent repository root.

# =========================
# User-facing configuration
# =========================
# Required: fill this path before running.
HF_HOME=""

# Model
MODEL_REPO="openai/gpt-oss-120b"
MODEL_DIR="${HF_HOME}/models/gpt-oss-120b"

# Runtime service: one H200, no tensor parallelism.
VLLM_PORT="8100"
VLLM_TP_SIZE="1"
HEALTH_TIMEOUT_S="1800"
METRICS_INTERVAL_S="5"

# SWE-bench run
SWEBENCH_SUBSET="lite"
SWEBENCH_SPLIT="test"
SWEBENCH_WORKERS="128"
SWEBENCH_OUTPUT="./swebench_output_gptoss120b_longest_prefix_vllm"

# Keep artifacts separate from FCFS vLLM and ThunderAgent runs.
LOG_DIR="./logs/gptoss120b_longest_prefix_vllm"
METRICS_DIR="${LOG_DIR}/metrics"

# =========================
# Internal wiring
# =========================
VLLM_LOG="${LOG_DIR}/vllm_gptoss120b_longest_prefix_${VLLM_PORT}.log"
BASE_CONFIG="./examples/inference/mini-swe-agent/src/minisweagent/config/extra/swebench.yaml"
VLLM_CONFIG="${LOG_DIR}/swebench_vllm.yaml"

VLLM_HEALTH_URL="http://localhost:${VLLM_PORT}/health"
VLLM_METRICS_URL="http://localhost:${VLLM_PORT}/metrics"

SCHEDULER_MODULE="longest_prefix_hit_scheduler.LongestPrefixHitScheduler"
SCHEDULER_DIR="../vllm/examples/features/automatic_prefix_caching"
SCHEDULER_FILE="${SCHEDULER_DIR}/longest_prefix_hit_scheduler.py"

VLLM_PID=""
METRICS_PID=""

log_info() {
  echo "[INFO] $*"
}

log_error() {
  echo "[ERROR] $*" >&2
}

die() {
  log_error "$*"
  exit 1
}

require_cmd() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    die "Missing required command: ${cmd}"
  fi
}

wait_for_health() {
  local url="$1"
  local name="$2"
  local timeout_s="${3:-600}"

  local start now out
  start="$(date +%s)"

  while true; do
    out=""
    if out="$(curl -sS -m 2 "${url}" 2>&1)"; then
      return 0
    fi

    now="$(date +%s)"
    if (( now - start >= timeout_s )); then
      log_error "${name} health check failed: timeout after ${timeout_s}s"
      log_error "last curl output: ${out}"
      return 1
    fi

    sleep 2
  done
}

cleanup() {
  set +e
  if [[ -n "${METRICS_PID}" ]] && kill -0 "${METRICS_PID}" 2>/dev/null; then
    kill "${METRICS_PID}" 2>/dev/null
    wait "${METRICS_PID}" 2>/dev/null
  fi
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    kill "${VLLM_PID}" 2>/dev/null
    wait "${VLLM_PID}" 2>/dev/null
  fi
}
trap cleanup EXIT

resolve_scheduler_path() {
  [[ -d "${SCHEDULER_DIR}" ]] || die "Scheduler directory not found: ${SCHEDULER_DIR}"
  [[ -f "${SCHEDULER_FILE}" ]] || die "Scheduler module not found: ${SCHEDULER_FILE}"
  SCHEDULER_DIR="$(cd "${SCHEDULER_DIR}" && pwd)"
  SCHEDULER_FILE="${SCHEDULER_DIR}/longest_prefix_hit_scheduler.py"
  log_info "Scheduler module: ${SCHEDULER_FILE}"
}

prepare_model_dir() {
  log_info "Model repo: ${MODEL_REPO}"
  log_info "Model cache directory: ${MODEL_DIR}"
  mkdir -p "${MODEL_DIR}"
}

download_model() {
  log_info "Downloading model snapshot to: ${MODEL_DIR}"
  HF_HOME="${HF_HOME}" MODEL_REPO="${MODEL_REPO}" MODEL_DIR="${MODEL_DIR}" python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["MODEL_REPO"],
    repo_type="model",
    local_dir=os.environ["MODEL_DIR"],
    local_dir_use_symlinks=False,
    resume_download=True,
)
PY
  log_info "Model download completed."
}

prepare_vllm_config() {
  log_info "Writing direct-vLLM mini-swe-agent config: ${VLLM_CONFIG}"
  BASE_CONFIG="${BASE_CONFIG}" VLLM_CONFIG="${VLLM_CONFIG}" \
    MODEL_REPO="${MODEL_REPO}" VLLM_PORT="${VLLM_PORT}" python - <<'PY'
import os
from pathlib import Path

import yaml

source = Path(os.environ["BASE_CONFIG"])
destination = Path(os.environ["VLLM_CONFIG"])
config = yaml.safe_load(source.read_text())

model = config.setdefault("model", {})
model["model_name"] = os.environ["MODEL_REPO"]
model["base_url"] = f"http://localhost:{os.environ['VLLM_PORT']}/v1"
model["api_key"] = "EMPTY"

destination.write_text(yaml.safe_dump(config, sort_keys=False))
PY
}

start_vllm() {
  log_info "Starting vLLM with longest-prefix-hit scheduling on port ${VLLM_PORT}"
  log_info "vLLM log: ${VLLM_LOG}"
  PYTHONPATH="${SCHEDULER_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
    nohup vllm serve "${MODEL_REPO}" \
      --download-dir "${MODEL_DIR}" \
      --tensor-parallel-size "${VLLM_TP_SIZE}" \
      --enable-prefix-caching \
      --scheduling-policy fcfs \
      --scheduler-cls "${SCHEDULER_MODULE}" \
      --port "${VLLM_PORT}" \
      >"${VLLM_LOG}" 2>&1 &
  VLLM_PID="$!"

  log_info "Waiting for vLLM health: ${VLLM_HEALTH_URL}"
  wait_for_health "${VLLM_HEALTH_URL}" "vLLM" "${HEALTH_TIMEOUT_S}"
}

start_metrics_collector() {
  log_info "Saving vLLM /metrics snapshots every ${METRICS_INTERVAL_S}s to ${METRICS_DIR}"
  (
    while true; do
      timestamp="$(date +%s.%N)"
      curl -fsS -m 5 "${VLLM_METRICS_URL}" \
        >"${METRICS_DIR}/vllm_metrics_${timestamp}.prom" || true
      sleep "${METRICS_INTERVAL_S}"
    done
  ) &
  METRICS_PID="$!"
}

run_swebench() {
  log_info "Running mini-extra swebench against longest-prefix-hit vLLM"
  mini-extra swebench \
    --subset "${SWEBENCH_SUBSET}" \
    --split "${SWEBENCH_SPLIT}" \
    --workers "${SWEBENCH_WORKERS}" \
    --config "${VLLM_CONFIG}" \
    --output "${SWEBENCH_OUTPUT}"
}

main() {
  [[ -n "${HF_HOME}" ]] || die "HF_HOME is required. Edit HF_HOME at the top of this script."
  [[ -d "./examples/inference/mini-swe-agent" ]] || die "Please run from the ThunderAgent repository root."
  [[ -f "${BASE_CONFIG}" ]] || die "Missing mini-swe-agent config: ${BASE_CONFIG}"

  require_cmd vllm
  require_cmd curl
  require_cmd mini-extra
  require_cmd python

  mkdir -p "${LOG_DIR}"
  mkdir -p "${METRICS_DIR}"

  resolve_scheduler_path
  prepare_model_dir
  download_model
  prepare_vllm_config
  start_vllm
  start_metrics_collector
  run_swebench

  log_info "Done."
  log_info "SWE-bench output: ${SWEBENCH_OUTPUT}"
  log_info "vLLM log: ${VLLM_LOG}"
  log_info "vLLM metrics snapshots: ${METRICS_DIR}"
}

main "$@"
