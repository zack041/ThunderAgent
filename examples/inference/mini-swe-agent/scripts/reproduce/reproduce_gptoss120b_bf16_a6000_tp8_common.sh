#!/usr/bin/env bash
set -euo pipefail

POLICY="${1:?usage: $0 thunderagent|vanilla_vllm|longest_prefix_vllm}"

case "${POLICY}" in
  thunderagent|vanilla_vllm|longest_prefix_vllm) ;;
  *)
    echo "Unknown policy: ${POLICY}" >&2
    exit 2
    ;;
esac

# Run from the ThunderAgent repository root.
[[ -d "./examples/inference/mini-swe-agent" ]] || {
  echo "Run this script from the ThunderAgent repository root." >&2
  exit 1
}

# RTX A6000 is Ampere, so use the BF16 checkpoint instead of native MXFP4.
HF_HOME="${HF_HOME:-${HOME}/huggingface}"
MODEL_REPO="${MODEL_REPO:-FriendliAI/gpt-oss-120b-BF16}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-openai/gpt-oss-120b}"
MODEL_DIR="${MODEL_DIR:-${HF_HOME}/models/gpt-oss-120b-bf16}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-8}"
VLLM_PORT="${VLLM_PORT:-8100}"
TA_PORT="${TA_PORT:-8000}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
# Preserve the model's full context limit for benchmark comparability. Override
# this to 32768 for an initial low-risk smoke test if startup is tight.
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-131072}"
# Keep the same backend KV headroom for all three policies.
VLLM_WATERMARK="${VLLM_WATERMARK:-0.05}"
LONGEST_PREFIX_DECODE_RESERVE_TOKENS="${LONGEST_PREFIX_DECODE_RESERVE_TOKENS:-1024}"
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-3600}"
METRICS_INTERVAL_S="${METRICS_INTERVAL_S:-5}"

SWEBENCH_SUBSET="${SWEBENCH_SUBSET:-lite}"
SWEBENCH_SPLIT="${SWEBENCH_SPLIT:-test}"
SWEBENCH_WORKERS="${SWEBENCH_WORKERS:-128}"
CONTAINER_TIMEOUT="${CONTAINER_TIMEOUT:-24h}"
SWEBENCH_OUTPUT="${SWEBENCH_OUTPUT:-./swebench_output_gptoss120b_bf16_a6000_tp8_${POLICY}}"
LOG_DIR="${LOG_DIR:-./logs/gptoss120b_bf16_a6000_tp8_${POLICY}}"
METRICS_DIR="${LOG_DIR}/metrics"
ROUTER_DIR="${LOG_DIR}/router_metrics"

BASE_CONFIG="./examples/inference/mini-swe-agent/src/minisweagent/config/extra/swebench.yaml"
VLLM_CONFIG="${LOG_DIR}/swebench_vllm.yaml"
VLLM_LOG="${LOG_DIR}/vllm_gptoss120b_bf16_tp8_${VLLM_PORT}.log"
TA_LOG="${LOG_DIR}/thunderagent_${TA_PORT}.log"

VLLM_HEALTH_URL="http://localhost:${VLLM_PORT}/health"
VLLM_METRICS_URL="http://localhost:${VLLM_PORT}/metrics"
TA_HEALTH_URL="http://localhost:${TA_PORT}/health"

SCHEDULER_MODULE="longest_prefix_hit_scheduler.LongestPrefixHitScheduler"
SCHEDULER_DIR="${SCHEDULER_DIR:-../vllm/examples/features/automatic_prefix_caching}"
SCHEDULER_FILE="${SCHEDULER_DIR}/longest_prefix_hit_scheduler.py"

VLLM_PID=""
TA_PID=""
METRICS_PID=""

log() {
  echo "[INFO] $*"
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

wait_for_health() {
  local url="$1"
  local name="$2"
  local started now last_output=""
  started="$(date +%s)"
  while true; do
    if last_output="$(curl -sS -m 3 "${url}" 2>&1)"; then
      return 0
    fi
    now="$(date +%s)"
    if (( now - started >= HEALTH_TIMEOUT_S )); then
      die "${name} health check timed out after ${HEALTH_TIMEOUT_S}s: ${last_output}"
    fi
    sleep 3
  done
}

cleanup() {
  set +e
  if [[ -n "${METRICS_PID}" ]] && kill -0 "${METRICS_PID}" 2>/dev/null; then
    kill "${METRICS_PID}" 2>/dev/null
    wait "${METRICS_PID}" 2>/dev/null
  fi
  if [[ -n "${TA_PID}" ]] && kill -0 "${TA_PID}" 2>/dev/null; then
    kill "${TA_PID}" 2>/dev/null
    wait "${TA_PID}" 2>/dev/null
  fi
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    kill "${VLLM_PID}" 2>/dev/null
    wait "${VLLM_PID}" 2>/dev/null
  fi
}
trap cleanup EXIT INT TERM

check_gpus() {
  local gpu_count
  gpu_count="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d " ")"
  [[ "${gpu_count}" -ge 8 ]] || die "Expected at least 8 GPUs, found ${gpu_count}"
  log "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
}

download_model() {
  mkdir -p "${MODEL_DIR}"
  log "Downloading ${MODEL_REPO} to ${MODEL_DIR}"
  HF_HOME="${HF_HOME}" MODEL_REPO="${MODEL_REPO}" MODEL_DIR="${MODEL_DIR}" \
    python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["MODEL_REPO"],
    local_dir=os.environ["MODEL_DIR"],
    repo_type="model",
)
PY
}

prepare_run_config() {
  local api_port="${VLLM_PORT}"
  if [[ "${POLICY}" == "thunderagent" ]]; then
    api_port="${TA_PORT}"
  fi

  BASE_CONFIG="${BASE_CONFIG}" VLLM_CONFIG="${VLLM_CONFIG}" \
    SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" API_PORT="${api_port}" \
    CONTAINER_TIMEOUT="${CONTAINER_TIMEOUT}" \
    python - <<'PY'
import os
from pathlib import Path
import yaml

source = Path(os.environ["BASE_CONFIG"])
destination = Path(os.environ["VLLM_CONFIG"])
config = yaml.safe_load(source.read_text())
model = config.setdefault("model", {})
model["model_name"] = os.environ["SERVED_MODEL_NAME"]
model["base_url"] = f"http://localhost:{os.environ['API_PORT']}/v1"
model["api_key"] = "EMPTY"
environment = config.setdefault("environment", {})
environment["container_timeout"] = os.environ["CONTAINER_TIMEOUT"]
destination.write_text(yaml.safe_dump(config, sort_keys=False))

written = yaml.safe_load(destination.read_text())
assert written["environment"]["container_timeout"] == os.environ["CONTAINER_TIMEOUT"]
print(
    "[INFO] Run config:",
    destination,
    "base_url=" + written["model"]["base_url"],
    "container_timeout=" + written["environment"]["container_timeout"],
)
PY
}

start_vllm() {
  local scheduler_env=()
  local scheduler_args=()

  if [[ "${POLICY}" == "longest_prefix_vllm" ]]; then
    [[ -f "${SCHEDULER_FILE}" ]] || die "Scheduler not found: ${SCHEDULER_FILE}"
    SCHEDULER_DIR="$(cd "${SCHEDULER_DIR}" && pwd)"
    scheduler_env=(
      "PYTHONPATH=${SCHEDULER_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
      "VLLM_LONGEST_PREFIX_DECODE_RESERVE_TOKENS=${LONGEST_PREFIX_DECODE_RESERVE_TOKENS}"
    )
    scheduler_args=(
      --scheduler-cls "${SCHEDULER_MODULE}"
    )
  fi

  log "Starting ${POLICY} backend with TP=${VLLM_TP_SIZE}, BF16, max_model_len=${VLLM_MAX_MODEL_LEN}"
  env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${scheduler_env[@]}" \
    nohup vllm serve "${MODEL_DIR}" \
      --served-model-name "${SERVED_MODEL_NAME}" \
      --tensor-parallel-size "${VLLM_TP_SIZE}" \
      --dtype bfloat16 \
      --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
      --max-model-len "${VLLM_MAX_MODEL_LEN}" \
      --enable-prefix-caching \
      --scheduling-policy fcfs \
      --watermark "${VLLM_WATERMARK}" \
      --port "${VLLM_PORT}" \
      "${scheduler_args[@]}" \
      >"${VLLM_LOG}" 2>&1 &
  VLLM_PID="$!"

  wait_for_health "${VLLM_HEALTH_URL}" "vLLM"
  log "vLLM is healthy"
}

smoke_test_vllm() {
  log "Running one-request BF16/Ampere smoke test"
  curl -fsS -m 300 "${VLLM_HEALTH_URL}" >/dev/null
  SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" VLLM_PORT="${VLLM_PORT}" python - <<'PY'
import json
import os
import urllib.request

payload = json.dumps({
    "model": os.environ["SERVED_MODEL_NAME"],
    "messages": [{"role": "user", "content": "Return exactly: READY"}],
    "max_tokens": 16,
    "temperature": 0,
}).encode()
request = urllib.request.Request(
    f"http://localhost:{os.environ['VLLM_PORT']}/v1/chat/completions",
    data=payload,
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=300) as response:
    result = json.load(response)
if not result.get("choices"):
    raise SystemExit(f"Smoke test returned no choices: {result}")
print("[INFO] Smoke test passed")
PY
}

start_thunderagent() {
  log "Starting ThunderAgent on port ${TA_PORT}"
  nohup python -m ThunderAgent \
    --backends "http://localhost:${VLLM_PORT}" \
    --port "${TA_PORT}" \
    --metrics \
    --profile \
    --profile-dir "${ROUTER_DIR}" \
    >"${TA_LOG}" 2>&1 &
  TA_PID="$!"
  wait_for_health "${TA_HEALTH_URL}" "ThunderAgent"
}

start_metrics_collector() {
  (
    while true; do
      local timestamp
      timestamp="$(date +%s.%N)"
      curl -fsS -m 5 "${VLLM_METRICS_URL}" \
        >"${METRICS_DIR}/vllm_metrics_${timestamp}.prom" || true
      sleep "${METRICS_INTERVAL_S}"
    done
  ) &
  METRICS_PID="$!"
}

run_swebench() {
  mini-extra swebench \
    --subset "${SWEBENCH_SUBSET}" \
    --split "${SWEBENCH_SPLIT}" \
    --workers "${SWEBENCH_WORKERS}" \
    --config "${VLLM_CONFIG}" \
    --output "${SWEBENCH_OUTPUT}"
}

main() {
  require_cmd nvidia-smi
  require_cmd vllm
  require_cmd curl
  require_cmd python
  require_cmd mini-extra
  [[ -f "${BASE_CONFIG}" ]] || die "Missing config: ${BASE_CONFIG}"

  export HF_HOME CUDA_VISIBLE_DEVICES
  export MSWEA_SWEBENCH_IMAGE_REGISTRY="${MSWEA_SWEBENCH_IMAGE_REGISTRY:-epoch}"
  mkdir -p "${LOG_DIR}" "${METRICS_DIR}" "${ROUTER_DIR}" "${SWEBENCH_OUTPUT}"

  check_gpus
  download_model
  prepare_run_config
  start_vllm
  smoke_test_vllm
  if [[ "${POLICY}" == "thunderagent" ]]; then
    start_thunderagent
  fi
  start_metrics_collector
  run_swebench

  log "Completed ${POLICY}"
  log "Output: ${SWEBENCH_OUTPUT}"
  log "Logs: ${LOG_DIR}"
}

main "$@"
