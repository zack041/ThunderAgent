#!/usr/bin/env bash
set -euo pipefail

POLICY="${1:?usage: run_policy_experiment_a6000_tp8.sh POLICY RUN_ROOT}"
RUN_ROOT="${2:?usage: run_policy_experiment_a6000_tp8.sh POLICY RUN_ROOT}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
POLICY_DIR="${RUN_ROOT}/${POLICY}"
OUTPUT_DIR="${POLICY_DIR}/swebench_output"
LOG_DIR="${POLICY_DIR}/logs"
RUNTIME_DIR="${POLICY_DIR}/runtime_metrics"
ATTEMPT_DIR="${POLICY_DIR}/attempts/$(date -u +%Y%m%dT%H%M%SZ)"

case "${POLICY}" in
  thunderagent)
    SCRIPT="${REPO_ROOT}/examples/inference/mini-swe-agent/scripts/reproduce/reproduce_gptoss120b_bf16_a6000_tp8.sh"
    ;;
  vanilla_vllm)
    SCRIPT="${REPO_ROOT}/examples/inference/mini-swe-agent/scripts/reproduce/reproduce_gptoss120b_bf16_a6000_tp8_vllm.sh"
    ;;
  longest_prefix_vllm)
    SCRIPT="${REPO_ROOT}/examples/inference/mini-swe-agent/scripts/reproduce/reproduce_gptoss120b_bf16_a6000_tp8_longest_prefix_vllm.sh"
    ;;
  *)
    echo "Unknown policy: ${POLICY}" >&2
    exit 2
    ;;
esac

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${RUNTIME_DIR}" "${ATTEMPT_DIR}"

start_ts="$(date +%s.%N)"
python "${REPO_ROOT}/scripts/experiments/collect_live_metrics.py" \
  --output-dir "${RUNTIME_DIR}" --interval 5 &
collector_pid="$!"

cleanup() {
  set +e
  kill "${collector_pid}" 2>/dev/null
  wait "${collector_pid}" 2>/dev/null
}
trap cleanup EXIT

set +e
SWEBENCH_WORKERS="${SWEBENCH_WORKERS:-128}" \
SWEBENCH_OUTPUT="${OUTPUT_DIR}" \
LOG_DIR="${LOG_DIR}" \
bash "${SCRIPT}" >"${ATTEMPT_DIR}/console.log" 2>&1
exit_code="$?"
set -e

end_ts="$(date +%s.%N)"
cleanup
trap - EXIT

pred_count="$(python - "${OUTPUT_DIR}/preds.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
print(len(json.loads(path.read_text())) if path.exists() else 0)
PY
)"
timing_count="$(find "${OUTPUT_DIR}" -name "*.timings.json" | wc -l | tr -d " ")"

python - "${ATTEMPT_DIR}/metadata.json" "${POLICY}" "${start_ts}" "${end_ts}" \
  "${exit_code}" "${pred_count}" "${timing_count}" <<'PY'
import json
import sys

path, policy, start, end, code, preds, timings = sys.argv[1:]
payload = {
    "policy": policy,
    "hardware": "8x RTX A6000",
    "model": "FriendliAI/gpt-oss-120b-BF16",
    "tensor_parallel_size": 8,
    "start_ts": float(start),
    "end_ts": float(end),
    "wall_time_s": float(end) - float(start),
    "exit_code": int(code),
    "prediction_count": int(preds),
    "timing_file_count": int(timings),
}
open(path, "w").write(json.dumps(payload, indent=2) + "\n")
PY
cp "${ATTEMPT_DIR}/metadata.json" "${POLICY_DIR}/latest_metadata.json"

if [[ "${exit_code}" -eq 0 && "${pred_count}" -eq 300 && "${timing_count}" -eq 300 ]]; then
  touch "${POLICY_DIR}/SUCCESS"
  exit 0
fi

echo "${POLICY} incomplete: exit=${exit_code} preds=${pred_count} timings=${timing_count}" >&2
exit 1
