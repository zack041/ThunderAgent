#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ROOT="${RUN_ROOT:-${HOME}/experiments/gptoss120b_bf16_a6000_tp8}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-1}"
export MSWEA_SWEBENCH_IMAGE_REGISTRY="${MSWEA_SWEBENCH_IMAGE_REGISTRY:-epoch}"
mkdir -p "${RUN_ROOT}"

if [[ ! -f "${RUN_ROOT}/PREFETCH_COMPLETE" ]]; then
  echo "[PREFETCH] Caching all SWE-bench Lite Docker images"
  python -m minisweagent.run.extra.prefetch_images \
    --subset lite \
    --split test \
    --count 300 \
    --workers 8 \
    --timeout-s 0 \
    >"${RUN_ROOT}/prefetch.log" 2>&1
  touch "${RUN_ROOT}/PREFETCH_COMPLETE"
fi

for policy in thunderagent vanilla_vllm longest_prefix_vllm; do
  if [[ -f "${RUN_ROOT}/${policy}/SUCCESS" ]]; then
    echo "[SKIP] ${policy} already complete"
    continue
  fi

  attempt=1
  until bash "${REPO_ROOT}/scripts/experiments/run_policy_experiment_a6000_tp8.sh" \
    "${policy}" "${RUN_ROOT}"; do
    if (( attempt >= MAX_ATTEMPTS )); then
      echo "[FAIL] ${policy} failed after ${attempt} attempt(s)" >&2
      exit 1
    fi
    attempt=$((attempt + 1))
    echo "[RETRY] ${policy}: attempt ${attempt}/${MAX_ATTEMPTS}"
    sleep 30
  done
  echo "[DONE] ${policy}"
done

echo "ALL_EXPERIMENTS_COMPLETE"
echo "Raw experiment data: ${RUN_ROOT}"
