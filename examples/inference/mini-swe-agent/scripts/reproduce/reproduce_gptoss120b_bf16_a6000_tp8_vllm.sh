#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/reproduce_gptoss120b_bf16_a6000_tp8_common.sh" vanilla_vllm
