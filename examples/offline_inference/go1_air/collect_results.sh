#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# GO-1-Air offline inference result collection script.
#
# Collects end2end.py output artifacts, environment summary, GPU snapshots,
# and timing information into a timestamped result directory.
#
# Defaults to CPU stub mode because CUDA real-checkpoint is currently blocked
# on this machine (driver 12.9 < torch CUDA 13.0 requirement).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT_DIR/../../.." && pwd)"

GO1_AIR_OUTPUT_ROOT="${GO1_AIR_OUTPUT_ROOT:-/root/gpufree-data/vllm_omni_onboarding/results/go1_air}"
GO1_AIR_DEVICE="${GO1_AIR_DEVICE:-cpu}"
GO1_AIR_DTYPE="${GO1_AIR_DTYPE:-float32}"
GO1_AIR_NUM_SAMPLES="${GO1_AIR_NUM_SAMPLES:-1}"
GO1_AIR_SEED="${GO1_AIR_SEED:-1234}"
GO1_AIR_MODEL_DIR="${GO1_AIR_MODEL_DIR:-}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_ROOT="${GO1_AIR_OUTPUT_ROOT}/run_${TIMESTAMP}"

mkdir -p "$RESULT_ROOT"

# ----- helpers -----

write_env_summary() {
  cat >"$RESULT_ROOT/env_summary.txt" <<EOF
date=$(date)
hostname=$(hostname 2>/dev/null || echo "unknown")
pwd=$(pwd)
repo_root=$REPO_ROOT
git_commit=$(cd "$REPO_ROOT" && git rev-parse --short HEAD 2>/dev/null || echo "unknown")
python=$(command -v python || true)
python_version=$(python --version 2>&1 || true)
pip_version=$(python -m pip --version 2>&1 || true)
device=$GO1_AIR_DEVICE
dtype=$GO1_AIR_DTYPE
num_samples=$GO1_AIR_NUM_SAMPLES
seed=$GO1_AIR_SEED
model_dir=${GO1_AIR_MODEL_DIR:-"(not set — stub mode)"}
EOF
  echo "" >>"$RESULT_ROOT/env_summary.txt"
  echo "--- df -h / ---" >>"$RESULT_ROOT/env_summary.txt"
  df -h / >>"$RESULT_ROOT/env_summary.txt" 2>&1 || true
  echo "" >>"$RESULT_ROOT/env_summary.txt"
  echo "--- df -h /root/gpufree-data ---" >>"$RESULT_ROOT/env_summary.txt"
  df -h /root/gpufree-data >>"$RESULT_ROOT/env_summary.txt" 2>&1 || true
}

capture_gpu_snapshot() {
  local output_file="$1"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi >"$output_file" 2>&1 || echo "nvidia-smi failed with exit code $?" >"$output_file"
  else
    echo "nvidia-smi not found" >"$output_file"
  fi
}

run_with_artifacts() {
  local name="$1"
  shift

  local log_file="$RESULT_ROOT/${name}.log"
  local time_file="$RESULT_ROOT/${name}_time.txt"
  local status_file="$RESULT_ROOT/${name}_status.txt"

  local exit_code=0
  local start_ts=0
  local end_ts=0
  start_ts="$(date +%s)"
  set +e
  if command -v /usr/bin/time >/dev/null 2>&1; then
    /usr/bin/time -v -o "$time_file" "$@" >"$log_file" 2>&1
    exit_code=$?
    if [[ ! -s "$time_file" ]]; then
      end_ts="$(date +%s)"
      {
        echo "timing_mode=usr_bin_time_empty_fallback"
        echo "start_ts=$start_ts"
        echo "end_ts=$end_ts"
        echo "elapsed_seconds=$((end_ts - start_ts))"
      } >"$time_file"
    fi
  else
    "$@" >"$log_file" 2>&1
    exit_code=$?
    end_ts="$(date +%s)"
    {
      echo "timing_mode=shell_date"
      echo "start_ts=$start_ts"
      echo "end_ts=$end_ts"
      echo "elapsed_seconds=$((end_ts - start_ts))"
    } >"$time_file"
  fi
  set -e

  echo "exit_code=$exit_code" >"$status_file"
  if [[ $exit_code -ne 0 ]]; then
    echo "[error] command failed for $name, see $log_file"
    return $exit_code
  fi
}

write_manifest() {
  cat >"$RESULT_ROOT/README.txt" <<EOF
GO-1-Air collected results — $(date)

Key files:
- env_summary.txt: environment and path summary
- nvidia_smi_before.txt / nvidia_smi_after.txt: GPU snapshots
- end2end.log: stdout+stderr from end2end.py
- time_end2end.txt: /usr/bin/time -v output (or shell-date fallback)
- outputs/: directory containing end2end.py artifacts:
  - actions.npy
  - metadata.json
  - env.json
  - gpu.json
  - comparison.json (if --upstream-action-path was provided)
  - forward_latency.json (if --benchmark-forward was used)
EOF
}

# ----- main -----

echo "[collect] result root: $RESULT_ROOT"

write_env_summary
write_manifest
capture_gpu_snapshot "$RESULT_ROOT/nvidia_smi_before.txt"

END2END_ARGS=(
  --device "$GO1_AIR_DEVICE"
  --dtype "$GO1_AIR_DTYPE"
  --num-samples "$GO1_AIR_NUM_SAMPLES"
  --seed "$GO1_AIR_SEED"
  --output-dir "$RESULT_ROOT/outputs"
)

if [[ -n "${GO1_AIR_MODEL_DIR:-}" ]]; then
  END2END_ARGS+=(--model-dir "$GO1_AIR_MODEL_DIR")
fi

if [[ -n "${GO1_AIR_UPSTREAM_ACTION_PATH:-}" ]]; then
  END2END_ARGS+=(--upstream-action-path "$GO1_AIR_UPSTREAM_ACTION_PATH")
fi

run_with_artifacts \
  "end2end" \
  python "$ROOT_DIR/end2end.py" "${END2END_ARGS[@]}"

capture_gpu_snapshot "$RESULT_ROOT/nvidia_smi_after.txt"

echo "[collect] results written to: $RESULT_ROOT"

# List key outputs for quick verification.
echo "[collect] outputs:"
find "$RESULT_ROOT" -maxdepth 2 -type f -ls 2>/dev/null || true
