#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# GO1_AIR_MODEL_DIR is optional; smoke test runs in stub mode without it.
if [[ -n "${GO1_AIR_MODEL_DIR:-}" ]]; then
  export GO1_AIR_MODEL_DIR
fi

# --end2end  → run the full end2end harness instead of the minimal smoke test.
# --collect  → run collect_results.sh for timestamped artifact archiving.
case "${1:-}" in
  --end2end)
    shift
    python "$ROOT_DIR/end2end.py" "$@"
    ;;
  --collect)
    shift
    bash "$ROOT_DIR/collect_results.sh" "$@"
    ;;
  *)
    python "$ROOT_DIR/smoke.py" "$@"
    ;;
esac
