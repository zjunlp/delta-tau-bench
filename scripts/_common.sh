#!/bin/bash
# Shared setup for the entry-point scripts. Sourced, not executed.
#
# Provides:
#   run_step <logfile> <command...>   run a stage, tee to log, honour DRY_RUN
#   ensure_tool_index <version>...    build a version's tool index if missing

set -Eeuo pipefail

# Every default path in the CLI is resolved against the working directory.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PY="${PY:-python}"
DTAU=("$PY" -m dtau_bench)
DRY_RUN="${DRY_RUN:-0}"
RAG_INDEX_BASE="${RAG_INDEX_BASE:-./rag_index_cache}"

# Bound numerical thread pools on CPU hosts; callers may override explicitly.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

run_step() {
  local log="$1"; shift
  if [ "$DRY_RUN" = "1" ]; then
    printf '  [DRY-RUN] %s\n' "$*"
    return 0
  fi
  mkdir -p "$(dirname "$log")"
  # Capture the whole pipeline status explicitly. This keeps command failures
  # visible even when run_step is used in an if/|| context where bash suppresses
  # errexit inside function bodies.
  set +e
  # Resume runs share the same stage log. Append so a boundary restart does not
  # discard the pre-resume evidence needed to audit the experiment.
  "$@" 2>&1 | tee -a "$log"
  local status=$?
  set -e
  return "$status"
}

ensure_tool_index() {
  local v
  local embedding_batch_size="${EMBEDDING_BATCH_SIZE:-8}"
  for v in "$@"; do
    local index_dir="${RAG_INDEX_BASE}/${v}"
    local log="${LOG_DIR}/build_tool_index_${v}.log"
    local validate_log="${LOG_DIR}/validate_tool_index_${v}.log"
    if [ -d "$index_dir" ] && run_step "$validate_log" \
      "${DTAU[@]}" build-tool-index --version "$v" --output-dir "$index_dir" \
      --batch-size "$embedding_batch_size" --validate-only; then
      echo "  [OK] ${v} index validated"
      continue
    fi

    echo "  [BUILD] ${v}"
    run_step "$log" \
      "${DTAU[@]}" build-tool-index --version "$v" --output-dir "$index_dir" \
      --batch-size "$embedding_batch_size" || return $?
    run_step "$validate_log" \
      "${DTAU[@]}" build-tool-index --version "$v" --output-dir "$index_dir" \
      --batch-size "$embedding_batch_size" --validate-only || return $?
  done
}
