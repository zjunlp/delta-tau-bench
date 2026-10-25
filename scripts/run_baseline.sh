#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
#  Baseline: iterative tool retrieval, no skill library
# ══════════════════════════════════════════════════════════════════════════════
#
#  Runs test-split rollouts for one or more catalog versions, evaluates each
#  version, then aggregates across versions.
#
#  Usage:
#    bash scripts/run_baseline.sh
#    VERSIONS="v1 v2"                    bash scripts/run_baseline.sh
#    AGENT_MODEL=my-model EXP_NAME=my    bash scripts/run_baseline.sh
#    DRY_RUN=1                           bash scripts/run_baseline.sh
#
#  Add the model to configs/models.yaml first.
# ══════════════════════════════════════════════════════════════════════════════

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

read -r -a VERSIONS <<< "${VERSIONS:-v1 v2 v3 v4 v5 v6}"

AGENT_MODEL="${AGENT_MODEL:-deepseek-v4-pro}"
USER_MODEL="${USER_MODEL:-deepseek-flash}"

# The experiment name carries the model so that two models never write into the
# same directory.
EXP_NAME="${EXP_NAME:-baseline_$(echo "$AGENT_MODEL" | tr '[:upper:]./' '[:lower:]__')}"

STRATEGY="${STRATEGY:-all_tools_iterative_rag}"
MAX_WORKERS="${MAX_WORKERS:-1}"
QUERY_WORKERS="${QUERY_WORKERS:-4}"
EVAL_WORKERS="${EVAL_WORKERS:-2}"

RUN_NAME_PATTERN="${EXP_NAME}_{version}_test"
LOG_DIR="logs/${EXP_NAME}"
if [ "$DRY_RUN" != "1" ]; then
  mkdir -p "$LOG_DIR"
fi

echo "═══════════════════════════════════════════════════════════"
echo "  Baseline: ${STRATEGY}"
echo "═══════════════════════════════════════════════════════════"
echo "  experiment:  ${EXP_NAME}"
echo "  agent model: ${AGENT_MODEL}"
echo "  user model:  ${USER_MODEL}"
echo "  versions:    ${VERSIONS[*]}"
echo "═══════════════════════════════════════════════════════════"
echo ""

echo "── Step 1: Tool retrieval index ──"
ensure_tool_index "${VERSIONS[@]}"
echo ""

echo "── Step 2: Rollout ──"
for v in "${VERSIONS[@]}"; do
  RUN_NAME="${RUN_NAME_PATTERN/\{version\}/$v}"
  echo "  [ROLLOUT] ${RUN_NAME}"

  run_step "${LOG_DIR}/${RUN_NAME}.log" \
    "${DTAU[@]}" rollout \
    --run-name "$RUN_NAME" \
    --strategy "$STRATEGY" \
    --split test \
    --version "$v" \
    --agent-model "$AGENT_MODEL" \
    --user-model "$USER_MODEL" \
    --max-workers "$MAX_WORKERS" \
    --query-workers "$QUERY_WORKERS" \
    --rag-index-dir "${RAG_INDEX_BASE}/${v}"
done
echo ""

echo "── Step 3: Evaluate ──"
for v in "${VERSIONS[@]}"; do
  RUN_NAME="${RUN_NAME_PATTERN/\{version\}/$v}"
  echo "  [EVAL] ${RUN_NAME}"

  run_step "${LOG_DIR}/${RUN_NAME}_eval.log" \
    "${DTAU[@]}" evaluate \
    --run-name "$RUN_NAME" \
    --version "$v" \
    --workers "$EVAL_WORKERS" \
    --verbose
done
echo ""

{
  echo "── Step 4: Aggregate metrics (including SPE) ──"
  run_step "${LOG_DIR}/${EXP_NAME}_multi_eval.log" \
    "${DTAU[@]}" evaluate-multi \
    --experiment-name "$EXP_NAME" \
    --versions "${VERSIONS[@]}" \
    --rollout-pattern "$RUN_NAME_PATTERN" \
    --output-dir "experiments/${EXP_NAME}_multi_eval/" \
    --workers "$EVAL_WORKERS" \
    --weight-modes unit cost \
    --report-format both \
    --verbose
}

if [ "$DRY_RUN" = "1" ]; then
  exit 0
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Done -- ${EXP_NAME} (${AGENT_MODEL})"
echo "═══════════════════════════════════════════════════════════"
for v in "${VERSIONS[@]}"; do
  RUN_NAME="${RUN_NAME_PATTERN/\{version\}/$v}"
  RESULT_FILE="experiments/${RUN_NAME}/evaluation/evaluation_results.jsonl"
  if [ -f "$RESULT_FILE" ]; then
    TOTAL=$(wc -l < "$RESULT_FILE")
    PASSED=$(grep -c '"passed": true' "$RESULT_FILE" 2>/dev/null || true)
    echo "  ${v}: ${PASSED:-0}/${TOTAL}"
  else
    echo "  ${v}: (incomplete)"
  fi
done
echo ""
echo "  Cross-version report: experiments/${EXP_NAME}_multi_eval/multi_version_report.md"
