#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
#  Steward: test phase — test rollouts, evaluation, cross-version aggregation
# ══════════════════════════════════════════════════════════════════════════════
#
#  Reuses the skill libraries produced by scripts/run_steward_train.sh, so it
#  must run with the same EXP_NAME. Each test version uses the library evolved
#  for that version.
#
#  Usage:
#    bash scripts/run_steward_test.sh
#    VERSIONS="v1 v2"                    bash scripts/run_steward_test.sh
#    EXP_NAME=my                         bash scripts/run_steward_test.sh
#    DRY_RUN=1                           bash scripts/run_steward_test.sh
#
#  Re-running the same command resumes: finished stages are skipped, tracked in
#  experiments/{EXP_NAME}/run_state.json.
# ══════════════════════════════════════════════════════════════════════════════

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/_steward_env.sh"

RUN_NAME_PATTERN="${EXP_NAME}_test_{version}"

echo "═══════════════════════════════════════════════════════════"
echo "  Steward — test phase"
echo "═══════════════════════════════════════════════════════════"
echo "  experiment:  ${EXP_NAME}"
echo "  agent model: ${AGENT_MODEL}"
echo "  user model:  ${USER_MODEL}"
echo "  versions:    ${VERSIONS[*]}"
echo "  strategy:    ${TEST_STRATEGY}"
echo "  rollout:     domains=${ROLLOUT_DOMAIN_WORKERS} queries=${ROLLOUT_QUERY_WORKERS}"
echo "  evaluate:    workers=${EVALUATE_WORKERS}"
echo "  skill:       batch=${SKILL_BATCH_SIZE} concurrent=${SKILL_MAX_CONCURRENT} merge_embedding_batch=${MERGE_EMBEDDING_BATCH_SIZE} index_embedding_batch=${EMBEDDING_BATCH_SIZE}"
echo "═══════════════════════════════════════════════════════════"
echo ""

echo "── Step 1: Tool retrieval index ──"
ensure_tool_index "${VERSIONS[@]}" || exit $?
echo ""

echo "── Step 2: Test rollouts and evaluation ──"
run_step "${LOG_DIR}/${EXP_NAME}_test.log" \
  "${DTAU[@]}" chain "${CHAIN_OPTS[@]}" --skip-train || exit $?

{
  echo ""
  echo "── Step 3: Aggregate metrics (including SPE) ──"
  run_step "${LOG_DIR}/${EXP_NAME}_multi_eval.log" \
    "${DTAU[@]}" evaluate-multi \
    --experiment-name "$EXP_NAME" \
    --versions "${VERSIONS[@]}" \
    --rollout-pattern "$RUN_NAME_PATTERN" \
    --output-dir "experiments/${EXP_NAME}_multi_eval/" \
    --workers "$EVALUATE_WORKERS" \
    --weight-modes unit cost \
    --report-format both \
    --verbose || exit $?
}

if [ "$DRY_RUN" = "1" ]; then
  exit 0
fi

for v in "${VERSIONS[@]}"; do
  RUN_NAME="${RUN_NAME_PATTERN/\{version\}/$v}"
  RESULT_FILE="experiments/${RUN_NAME}/evaluation/evaluation_results.jsonl"
  REPORT_FILE="experiments/${RUN_NAME}/evaluation/evaluation_report.json"
  if [ ! -s "$RESULT_FILE" ] || [ ! -s "$REPORT_FILE" ]; then
    echo "[ERROR] Incomplete evaluation for ${v}: ${RESULT_FILE}" >&2
    exit 1
  fi
done
if { [ ! -s "experiments/${EXP_NAME}_multi_eval/multi_version_report.json" ] || \
     [ ! -s "experiments/${EXP_NAME}_multi_eval/multi_version_report.md" ]; }; then
  echo "[ERROR] Cross-version report is incomplete for ${EXP_NAME}" >&2
  exit 1
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Done -- ${EXP_NAME} (${AGENT_MODEL})"
echo "═══════════════════════════════════════════════════════════"
for v in "${VERSIONS[@]}"; do
  RUN_NAME="${RUN_NAME_PATTERN/\{version\}/$v}"
  RESULT_FILE="experiments/${RUN_NAME}/evaluation/evaluation_results.jsonl"
  TOTAL=$(wc -l < "$RESULT_FILE")
  PASSED=$(grep -c '"passed": true' "$RESULT_FILE" 2>/dev/null || true)
  echo "  ${v}: ${PASSED:-0}/${TOTAL}"
done
echo ""
echo "  Cross-version report: experiments/${EXP_NAME}_multi_eval/multi_version_report.md"
