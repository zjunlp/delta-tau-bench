#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
#  Steward: train phase — build one skill library per catalog version
# ══════════════════════════════════════════════════════════════════════════════
#
#  For each version in order:
#    train rollout -> evaluate -> bootstrap/evolve the library -> build its index
#
#  The first version has no previous library, so its rollout uses
#  single_domain_full; later versions use train_steward with the library of the
#  preceding version.
#
#  Usage:
#    bash scripts/run_steward_train.sh
#    VERSIONS="v1 v2"                    bash scripts/run_steward_train.sh
#    AGENT_MODEL=my-model EXP_NAME=my    bash scripts/run_steward_train.sh
#    DRY_RUN=1                           bash scripts/run_steward_train.sh
#
#  Re-running the same command resumes: finished stages are skipped, tracked in
#  experiments/{EXP_NAME}/run_state.json.
#
#  Run scripts/run_steward_test.sh afterwards with the same EXP_NAME.
# ══════════════════════════════════════════════════════════════════════════════

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/_steward_env.sh"

echo "═══════════════════════════════════════════════════════════"
echo "  Steward — train phase"
echo "═══════════════════════════════════════════════════════════"
echo "  experiment:   ${EXP_NAME}"
echo "  agent model:  ${AGENT_MODEL}"
echo "  user model:   ${USER_MODEL}"
echo "  evolve model: ${EVOLVE_MODEL}"
echo "  versions:     ${VERSIONS[*]}"
echo "  strategies:   bootstrap=${BOOTSTRAP_STRATEGY} train=${TRAIN_STRATEGY}"
echo "  rollout:      domains=${ROLLOUT_DOMAIN_WORKERS} queries=${ROLLOUT_QUERY_WORKERS}"
echo "  evaluate:     workers=${EVALUATE_WORKERS}"
echo "  skill:        batch=${SKILL_BATCH_SIZE} concurrent=${SKILL_MAX_CONCURRENT} merge_embedding_batch=${MERGE_EMBEDDING_BATCH_SIZE} index_embedding_batch=${EMBEDDING_BATCH_SIZE}"
echo "═══════════════════════════════════════════════════════════"
echo ""

echo "── Step 1: Tool retrieval index ──"
ensure_tool_index "${VERSIONS[@]}" || exit $?
echo ""

echo "── Step 2: Version chain (train only) ──"
run_step "${LOG_DIR}/${EXP_NAME}_train.log" \
  "${DTAU[@]}" chain "${CHAIN_OPTS[@]}" --skip-test || exit $?

if [ "$DRY_RUN" = "1" ]; then
  exit 0
fi

for v in "${VERSIONS[@]}"; do
  LIB_ROOT="experiments/${EXP_NAME}_evolve_to_${v}/skills"
  if [ -d "${LIB_ROOT}/library" ]; then
    LIB="${LIB_ROOT}/library"
  else
    LIB="$LIB_ROOT"
  fi
  if [ ! -s "${LIB}/plan_library.json" ] || \
     [ ! -s "${LIB}/library_metadata.json" ] || \
     [ ! -s "${LIB}/skill_graph.json" ]; then
    echo "[ERROR] Incomplete skill library for ${v}: ${LIB}" >&2
    exit 1
  fi
done

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Train phase done -- ${EXP_NAME}"
echo "═══════════════════════════════════════════════════════════"
for v in "${VERSIONS[@]}"; do
  LIB_ROOT="experiments/${EXP_NAME}_evolve_to_${v}/skills"
  if [ -d "${LIB_ROOT}/library" ]; then
    LIB="${LIB_ROOT}/library"
  else
    LIB="$LIB_ROOT"
  fi
  echo "  ${v}: ${LIB}"
done
echo ""
echo "  Next: EXP_NAME=${EXP_NAME} bash scripts/run_steward_test.sh"
