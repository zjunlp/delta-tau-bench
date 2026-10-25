#!/bin/bash
# Shared configuration for the two Steward entry points. Sourced, not executed.

read -r -a VERSIONS <<< "${VERSIONS:-v1 v2 v3 v4 v5 v6}"

AGENT_MODEL="${AGENT_MODEL:-deepseek-v4-pro}"
USER_MODEL="${USER_MODEL:-deepseek-flash}"
EVOLVE_MODEL="${EVOLVE_MODEL:-$AGENT_MODEL}"

# The experiment name carries the model so that two models never write into the
# same directory.
EXP_NAME="${EXP_NAME:-steward_$(echo "$AGENT_MODEL" | tr '[:upper:]./' '[:lower:]__')}"

BOOTSTRAP_STRATEGY="${BOOTSTRAP_STRATEGY:-single_domain_full}"
TRAIN_STRATEGY="${TRAIN_STRATEGY:-train_steward}"
TEST_STRATEGY="${TEST_STRATEGY:-steward}"

# Conservative defaults for CPU-only hosts. Override individual values through
# the environment when the machine has enough memory and cores.
ROLLOUT_DOMAIN_WORKERS="${ROLLOUT_DOMAIN_WORKERS:-1}"
ROLLOUT_QUERY_WORKERS="${ROLLOUT_QUERY_WORKERS:-4}"
EVALUATE_WORKERS="${EVALUATE_WORKERS:-2}"
SKILL_BATCH_SIZE="${SKILL_BATCH_SIZE:-2}"
SKILL_MAX_CONCURRENT="${SKILL_MAX_CONCURRENT:-2}"
MERGE_EMBEDDING_BATCH_SIZE="${MERGE_EMBEDDING_BATCH_SIZE:-8}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-8}"
ALLOW_RESOURCE_CHANGE_ON_RESUME="${ALLOW_RESOURCE_CHANGE_ON_RESUME:-0}"

LOG_DIR="logs/${EXP_NAME}"
if [ "$DRY_RUN" != "1" ]; then
  mkdir -p "$LOG_DIR"
fi

# Both phases drive the same chain command; only --skip-train/--skip-test differ.
CHAIN_OPTS=(
  --experiment-name "$EXP_NAME"
  --train-versions "${VERSIONS[@]}"
  --test-versions "${VERSIONS[@]}"
  --bootstrap-strategy "$BOOTSTRAP_STRATEGY"
  --train-strategy "$TRAIN_STRATEGY"
  --test-strategy "$TEST_STRATEGY"
  --agent-model "$AGENT_MODEL"
  --user-model "$USER_MODEL"
  --evolve-model "$EVOLVE_MODEL"
  --rollout-domain-workers "$ROLLOUT_DOMAIN_WORKERS"
  --rollout-query-workers "$ROLLOUT_QUERY_WORKERS"
  --evaluate-workers "$EVALUATE_WORKERS"
  --skill-batch-size "$SKILL_BATCH_SIZE"
  --skill-max-concurrent "$SKILL_MAX_CONCURRENT"
  --merge-embedding-batch-size "$MERGE_EMBEDDING_BATCH_SIZE"
  --embedding-batch-size "$EMBEDDING_BATCH_SIZE"
  --resume
  --verbose
)
if [ "$DRY_RUN" = "1" ]; then
  CHAIN_OPTS+=(--dry-run)
fi
if [ "$ALLOW_RESOURCE_CHANGE_ON_RESUME" = "1" ]; then
  CHAIN_OPTS+=(--allow-resource-change-on-resume)
fi
