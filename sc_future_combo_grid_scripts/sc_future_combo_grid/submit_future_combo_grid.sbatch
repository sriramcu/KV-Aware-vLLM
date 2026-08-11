#!/bin/bash
#SBATCH --partition=gpu-h100-dchen
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16

set -euxo pipefail

REPO=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

if [[ "${SC_LMCACHE_SOURCE_DEFAULT_KNOBS:-1}" == "1" ]]; then
  # The grid exports every tested SC_* value explicitly. The shared file fills
  # in only unrelated defaults and validates the values before Python starts.
  source "$REPO/local_repro/sc_lmcache_knobs.sh"
fi

: "${GRID_RUN_ID:?GRID_RUN_ID is required}"
: "${GRID_RUN_LABEL:?GRID_RUN_LABEL is required}"
: "${GRID_CONFIG_SHA256:?GRID_CONFIG_SHA256 is required}"
: "${MAX_QUESTIONS:?MAX_QUESTIONS is required}"
: "${SUBMISSION_BATCH_SIZE:?SUBMISSION_BATCH_SIZE is required}"
: "${DATASET_NAME:?DATASET_NAME is required}"
: "${VLLM_MAX_NUM_SEQS:?VLLM_MAX_NUM_SEQS is required}"

export PYTHONFAULTHANDLER=1

printf '%s\n' \
  "=== SC FUTURE COMBO GRID JOB ===" \
  "GRID_RUN_ID=$GRID_RUN_ID" \
  "GRID_RUN_LABEL=$GRID_RUN_LABEL" \
  "GRID_CONFIG_SHA256=$GRID_CONFIG_SHA256" \
  "MAX_QUESTIONS=$MAX_QUESTIONS" \
  "DATASET_NAME=$DATASET_NAME" \
  "SUBMISSION_BATCH_SIZE=$SUBMISSION_BATCH_SIZE" \
  "VLLM_MAX_NUM_SEQS=$VLLM_MAX_NUM_SEQS" \
  "SC_LMCACHE_DATA_DIR=${SC_LMCACHE_DATA_DIR:-driver-default}"

"$REPO/local_repro/run_driver.sh" wo_gnn 2>&1
