#!/bin/bash
set -euxo pipefail

MODE="${1:-wo_gnn}"
shift || true
export MODE

if [[ "$MODE" != "with_gnn" && "$MODE" != "wo_gnn" ]]; then
  echo "Unknown mode: $MODE"
  echo "usage: run_driver.sh with_gnn|wo_gnn"
  exit 2
fi

HOME_DIR=/mnt/shared/gpfs/home/sriramc2
REPO="$HOME_DIR/KV-Aware-vLLM"
RUN_ROOT="$HOME_DIR/runs/kvaware_repro"

DATASET_NAME="${DATASET_NAME:-hotpotqa}"
MAX_QUESTIONS="${MAX_QUESTIONS:-250}"
QUESTIONS_JSON="$REPO/Hierarchical_KV/LinearRAG/dataset/${DATASET_NAME}/questions.json"

source "$HOME_DIR/venvs/kvaware/bin/activate"

mkdir -p "$RUN_ROOT/logs"
mkdir -p "$RUN_ROOT/sidecars"
mkdir -p "$RUN_ROOT/lmcache_vllm"
mkdir -p "$RUN_ROOT/lmcache_hit_hook"
mkdir -p "$RUN_ROOT/prometheus_vllm"
mkdir -p "$RUN_ROOT/tmp"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"

export HF_HOME="$HOME_DIR/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"

export VLLM_USE_V1=1
export PYTHONHASHSEED=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

if [[ "$MODE" == "with_gnn" ]]; then
  export VLLM_KV_IMPORTANCE_ENABLE=1
else
  export VLLM_KV_IMPORTANCE_ENABLE=0
fi

export GNN_KV_BLOCK_SIZE=16

JOB_TAG="${DATASET_NAME}_${MODE}_${SLURM_JOB_ID:-manual}"

export VLLM_KV_IMPORTANCE_TIERS="$RUN_ROOT/sidecars/kv_importance_tiers_${JOB_TAG}.json"

mkdir -p "$RUN_ROOT/lmcache_vllm"
mkdir -p "$RUN_ROOT/lmcache_hit_hook"
mkdir -p "$RUN_ROOT/prometheus_vllm"

# delete old jobs' cache dirs if no other job is running
if [[ "${CLEAN_OLD_LMCACHE:-1}" == "1" ]]; then
  if squeue -u "$USER" -h | grep -v "${SLURM_JOB_ID:-NO_CURRENT_JOB}" | grep -q .; then
    echo "Other jobs are running; not deleting old LMCache caches."
    squeue -u "$USER"
  else
    echo "Deleting old LMCache caches..."
    rm -rf "$RUN_ROOT/lmcache_vllm"/*
    rm -rf "$RUN_ROOT/lmcache_hit_hook"/*
    rm -rf "$RUN_ROOT/prometheus_vllm"/*
  fi
fi

export SRIRAM_LMCACHE_DIR="$RUN_ROOT/lmcache_vllm/${JOB_TAG}"
export LMCACHE_HOOK_LOG_DIR="$RUN_ROOT/lmcache_hit_hook/${JOB_TAG}"
export PROMETHEUS_MULTIPROC_DIR="$RUN_ROOT/prometheus_vllm/${JOB_TAG}"

rm -rf "$SRIRAM_LMCACHE_DIR" "$LMCACHE_HOOK_LOG_DIR" "$PROMETHEUS_MULTIPROC_DIR"
mkdir -p "$SRIRAM_LMCACHE_DIR" "$LMCACHE_HOOK_LOG_DIR" "$PROMETHEUS_MULTIPROC_DIR"

# Use short node-local path only for sockets/temp files.
export TMPDIR="/tmp/sr_${SLURM_JOB_ID:-manual}"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
rm -rf "$TMPDIR"
mkdir -p "$TMPDIR"

echo "=== RUN CONFIG ==="
date
hostname
whoami
echo "REPO=$REPO"
echo "MODE=$MODE"
echo "DATASET_NAME=$DATASET_NAME"
echo "MAX_QUESTIONS=$MAX_QUESTIONS"
echo "QUESTIONS_JSON=$QUESTIONS_JSON"
echo "VLLM_KV_IMPORTANCE_ENABLE=$VLLM_KV_IMPORTANCE_ENABLE"
echo "VLLM_KV_IMPORTANCE_TIERS=$VLLM_KV_IMPORTANCE_TIERS"
echo "SRIRAM_LMCACHE_DIR=$SRIRAM_LMCACHE_DIR"
echo "LMCACHE_HOOK_LOG_DIR=$LMCACHE_HOOK_LOG_DIR"
echo "PROMETHEUS_MULTIPROC_DIR=$PROMETHEUS_MULTIPROC_DIR"
echo "TMPDIR=$TMPDIR"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-unset}"

if [[ ! -f "$QUESTIONS_JSON" ]]; then
  echo "Missing questions file: $QUESTIONS_JSON"
  exit 3
fi

nvidia-smi

echo "=== PYTHON LOCATION ==="
which python
python --version

echo "=== VLLM LOCATION ==="
python - <<'PY'
import vllm
print(vllm.__file__)
PY

echo "=== START DRIVER ==="
cd "$TMPDIR"

python "$REPO/local_repro/cpu_offload_lmcache_sriram.py" \
  --llm_model meta-llama/Llama-3.3-70B-Instruct \
  --dataset_name "$DATASET_NAME" \
  --questions_json "$QUESTIONS_JSON" \
  --max_questions "$MAX_QUESTIONS" \
  "$@"

echo "=== SIDE CAR CHECK ==="
ls -lh "$VLLM_KV_IMPORTANCE_TIERS" || true

python - <<'PY'
import json, os
p = os.environ["VLLM_KV_IMPORTANCE_TIERS"]
print("sidecar:", p)
if os.path.exists(p):
    with open(p) as f:
        data = json.load(f)
    print("num requests in sidecar:", len(data))
    first_key = next(iter(data), None)
    print("first key:", first_key)
    if first_key is not None:
        blocks = data[first_key]
        print("num blocks first request:", len(blocks))
        first_block = next(iter(blocks), None)
        print("first block:", first_block, blocks[first_block] if first_block is not None else None)
else:
    print("sidecar missing")
PY

echo "=== CACHE SIZE ==="
du -sh "$SRIRAM_LMCACHE_DIR" || true

echo "=== DONE ==="
date