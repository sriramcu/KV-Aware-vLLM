#!/bin/bash
set -euxo pipefail

MODE="${1:?usage: run_driver.sh with_gnn|wo_gnn}"

export HOME_DIR=/mnt/shared/gpfs/home/sriramc2
export PROJECT_DIR=$HOME_DIR/KV-Aware-vLLM
export RUN_ROOT=$HOME_DIR/runs/kvaware_repro

source $HOME_DIR/venvs/kvaware/bin/activate

cd "$PROJECT_DIR"

mkdir -p "$RUN_ROOT/outputs/$MODE"
mkdir -p "$RUN_ROOT/tmp/$MODE"
mkdir -p "$RUN_ROOT/logs"
mkdir -p "/tmp/sriramc2_lmcache_vllm"
mkdir -p "/tmp/sriramc2_prometheus_vllm"

export HF_HOME=$HOME_DIR/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_HUB_CACHE=$HF_HOME/hub

export VLLM_USE_V1=1
export PYTHONHASHSEED=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

export VLLM_KV_IMPORTANCE_TIERS="/tmp/kv_importance_tiers_${USER}_${MODE}_${SLURM_JOB_ID}.json"
export GNN_KV_BLOCK_SIZE=16

if [ "$MODE" = "with_gnn" ]; then
  export VLLM_KV_IMPORTANCE_ENABLE=1
elif [ "$MODE" = "wo_gnn" ]; then
  export VLLM_KV_IMPORTANCE_ENABLE=0
else
  echo "Unknown mode: $MODE"
  exit 2
fi

echo "=== RUN CONFIG ==="
date
hostname
whoami
pwd
echo "MODE=$MODE"
echo "VLLM_KV_IMPORTANCE_ENABLE=$VLLM_KV_IMPORTANCE_ENABLE"
echo "VLLM_KV_IMPORTANCE_TIERS=$VLLM_KV_IMPORTANCE_TIERS"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
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
python local_repro/cpu_offload_lmcache_sriram.py \
  --dataset_name hotpotqa \
  2>&1 | tee "$RUN_ROOT/outputs/$MODE/vllm-results-${MODE}-${SLURM_JOB_ID}.txt"

echo "=== SIDE CAR CHECK ==="
ls -lh "$VLLM_KV_IMPORTANCE_TIERS" || true
python - <<'PY'
import json, os
p = os.environ["VLLM_KV_IMPORTANCE_TIERS"]
print("sidecar:", p)
if os.path.exists(p):
    data = json.load(open(p))
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

echo "=== DONE ==="
date
