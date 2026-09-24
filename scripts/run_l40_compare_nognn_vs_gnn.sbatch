#!/usr/bin/env bash
#SBATCH --job-name=l40-nognn-vs-gnn
#SBATCH --partition=gpu-l40-dchen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.out
#SBATCH --error=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.err
set -Eeuo pipefail

GNN_BS="${1:?usage: sbatch $0 <gnn_inference_batch_size>}"
REPO=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
RUN_ROOT=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro
COMPARE_DIR="${RUN_ROOT}/l40_nognn_vs_gnn_${SLURM_JOB_ID}"
mkdir -p "$COMPARE_DIR" "${RUN_ROOT}/logs"
cd "$REPO"

# Keep the safeguards/diagnostics identical where the no-GNN runner inherits them.
export LMCACHE_MP_STAGE1_FRESHNESS_GUARD_S=270
export LMCACHE_MP_CONGESTION_DEBUG=1
export LMCACHE_MP_CONGESTION_LOG_EVERY=25
export LMCACHE_MP_CONGESTION_SLOW_S=5
export LMCACHE_MP_CHTHM_DEBUG=1

# ---------------------------------------------------------------------------
# 1) Normal full project, no GNN. VPC remains enabled.
#    0.64 * 48 GiB is approximately the same total GPU budget as
#    GNN's 0.60 vLLM pool + 2 GiB persistent L0.
# ---------------------------------------------------------------------------
echo "===== FULL NO-GNN BASELINE ====="
env \
  MODEL=meta-llama/Llama-3.1-8B-Instruct \
  QUANTIZATION= \
  TENSOR_PARALLEL_SIZE=1 \
  MAX_MODEL_LEN=8000 \
  GPU_MEMORY_UTILIZATION=0.64 \
  MAX_NUM_SEQS=12 \
  DATASET_NAME=medical \
  NUM_QUESTIONS=200 \
  RETRIEVAL_TOP_K=5 \
  REQUEST_ORDER=legacy_prefix_hash \
  PREFIX_SORT_DEPTH=2 \
  REQUEST_ORDER_SEED=0 \
  WARM_ORDER=reverse \
  SUBMISSION_BATCH_SIZE=200 \
  MIN_TOKENS=32 \
  MAX_TOKENS=128 \
  ENABLE_VPC=1 \
  LMCACHE_CHUNK_SIZE=512 \
  LMCACHE_L1_SIZE_GB=64 \
  LMCACHE_MAX_WORKERS=8 \
  LMCACHE_L2_WORKERS=4 \
  LMCACHE_L2_PREFETCH_MAX_IN_FLIGHT=2 \
  CACHE_TAG="compare_nognn_${SLURM_JOB_ID}" \
  CLEAN_OLD_LMCACHE=1 \
  bash scripts/run_l40_mp_nognn_q10.sbatch

NO_GNN_DIR="${RUN_ROOT}/l40_mp_nognn_${SLURM_JOB_ID}"
cp "${NO_GNN_DIR}/results/summary.json" "${COMPARE_DIR}/nognn_summary.json"

# Make sure the first stack has actually relinquished the ports/GPU before phase 2.
for _ in $(seq 1 30); do
  if ! ss -ltn | grep -Eq ':5556[[:space:]]|:8000[[:space:]]'; then break; fi
  sleep 1
done

# ---------------------------------------------------------------------------
# 2) Full GNN project run. No smoke tier forcing for performance comparison.
# ---------------------------------------------------------------------------
echo "===== FULL GNN RUN (inference batch=${GNN_BS}) ====="
export KV_GNN_PROFILE=l40_smoke
export GNN_INFERENCE_BATCH_SIZE="$GNN_BS"
export SMOKE_FORCE_MIN_UNIQUE_PER_TIER=0
bash scripts/run_shortq_gnn_exclusive_common.sh

GNN_DIR="${RUN_ROOT}/l40_shortq_gnn_smoke_${SLURM_JOB_ID}"
cp "${GNN_DIR}/results/summary.json" "${COMPARE_DIR}/gnn_summary.json"
cp "${GNN_DIR}/placement/gnn_placement_summary.json" "${COMPARE_DIR}/gnn_precompute_summary.json"

python - \
  "${COMPARE_DIR}/nognn_summary.json" \
  "${COMPARE_DIR}/gnn_summary.json" \
  "${COMPARE_DIR}/gnn_precompute_summary.json" \
  "$GNN_BS" \
  "${COMPARE_DIR}/comparison.json" <<'PY'
import json, sys
n_path, g_path, p_path, bs, out_path = sys.argv[1:]
n = json.load(open(n_path))
g = json.load(open(g_path))
p = json.load(open(p_path))

def phase(s, name):
    x = s[name]
    return {
        "requests": x["requests"],
        "successful": x["successful"],
        "elapsed_seconds": x["elapsed_seconds"],
        "prompt_tokens_per_second": x.get("prompt_tokens_per_second"),
        "request_latency_mean_seconds": x.get("request_latency_mean_seconds"),
    }

out = {
    "gnn_inference_batch_size": int(bs),
    "nognn": {"cold": phase(n, "cold"), "warm": phase(n, "warm")},
    "gnn": {"cold": phase(g, "cold"), "warm": phase(g, "warm")},
    "gnn_precompute": {
        "elapsed_seconds": p["elapsed_seconds"],
        "request_prediction_loop_seconds": p["request_prediction_loop_seconds"],
        "request_prediction_throughput_rps": p.get("request_prediction_throughput_rps"),
        "peak_cuda_memory_bytes": p.get("peak_cuda_memory_bytes"),
    },
}
json.dump(out, open(out_path, "w"), indent=2)
print(json.dumps(out, indent=2))
PY

echo "comparison_dir=$COMPARE_DIR"
