#!/usr/bin/env bash
#SBATCH --job-name=gnn-h100-bs-sweep
#SBATCH --partition=gpu-h100-dchen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.out
#SBATCH --error=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.err
set -Eeuo pipefail

# Standalone Short-Q precompute/inference sweep ONLY.
# This does NOT start LMCache or vLLM and does NOT run cold/warm serving.
# Requires the earlier batching patch that adds --inference_batch_size.

REPO=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
VENV=/mnt/shared/gpfs/home/sriramc2/venvs/kvaware-vllm029
CUDA_HOME=/usr/local/cuda-13.1
RUN_ROOT=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro

BATCH_CSV="${1:-1,2,4,8,16}"
NUM_QUESTIONS="${2:-96}"
DATASET_NAME="${DATASET_NAME:-medical}"
FEATURE_MODEL="${FEATURE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
# Match the real H100 profile's serving tokenizer. Only the tokenizer is loaded;
# the 70B serving weights are NOT loaded by this standalone precompute sweep.
SERVING_MODEL="${SERVING_MODEL:-meta-llama/Llama-3.3-70B-Instruct}"
CHECKPOINT="${REPO}/Hierarchical_KV/shortq_placement/model.pt"

SWEEP_DIR="${RUN_ROOT}/h100_shortq_gnn_batch_sweep_${SLURM_JOB_ID}"
mkdir -p "$SWEEP_DIR" "${RUN_ROOT}/logs"

cd "$REPO"
source "${VENV}/bin/activate"
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/mnt/shared/gpfs/home/sriramc2/.cache/huggingface}"

# Preserve Slurm's GPU assignment and bypass any cluster MPS socket for this
# direct-CUDA standalone job, matching the current H100 project convention.
export CUDA_MPS_PIPE_DIRECTORY="/tmp/kvaware-no-mps-${USER}-${SLURM_JOB_ID}-DO-NOT-CREATE"

printf 'SLURM_JOB_GPUS=%s\n' "${SLURM_JOB_GPUS:-<unset>}"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
assert torch.cuda.device_count() == 1, torch.cuda.device_count()
print("torch_cuda_device:", torch.cuda.get_device_name(0))
print("torch_version:", torch.__version__)
print("torch_cuda:", torch.version.cuda)
PY

IFS=',' read -r -a BATCHES <<< "$BATCH_CSV"
RESULTS_TSV="${SWEEP_DIR}/sweep.tsv"
printf 'batch\tstatus\tloop_s\trps\telapsed_s\tpeak_cuda_gib\tmetadata_matches_bs1\n' > "$RESULTS_TSV"
REF_METADATA=""

for BS in "${BATCHES[@]}"; do
  BS="${BS//[[:space:]]/}"
  [[ -n "$BS" ]] || continue
  OUT="${SWEEP_DIR}/bs${BS}"
  mkdir -p "$OUT"
  echo "===== H100 batch_size=${BS}, questions=${NUM_QUESTIONS} ====="

  set +e
  python experiments/precompute_shortq_placements.py \
    --dataset_name "$DATASET_NAME" \
    --max_questions "$NUM_QUESTIONS" \
    --retrieval_top_k 5 \
    --request_order legacy_prefix_hash \
    --prefix_sort_depth 2 \
    --request_order_seed 0 \
    --serving_model "$SERVING_MODEL" \
    --feature_model "$FEATURE_MODEL" \
    --checkpoint "$CHECKPOINT" \
    --chunk_size 512 \
    --max_seq_len 8000 \
    --device cuda:0 \
    --inference_batch_size "$BS" \
    --runtime_metadata "${OUT}/runtime_hash_to_tier.json" \
    --placement_trace "${OUT}/gnn_chunk_placements.jsonl" \
    --hash_prediction_trace "${OUT}/gnn_hash_prediction_occurrences.jsonl" \
    --timing_trace "${OUT}/gnn_prediction_timing.jsonl" \
    --summary "${OUT}/gnn_placement_summary.json" \
    --smoke_force_min_unique_per_tier 0 \
    > "${OUT}/precompute.log" 2>&1
  RC=$?
  set -e

  if (( RC != 0 )); then
    printf '%s\tFAIL(rc=%s)\t\t\t\t\t\n' "$BS" "$RC" >> "$RESULTS_TSV"
    tail -n 100 "${OUT}/precompute.log" || true
    continue
  fi

  if [[ "$BS" == "1" ]]; then
    REF_METADATA="${OUT}/runtime_hash_to_tier.json"
    MATCH=yes
  elif [[ -n "$REF_METADATA" ]] && cmp -s "$REF_METADATA" "${OUT}/runtime_hash_to_tier.json"; then
    MATCH=yes
  elif [[ -n "$REF_METADATA" ]]; then
    MATCH=NO
  else
    MATCH=no_bs1_reference
  fi

  python - "$BS" "${OUT}/gnn_placement_summary.json" "$MATCH" >> "$RESULTS_TSV" <<'PY'
import json, sys
bs, path, match = sys.argv[1:]
with open(path) as f:
    s = json.load(f)
peak = s.get("peak_cuda_memory_bytes")
peak_gib = "" if peak is None else f"{peak / 2**30:.3f}"
print("\t".join([
    bs,
    "OK",
    f"{s['request_prediction_loop_seconds']:.6f}",
    f"{s['request_prediction_throughput_rps']:.6f}",
    f"{s['elapsed_seconds']:.6f}",
    peak_gib,
    match,
]))
PY

done

echo
echo "===== H100 GNN BATCH SWEEP RESULTS ====="
column -t -s $'\t' "$RESULTS_TSV" || cat "$RESULTS_TSV"
echo "sweep_dir=$SWEEP_DIR"
