#!/usr/bin/env bash
#SBATCH --job-name=gnn-head-probe-l40
#SBATCH --partition=gpu-l40-dchen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/shared/gpfs/home/sriramc2/runs/kvaware/%x_%j.out
#SBATCH --error=/mnt/shared/gpfs/home/sriramc2/runs/kvaware/%x_%j.err

# Single GNN PREDICTION ONLY run: no vLLM server, no LMCache server,
# no cache storage or cleanup, no queued H100 source modification.
set -Eeuo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
REPO=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
VENV=/mnt/shared/gpfs/home/sriramc2/venvs/kvaware-vllm029
ROOT=/mnt/shared/gpfs/home/sriramc2/runs/kvaware
RUN="$ROOT/l40_gnn_head_probe_original_q650_${SLURM_JOB_ID}"
REF="${REFERENCE_RUN_DIR:-$ROOT/l40_bayesian_dynamic_vpc_q650_l40_bayes_three_cost_original_b200_mns10_29862}"
PRIOR="${L40_PRIOR_PATH:-$ROOT/l40_three_cost_preflight_29859/rank_l40_calibrated.json}"
mkdir -p "$RUN"/placement "$RUN"/analysis "$RUN"/logs
archive_on_exit() {
    local rc=$?
    trap - EXIT
    (cd "$ROOT" && zip -qr "$(basename "$RUN").zip" "$(basename "$RUN")") || true
    echo "[GNN_HEAD_PROBE_ARCHIVE] $RUN.zip exit_status=$rc" >&2
    exit "$rc"
}
trap archive_on_exit EXIT
cd "$REPO"
source "$VENV/bin/activate"
[[ -f experiments/precompute_shortq_heads_probe.py ]] || { echo 'Missing probe .py' >&2; exit 2; }
[[ -f experiments/analyze_shortq_heads_probe.py ]] || { echo 'Missing analysis .py' >&2; exit 2; }
[[ -f "$REF/placement/gnn_chunk_scores.jsonl" ]] || { echo "Missing original Q650 scores under $REF" >&2; exit 2; }
[[ -f "$REF/results/bayes_wave_history.json" ]] || { echo "Missing original Bayesian wave history under $REF" >&2; exit 2; }
[[ -f "$PRIOR" ]] || { echo "Missing L40 calibration prior at $PRIOR" >&2; exit 2; }

# Preserve source provenance; no use of an existing queued H100 job directory.
{
  date -Is; hostname; echo "SLURM_JOB_ID=$SLURM_JOB_ID"
  git rev-parse HEAD || true
  git status --short || true
  sha256sum experiments/precompute_shortq_heads_probe.py experiments/analyze_shortq_heads_probe.py Hierarchical_KV/shortq_placement/model.pt
  scontrol show job "$SLURM_JOB_ID" | head -60 || true
} > "$RUN/provenance.txt" 2>&1

# Exact L40/29862 prompt conditions: medical, original top-5, Q650, seed0
# request order; same feature + serving 8B tokenizer and ranker checkpoint.
# These vote envs DO NOT affect GNN output: offline analyzer explicitly
# reconstructs both t8/t22, class-head and rank-head counterfactuals.
export GNN_CHUNK_VOTE_POLICY=threshold_with_cpu_top2
export GNN_CHUNK_GPU_MIN=22 GNN_CHUNK_CPU_MIN=2 GNN_CHUNK_CPU_TOP2_MIN=6
export PYTHONUNBUFFERED=1
python experiments/precompute_shortq_heads_probe.py \
  --dataset_name medical \
  --max_questions 650 \
  --retrieval_top_k 5 \
  --passage_order_policy original \
  --request_order seeded_shuffle \
  --request_order_seed 0 \
  --serving_model meta-llama/Llama-3.1-8B-Instruct \
  --feature_model meta-llama/Llama-3.1-8B-Instruct \
  --checkpoint Hierarchical_KV/shortq_placement/model.pt \
  --chunk_size 512 \
  --max_seq_len 8000 \
  --device cuda:0 \
  --inference_batch_size 1 \
  --bayes_scores_only \
  --bayes_score_file "$RUN/placement/gnn_chunk_scores.jsonl" \
  --prediction_heads_file "$RUN/placement/gnn_raw_heads_16token.jsonl" \
  --runtime_metadata "$RUN/placement/unused_runtime_metadata.json" \
  --placement_trace "$RUN/placement/unused_placements.jsonl" \
  --hash_prediction_trace "$RUN/placement/unused_hash_predictions.jsonl" \
  --timing_trace "$RUN/placement/gnn_prediction_timings.jsonl" \
  --summary "$RUN/placement/gnn_precompute_summary.json" \
  > "$RUN/logs/precompute.log" 2>&1

python experiments/analyze_shortq_heads_probe.py \
  --heads "$RUN/placement/gnn_raw_heads_16token.jsonl" \
  --scores "$RUN/placement/gnn_chunk_scores.jsonl" \
  --reference-run-dir "$REF" \
  --prior-file "$PRIOR" \
  --gpu-gb 2 --cpu-gb 64 --disk-gb 100 \
  --out "$RUN/analysis" \
  > "$RUN/logs/analysis.log" 2>&1

# Hard gate: wrong prompt hashes or replay disagreement means comparison invalid.
python - "$RUN" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1]);s=json.loads((r/'analysis/analysis_summary.json').read_text())
assert s['num_requests']==650, s
assert s['historical_29862_join']['hash_mismatches']==0, s
assert s['historical_29862_join']['max_rank_score_absolute_difference']<0.001, s
replay=s['bayesian_replay_vs_runtime']
assert replay['max_threshold_diff']<1e-8, replay
assert replay['final_metadata_mismatch_hashes']==0, replay
assert replay['wave_count_mismatches']==0, replay
print('[GNN_HEAD_PROBE_VALIDATED]',r,s['chunk_occurrences'],s['unique_prefix_hashes'],replay,flush=True)
PY

# Retain the exact reference inputs for standalone replay review later, without
# duplicating the 34-MB prompt records or sensitive model checkpoint.
cp "$PRIOR" "$RUN/analysis/reference_l40_measured_prior.json"
cp "$REF/results/bayes_wave_history.json" "$RUN/analysis/reference_29862_bayes_wave_history.json"
cp "$REF/run_config.json" "$RUN/analysis/reference_29862_run_config.json"
cp "$REF/placement/gnn_chunk_scores.jsonl" "$RUN/analysis/reference_29862_gnn_chunk_scores.jsonl"
cp "$REF/placement/runtime_hash_to_placement.json" "$RUN/analysis/reference_29862_runtime_hash_to_placement.json"
echo "[GNN_HEAD_PROBE_DONE] dir=$RUN archive=$RUN.zip" >&2
