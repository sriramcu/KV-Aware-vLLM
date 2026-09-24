#!/usr/bin/env bash
set -Eeuo pipefail

PROFILE="${KV_GNN_PROFILE:?wrapper must set KV_GNN_PROFILE}"
REPO=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
VENV=/mnt/shared/gpfs/home/sriramc2/venvs/kvaware-vllm029
RUN_ROOT=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro
DATASET_NAME=medical
RETRIEVAL_TOP_K=5
REQUEST_ORDER=legacy_prefix_hash
PREFIX_SORT_DEPTH=2
REQUEST_ORDER_SEED=0
WARM_ORDER=reverse
LMCACHE_CHUNK_SIZE=512
LMCACHE_L2_PREFETCH_MAX_IN_FLIGHT=2
LMCACHE_MAX_WORKERS=8
LMCACHE_MP_TIMEOUT=30
FEATURE_MODEL=meta-llama/Llama-3.1-8B-Instruct
CHECKPOINT="${REPO}/Hierarchical_KV/shortq_placement/model.pt"
GNN_INFERENCE_BATCH_SIZE="${GNN_INFERENCE_BATCH_SIZE:-1}"

case "$PROFILE" in
  h100_smoke)
    CUDA_HOME=/usr/local/cuda-13.1
    MODEL=meta-llama/Llama-3.3-70B-Instruct
    TP=2; QUANTIZATION=fp8; GPU_UTIL=0.60; MAX_SEQS=16
    L0_GB=4; L1_GB=96; L2_WORKERS=8
    NUM_QUESTIONS=96; SUBMISSION_BATCH_SIZE=96; MIN_TOKENS=32; MAX_TOKENS=128
    RUN_LABEL=h100_shortq_gnn_smoke
    L2_DIR="/scratch2/sriramc2/lmcache_mp_l2/gnn_h100_smoke_${SLURM_JOB_ID}"
    SMOKE_FORCE_MIN_UNIQUE_PER_TIER="${SMOKE_FORCE_MIN_UNIQUE_PER_TIER:-2}"
    ;;
  l40_smoke)
    CUDA_HOME=/usr/local/cuda-13.0
    MODEL=meta-llama/Llama-3.1-8B-Instruct
    TP=1; QUANTIZATION=""; GPU_UTIL=0.60; MAX_SEQS=12
    L0_GB=2; L1_GB=64; L2_WORKERS=4
    NUM_QUESTIONS="${NUM_QUESTIONS:-96}"
    SUBMISSION_BATCH_SIZE="${SUBMISSION_BATCH_SIZE:-$NUM_QUESTIONS}"
    MIN_TOKENS="${MIN_TOKENS:-32}"
    MAX_TOKENS="${MAX_TOKENS:-128}"
    RUN_LABEL=l40_shortq_gnn_smoke
    L2_DIR="${RUN_ROOT}/lmcache_mp_l2/gnn_l40_smoke_${SLURM_JOB_ID}"
    SMOKE_FORCE_MIN_UNIQUE_PER_TIER="${SMOKE_FORCE_MIN_UNIQUE_PER_TIER:-2}"
    ;;
  h100_q650)
    CUDA_HOME=/usr/local/cuda-13.1
    MODEL=meta-llama/Llama-3.3-70B-Instruct
    TP=2; QUANTIZATION=fp8; GPU_UTIL=0.60; MAX_SEQS=16
    L0_GB="${L0_GB:-4}"; L1_GB=200; L2_WORKERS=8
    NUM_QUESTIONS=650; SUBMISSION_BATCH_SIZE=650; MIN_TOKENS=128; MAX_TOKENS=512
    RUN_LABEL=h100_shortq_gnn_q650
    L2_DIR="/scratch2/sriramc2/lmcache_mp_l2/gnn_h100_q650_${SLURM_JOB_ID}"
    SMOKE_FORCE_MIN_UNIQUE_PER_TIER="${SMOKE_FORCE_MIN_UNIQUE_PER_TIER:-0}"
    ;;
  l40_q200)
    CUDA_HOME=/usr/local/cuda-13.0
    MODEL=meta-llama/Llama-3.1-8B-Instruct
    TP=1
    QUANTIZATION=""
    GPU_UTIL=0.60
    MAX_SEQS=12

    L0_GB=2
    L1_GB=64
    L2_WORKERS=4

    NUM_QUESTIONS=200
    SUBMISSION_BATCH_SIZE=200
    MIN_TOKENS=32
    MAX_TOKENS=128

    RUN_LABEL=l40_shortq_gnn_q200_gpu6_cpu5
    L2_DIR="${RUN_ROOT}/lmcache_mp_l2/gnn_l40_q200_gpu6_cpu5_${SLURM_JOB_ID}"

    # Real performance experiment: never force placement tier coverage.
    SMOKE_FORCE_MIN_UNIQUE_PER_TIER=0
    ;;
  *) echo "unknown KV_GNN_PROFILE=$PROFILE" >&2; exit 2;;
esac

RUN_DIR="${RUN_ROOT}/${RUN_LABEL}_${SLURM_JOB_ID}"
LOG_DIR="${RUN_DIR}/logs"; RESULT_DIR="${RUN_DIR}/results"; PLACEMENT_DIR="${RUN_DIR}/placement"
RUNTIME_METADATA="${PLACEMENT_DIR}/runtime_hash_to_tier.json"
PLACEMENT_TRACE="${PLACEMENT_DIR}/gnn_chunk_placements.jsonl"
HASH_PREDICTION_TRACE="${PLACEMENT_DIR}/gnn_hash_prediction_occurrences.jsonl"
GNN_TIMING_TRACE="${PLACEMENT_DIR}/gnn_prediction_timing.jsonl"
PLACEMENT_SUMMARY="${PLACEMENT_DIR}/gnn_placement_summary.json"

CLEAN_OLD_LMCACHE="${CLEAN_OLD_LMCACHE:-1}"

clean_old_lmcache() {
    if [[ "$CLEAN_OLD_LMCACHE" != "1" ]]; then
        echo "Skipping old LMCache cleanup: CLEAN_OLD_LMCACHE=$CLEAN_OLD_LMCACHE"
        return
    fi

    case "$PROFILE" in
        h100_*)
            # Match the existing H100 no-GNN clean-all behavior:
            # preserve /scratch2/sriramc2 itself, delete EVERYTHING below it.
            local root="/scratch2/sriramc2"

            if [[ "$root" != "/scratch2/sriramc2" ]]; then
                echo "ERROR: refusing destructive cleanup of unexpected scratch root: $root"
                exit 12
            fi

            echo "WARNING: deleting EVERYTHING under $root"
            mkdir -p "$root"
            find "$root" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
            ;;

        l40_*)
            # L40 uses NFS. Delete all previous LMCache L2 runs,
            # but do NOT touch the rest of RUN_ROOT.
            local root="${RUN_ROOT}/lmcache_mp_l2"

            if [[ "$root" != "/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/lmcache_mp_l2" ]]; then
                echo "ERROR: refusing destructive cleanup of unexpected NFS LMCache root: $root"
                exit 13
            fi

            echo "WARNING: deleting all old LMCache data under $root"
            mkdir -p "$root"
            find "$root" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
            ;;

        *)
            echo "ERROR: no cleanup policy for PROFILE=$PROFILE"
            exit 14
            ;;
    esac
}

clean_old_lmcache

mkdir -p "$LOG_DIR" "$RESULT_DIR" "$PLACEMENT_DIR" "$L2_DIR"

cd "$REPO"
source "${VENV}/bin/activate"

GNN_CHUNK_VOTE_POLICY="$(
python - <<'PY'
from Hierarchical_KV.shortq_placement.chunk_voting import (
    selected_vote_policy_name,
)
print(selected_vote_policy_name())
PY
)"
export CUDA_HOME PATH="${CUDA_HOME}/bin:${PATH}" LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_MPS_PIPE_DIRECTORY="/tmp/kvaware-no-mps-${USER}-${SLURM_JOB_ID}-DO-NOT-CREATE"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HOME=/mnt/shared/gpfs/home/sriramc2/.cache/huggingface
export VLLM_KV_IMPORTANCE_ENABLE=0
export VLLM_KV_RECOMPUTE_DEBUG=1
export LMCACHE_KV_ACCOUNTING_DEBUG=1
export LMCACHE_PREFETCH_LIFETIME_DEBUG=1 LMCACHE_PREFETCH_LIFETIME_WARN_S=30
export LMCACHE_MP_STAGE1_FRESHNESS_GUARD_S=270
export LMCACHE_MP_CONGESTION_DEBUG=1 LMCACHE_MP_CONGESTION_LOG_EVERY=25 LMCACHE_MP_CONGESTION_SLOW_S=5
export LMCACHE_MP_CHTHM_DEBUG=1
export LMCACHE_CHUNK_SIZE
export LMCACHE_L0_SMOKE_TRACE_D2D=1
export LMCACHE_L0_SMOKE_STORE=0 LMCACHE_L0_VPC_IMITATION=0
export LMCACHE_GNN_EXCLUSIVE_PLACEMENT=1
export LMCACHE_GNN_PLACEMENT_METADATA="$RUNTIME_METADATA"

LMCACHE_PID=""; VLLM_PID=""
cleanup(){
  set +e
  for pid in "$VLLM_PID" "$LMCACHE_PID"; do
    [[ -n "$pid" ]] || continue
    kill -TERM "$pid" 2>/dev/null || true; sleep 2; kill -KILL "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

wait_tcp(){ local port=$1 pid=$2 limit=$3 start=$SECONDS; while (( SECONDS-start < limit )); do kill -0 "$pid" 2>/dev/null || return 1; python - "$port" <<'PY' >/dev/null 2>&1 && return 0
import socket,sys
s=socket.socket(); s.settimeout(.25); r=s.connect_ex(('127.0.0.1',int(sys.argv[1]))); s.close(); raise SystemExit(0 if r==0 else 1)
PY
sleep 1; done; return 1; }
wait_health(){ local pid=$1 limit=$2 start=$SECONDS; while (( SECONDS-start < limit )); do kill -0 "$pid" 2>/dev/null || return 1; curl -sf http://127.0.0.1:8000/health >/dev/null && return 0; sleep 2; done; return 1; }

cat > "${RUN_DIR}/run_config.json" <<JSON
{"profile":"$PROFILE","model":"$MODEL","feature_model":"$FEATURE_MODEL","questions":$NUM_QUESTIONS,"tp":$TP,"gpu_memory_utilization":$GPU_UTIL,"max_num_seqs":$MAX_SEQS,"l0_gb":$L0_GB,"l1_gb":$L1_GB,"chunk_size":$LMCACHE_CHUNK_SIZE,"prefetch_max_in_flight":$LMCACHE_L2_PREFETCH_MAX_IN_FLIGHT,"vpc":false,"placement":"shortq_class_logits_exclusive", "chunk_vote_policy":"${GNN_CHUNK_VOTE_POLICY}","runtime_metadata":"$RUNTIME_METADATA","hash_prediction_trace":"$HASH_PREDICTION_TRACE","gnn_timing_trace":"$GNN_TIMING_TRACE","gnn_inference_batch_size":$GNN_INFERENCE_BATCH_SIZE,"smoke_force_min_unique_per_tier":$SMOKE_FORCE_MIN_UNIQUE_PER_TIER}
JSON

echo "===== SHORT-Q PRECOMPUTE ($PROFILE) ====="
python experiments/precompute_shortq_placements.py \
  --dataset_name "$DATASET_NAME" --max_questions "$NUM_QUESTIONS" --retrieval_top_k "$RETRIEVAL_TOP_K" \
  --request_order "$REQUEST_ORDER" --prefix_sort_depth "$PREFIX_SORT_DEPTH" --request_order_seed "$REQUEST_ORDER_SEED" \
  --serving_model "$MODEL" --feature_model "$FEATURE_MODEL" --checkpoint "$CHECKPOINT" --chunk_size "$LMCACHE_CHUNK_SIZE" \
  --max_seq_len 8000 --device cuda:0 --inference_batch_size "$GNN_INFERENCE_BATCH_SIZE" --runtime_metadata "$RUNTIME_METADATA" \
  --placement_trace "$PLACEMENT_TRACE" --hash_prediction_trace "$HASH_PREDICTION_TRACE" \
  --timing_trace "$GNN_TIMING_TRACE" --summary "$PLACEMENT_SUMMARY" \
  --smoke_force_min_unique_per_tier "$SMOKE_FORCE_MIN_UNIQUE_PER_TIER" \
  > "${LOG_DIR}/gnn_precompute.log" 2>&1
python - "$RUNTIME_METADATA" "$PLACEMENT_SUMMARY" <<'PY'
import json,sys
m=json.load(open(sys.argv[1])); s=json.load(open(sys.argv[2])); assert m and all(v in {'L0','L1','L2'} for v in m.values())
print('placement_entries',len(m)); print('tier_counts',s['unique_runtime_tier_counts']); print('conflicts',s['conflicting_hashes'])
PY
# Release all feature-model CUDA state before starting the serving stack.
python - <<'PY'
import torch
if torch.cuda.is_available(): torch.cuda.empty_cache()
PY

L2_JSON=$(python - "$L2_DIR" "$L2_WORKERS" <<'PY'
import json,sys
print(json.dumps({'type':'fs_native','base_path':sys.argv[1],'num_workers':int(sys.argv[2]),'use_odirect':False}))
PY
)
echo "===== START LMCACHE ====="
lmcache server --host localhost --port 5556 --chunk-size "$LMCACHE_CHUNK_SIZE" \
  --l0-enable --l0-capacity-gb "$L0_GB" --l1-size-gb "$L1_GB" \
  --l2-prefetch-max-in-flight "$LMCACHE_L2_PREFETCH_MAX_IN_FLIGHT" --l2-store-policy gnn_exclusive \
  --eviction-policy LRU --max-workers "$LMCACHE_MAX_WORKERS" --l2-adapter "$L2_JSON" \
  > "${LOG_DIR}/lmcache.log" 2>&1 &
LMCACHE_PID=$!
wait_tcp 5556 "$LMCACHE_PID" 240 || { tail -200 "${LOG_DIR}/lmcache.log"; exit 30; }
curl -sf http://127.0.0.1:8080/metrics > "${RUN_DIR}/lmcache_metrics_before.txt" || true

KV_CONFIG=$(python - "$LMCACHE_MP_TIMEOUT" <<'PY'
import json,sys
print(json.dumps({'kv_connector':'LMCacheMPConnector','kv_connector_module_path':'lmcache.integration.vllm.lmcache_mp_connector','kv_role':'kv_both','kv_load_failure_policy':'recompute','kv_connector_extra_config':{'lmcache.mp.host':'tcp://localhost','lmcache.mp.port':5556,'lmcache.mp.mq_timeout':float(sys.argv[1]),'lmcache.mp.lookup_timeout':0.0,'lmcache.mp.starvation_fallback':True}}))
PY
)
VARGS=(serve "$MODEL" --host 127.0.0.1 --port 8000 --dtype bfloat16 --tensor-parallel-size "$TP" --max-model-len 8000 --gpu-memory-utilization "$GPU_UTIL" --max-num-seqs "$MAX_SEQS" --enable-chunked-prefill --disable-hybrid-kv-cache-manager --no-enable-prefix-caching --kv-transfer-config "$KV_CONFIG")
[[ -z "$QUANTIZATION" ]] || VARGS+=(--quantization "$QUANTIZATION")
echo "===== START VLLM ====="
vllm "${VARGS[@]}" > "${LOG_DIR}/vllm.log" 2>&1 &
VLLM_PID=$!
wait_health "$VLLM_PID" 1500 || { tail -300 "${LOG_DIR}/vllm.log"; exit 40; }
curl -sf http://127.0.0.1:8000/metrics > "${RUN_DIR}/metrics_before.txt"

DARGS=(--dataset_name "$DATASET_NAME" --max_questions "$NUM_QUESTIONS" --retrieval_top_k "$RETRIEVAL_TOP_K" --llm_model "$MODEL" --request_order "$REQUEST_ORDER" --prefix_sort_depth "$PREFIX_SORT_DEPTH" --request_order_seed "$REQUEST_ORDER_SEED" --warm_order "$WARM_ORDER" --submission_batch_size "$SUBMISSION_BATCH_SIZE" --server_url http://127.0.0.1:8000 --request_timeout_s 7200 --temperature 0 --top_p 0.95 --min_tokens "$MIN_TOKENS" --max_tokens "$MAX_TOKENS" --metrics_after_cold_path "${RUN_DIR}/metrics_after_cold.txt" --output_dir "$RESULT_DIR")
echo "===== COLD + WARM WORKLOAD ====="
CUDA_VISIBLE_DEVICES="" python experiments/mp_nognn_project.py "${DARGS[@]}" > "${LOG_DIR}/driver.log" 2>&1 || { tail -300 "${LOG_DIR}/driver.log"; exit 50; }
sleep 5
curl -sf http://127.0.0.1:8000/metrics > "${RUN_DIR}/metrics_after.txt" || true
curl -sf http://127.0.0.1:8080/metrics > "${RUN_DIR}/lmcache_metrics_after.txt" || true

python scripts/analyze_mp_congestion_chthm.py --lmcache-log "${LOG_DIR}/lmcache.log" --vllm-log "${LOG_DIR}/vllm.log" --results-dir "$RESULT_DIR" --output "${RUN_DIR}/mp_diag_summary.json" > "${RUN_DIR}/mp_diag_summary.txt" 2>&1 || true
python scripts/analyze_vllm_phase_metrics.py --before "${RUN_DIR}/metrics_before.txt" --after-cold "${RUN_DIR}/metrics_after_cold.txt" --after-warm "${RUN_DIR}/metrics_after.txt" --output "${RUN_DIR}/vllm_phase_metrics.json" > "${RUN_DIR}/vllm_phase_metrics.txt" 2>&1 || true
python scripts/analyze_gnn_placement_run.py --precompute-summary "$PLACEMENT_SUMMARY" --hash-prediction-trace "$HASH_PREDICTION_TRACE" --lmcache-log "${LOG_DIR}/lmcache.log" --lmcache-metrics "${RUN_DIR}/lmcache_metrics_after.txt" --l2-dir "$L2_DIR" --output "${RUN_DIR}/gnn_runtime_summary.json" > "${RUN_DIR}/gnn_runtime_summary.txt" 2>&1 || true

python - "$RESULT_DIR" "$PLACEMENT_SUMMARY" "$LOG_DIR" "$PROFILE" <<'PY'
import json,sys,pathlib,re
r=pathlib.Path(sys.argv[1]); p=json.load(open(sys.argv[2])); logs=pathlib.Path(sys.argv[3]); profile=sys.argv[4]
s=json.load(open(r/'summary.json')); assert s['cold']['successful']==s['cold']['requests']; assert s['warm']['successful']==s['warm']['requests']
lm=(logs/'lmcache.log').read_text(errors='replace'); vl=(logs/'vllm.log').read_text(errors='replace')
assert '[GNN_EXCLUSIVE_STORE]' in lm, 'no GNN exclusive store marker'; assert '[MP_CHTHM_RAW]' in lm, 'no raw CHTHM marker'; assert '[MP_CHTHM_ADMIT]' in vl, 'no useful CHTHM marker'
assert '[GNN_PLACEMENT_METADATA_MISS]' not in lm, 'prompt placement metadata miss detected; GNN benchmark is not clean'
for tier,n in p['unique_runtime_tier_counts'].items():
    if n: print('runtime_tier_hashes',tier,n)
if profile.endswith('_smoke'):
    counts=p['unique_runtime_tier_counts']
    assert all(counts.get(t,0) >= 2 for t in ('L0','L1','L2')), f'smoke runtime tier coverage missing: {counts}'
    # Smoke overrides are allowed only to ensure the mechanics are actually exercised.
    # Require evidence that each physical path was reached, not merely present in metadata.
    assert re.search(r'\[GNN_EXCLUSIVE_STORE\].*target_L0=[1-9]\d*.*L0_reserved=[1-9]\d*', lm), 'smoke never physically stored an L0 target'
    assert re.search(r'\[GNN_EXCLUSIVE_STORE\].*target_L1=[1-9]\d*.*L1_stage_reserved=[1-9]\d*', lm), 'smoke never physically staged/persisted an L1 target'
    assert re.search(r'\[GNN_EXCLUSIVE_STORE\].*target_L2=[1-9]\d*.*L1_stage_reserved=[1-9]\d*', lm), 'smoke never physically staged an L2 target'
    assert '[GNN_EXCLUSIVE_L2_COMMIT]' in lm, 'smoke never completed an L2 commit'
    print('GNN_EXCLUSIVE_SMOKE_TIER_PATHS_PASS')
print('GNN_EXCLUSIVE_RUN_PASS')
PY

echo "===== RESULTS ====="
cat "${RESULT_DIR}/summary.json"
echo "placement:"; cat "$PLACEMENT_SUMMARY"
echo "run_dir=$RUN_DIR"
