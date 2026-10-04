#!/usr/bin/env bash
#SBATCH --job-name=kvaware-l40-gnn-q200-disk
#SBATCH --partition=gpu-l40-dchen
#SBATCH --nodelist=codenimbus-002-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/shared/gpfs/home/sriramc2/runs/kvaware/%x_%j.out
#SBATCH --error=/mnt/shared/gpfs/home/sriramc2/runs/kvaware/%x_%j.err

set -Eeuo pipefail

REPO=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
VENV=/mnt/shared/gpfs/home/sriramc2/venvs/kvaware-vllm029
CUDA_HOME=/usr/local/cuda-13.0
RUN_ROOT=/mnt/shared/gpfs/home/sriramc2/runs/kvaware
FEATURE_MODEL=meta-llama/Llama-3.1-8B-Instruct
CHECKPOINT="${REPO}/Hierarchical_KV/shortq_placement/model.pt"

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
TP=1
GPU_UTIL="${GPU_MEMORY_UTILIZATION:-0.60}"
MAX_SEQS="${MAX_NUM_SEQS:-12}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8000}"
LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-512}"
L1_GB="${LMCACHE_L1_SIZE_GB:-64}"
LOOKUP_PF="${LMCACHE_L2_LOOKUP_MAX_IN_FLIGHT:-16}"
GET_PF="${LMCACHE_L2_LOAD_MAX_IN_FLIGHT:-2}"
# 0 = unlimited/disabled rolling cap: intentional disk-stress characterization.
STORE_CAP_GB="${LMCACHE_MP_MAX_INFLIGHT_STORE_GB:-0}"
LMCACHE_MAX_WORKERS="${LMCACHE_MAX_WORKERS:-8}"
MQ_TIMEOUT="${LMCACHE_MP_TIMEOUT:-120}"
HEARTBEAT="${LMCACHE_MP_HEARTBEAT_INTERVAL:-40}"
GNN_INFERENCE_BATCH_SIZE="${GNN_INFERENCE_BATCH_SIZE:-1}"

DATASET_NAME="${DATASET_NAME:-medical}"
NUM_QUESTIONS="${NUM_QUESTIONS:-200}"
RETRIEVAL_TOP_K="${RETRIEVAL_TOP_K:-5}"
REQUEST_ORDER="${KV_REQUEST_ORDER:-seeded_shuffle}"
REQUEST_ORDER_SEED="${KV_REQUEST_ORDER_SEED:-0}"
WARM_ORDER="${WARM_ORDER:-random}"
WARM_ORDER_SEED="${WARM_ORDER_SEED:-1}"
SUBMISSION_BATCH_SIZE="${SUBMISSION_BATCH_SIZE:-200}"
MIN_TOKENS="${MIN_TOKENS:-128}"
MAX_TOKENS="${MAX_TOKENS:-128}"

KV_GNN_AWARE_VPC="${KV_GNN_AWARE_VPC:-1}"
VLLM_GNN_AWARE_VPC_WINDOW="${VLLM_GNN_AWARE_VPC_WINDOW:-256}"
KV_GNN_L1_BACKING="${KV_GNN_L1_BACKING:-1}"
KV_GNN_L2_BACKING="${KV_GNN_L2_BACKING:-0}"
KV_VPC_SUFFICIENT_BYPASS="${KV_VPC_SUFFICIENT_BYPASS:-1}"
KV_MUTUAL_PREFIX="${KV_MUTUAL_PREFIX:-1}"
KV_STAGE1_STARVATION_FALLBACK="${KV_STAGE1_STARVATION_FALLBACK:-1}"
KV_COLD_WARM_STORE_BARRIER="${KV_COLD_WARM_STORE_BARRIER:-1}"
KV_STORE_BARRIER_TIMEOUT_S="${KV_STORE_BARRIER_TIMEOUT_S:-7200}"
KV_STORE_BARRIER_POLL_S="${KV_STORE_BARRIER_POLL_S:-1}"
KV_STORE_BARRIER_STABLE_POLLS="${KV_STORE_BARRIER_STABLE_POLLS:-5}"
KV_STORE_BARRIER_STALL_S="${KV_STORE_BARRIER_STALL_S:-120}"

KV_FS_SHARED_WORKERS="${KV_FS_SHARED_WORKERS:-5}"
KV_FS_LOOKUP_WORKERS="${KV_FS_LOOKUP_WORKERS:-1}"
KV_FS_RETRIEVE_WORKERS="${KV_FS_RETRIEVE_WORKERS:-2}"
KV_FS_STORE_WORKERS="${KV_FS_STORE_WORKERS:-0}"
KV_FS_DELETE_WORKERS="${KV_FS_DELETE_WORKERS:-0}"

export LMCACHE_MP_STAGE1_FRESHNESS_GUARD_S="${LMCACHE_MP_STAGE1_FRESHNESS_GUARD_S:-270}"
export LMCACHE_MP_CONGESTION_DEBUG="${LMCACHE_MP_CONGESTION_DEBUG:-1}"
export LMCACHE_MP_CONGESTION_LOG_EVERY="${LMCACHE_MP_CONGESTION_LOG_EVERY:-25}"
export LMCACHE_MP_CONGESTION_SLOW_S="${LMCACHE_MP_CONGESTION_SLOW_S:-5}"
export LMCACHE_MP_CHTHM_DEBUG="${LMCACHE_MP_CHTHM_DEBUG:-1}"
export LMCACHE_MP_PREFIX_DIAGNOSTICS="${LMCACHE_MP_PREFIX_DIAGNOSTICS:-1}"
export VLLM_KV_RECOMPUTE_DEBUG=1 LMCACHE_KV_ACCOUNTING_DEBUG=1
export LMCACHE_PREFETCH_LIFETIME_DEBUG=1 LMCACHE_PREFETCH_LIFETIME_WARN_S=30

EXPERIMENT_LABEL="${EXPERIMENT_LABEL:-diskstress_uncapped_mns12_fixed128_rand0_rand1}"
RUN_DIR="${RUN_ROOT}/l40_shortq_gnn_dynamic_vpc_q200_${EXPERIMENT_LABEL}_${SLURM_JOB_ID}"
LOG_DIR="$RUN_DIR/logs"; RESULT_DIR="$RUN_DIR/results"; PLACEMENT_DIR="$RUN_DIR/placement"
RUNTIME_METADATA="$PLACEMENT_DIR/runtime_hash_to_placement.json"
VPC_IMPORTANCE_SIDECAR="$PLACEMENT_DIR/vpc_importance_by_request.json"
PLACEMENT_TRACE="$PLACEMENT_DIR/gnn_chunk_placements.jsonl"
HASH_PREDICTION_TRACE="$PLACEMENT_DIR/gnn_hash_prediction_occurrences.jsonl"
GNN_TIMING_TRACE="$PLACEMENT_DIR/gnn_prediction_timing.jsonl"
PLACEMENT_SUMMARY="$PLACEMENT_DIR/gnn_placement_summary.json"

LMCACHE_L2_ROOT="${LMCACHE_L2_ROOT:-/mnt/shared/gpfs/home/sriramc2/runs/kvaware/lmcache_mp_l2_l40}"
L2_DIR="$LMCACHE_L2_ROOT/gnn_q200_${SLURM_JOB_ID}"
CLEAN_OLD_LMCACHE="${CLEAN_OLD_LMCACHE:-1}"

mkdir -p "$LOG_DIR" "$RESULT_DIR" "$PLACEMENT_DIR"
cd "$REPO"
source "$VENV/bin/activate"
export CUDA_HOME PATH="${CUDA_HOME}/bin:${PATH}" LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_MPS_PIPE_DIRECTORY="/tmp/kvaware-no-mps-${USER}-${SLURM_JOB_ID}-DO-NOT-CREATE"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/mnt/shared/gpfs/home/sriramc2/.cache/huggingface}"

export VLLM_GNN_AWARE_VPC="$KV_GNN_AWARE_VPC"
export VLLM_GNN_AWARE_VPC_WINDOW
export VLLM_KV_IMPORTANCE_PLACEMENTS="$VPC_IMPORTANCE_SIDECAR"
export LMCACHE_GNN_DYNAMIC_STORE=1
export LMCACHE_GNN_L1_BACKING="$KV_GNN_L1_BACKING"
export LMCACHE_GNN_L2_BACKING="$KV_GNN_L2_BACKING"
export LMCACHE_GNN_PLACEMENT_METADATA="$RUNTIME_METADATA"
export LMCACHE_CHUNK_SIZE

LMCACHE_PID=""; VLLM_PID=""
cleanup() {
  set +e
  for pid in "$VLLM_PID" "$LMCACHE_PID"; do
    [[ -n "$pid" ]] || continue
    kill -TERM "$pid" 2>/dev/null || true; sleep 2; kill -KILL "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT
wait_tcp() {
  local port=$1 pid=$2 limit=$3 start=$SECONDS
  while (( SECONDS-start < limit )); do
    kill -0 "$pid" 2>/dev/null || return 1
    python - "$port" <<'PY' >/dev/null 2>&1 && return 0
import socket,sys
s=socket.socket(); s.settimeout(.25); rc=s.connect_ex(('127.0.0.1',int(sys.argv[1]))); s.close(); raise SystemExit(0 if rc==0 else 1)
PY
    sleep 1
  done; return 1
}
wait_health() {
  local pid=$1 limit=$2 start=$SECONDS
  while (( SECONDS-start < limit )); do
    kill -0 "$pid" 2>/dev/null || return 1
    curl -sf http://127.0.0.1:8000/health >/dev/null && return 0
    sleep 2
  done; return 1
}

if [[ "$CLEAN_OLD_LMCACHE" == "1" ]]; then
  [[ "$LMCACHE_L2_ROOT" == "/mnt/shared/gpfs/home/sriramc2/runs/kvaware/lmcache_mp_l2_l40" ]] || {
    echo "ERROR: refusing to clean unexpected NFS cache root: $LMCACHE_L2_ROOT" >&2; exit 12;
  }
  mkdir -p "$LMCACHE_L2_ROOT"; find "$LMCACHE_L2_ROOT" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
fi
mkdir -p "$L2_DIR"

for n in KV_GNN_AWARE_VPC KV_GNN_L1_BACKING KV_GNN_L2_BACKING KV_VPC_SUFFICIENT_BYPASS KV_MUTUAL_PREFIX KV_STAGE1_STARVATION_FALLBACK KV_COLD_WARM_STORE_BARRIER; do
  v="${!n}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "ERROR $n=$v"; exit 2; }
done

echo "===== L40 GNN DYNAMIC-VPC Q200 DISK-STRESS ====="
echo "node=$(hostname) model=$MODEL TP=$TP GPU_UTIL=$GPU_UTIL MNS=$MAX_SEQS"
echo "NFS_L2=$L2_DIR U=$STORE_CAP_GB LOOKUP_PF=$LOOKUP_PF GET_PF=$GET_PF"
echo "GNN_VPC=$KV_GNN_AWARE_VPC W=$VLLM_GNN_AWARE_VPC_WINDOW L1_BACK=$KV_GNN_L1_BACKING L2_FULL_BACK=$KV_GNN_L2_BACKING"
findmnt "$LMCACHE_L2_ROOT" || true; df -hT "$LMCACHE_L2_ROOT" || true
nvidia-smi --query-gpu=index,name,memory.total,memory.free,compute_cap --format=csv,noheader || true
if nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -v 'nvidia-cuda-mps-server' | grep -q '[0-9]'; then
  echo "ERROR: pre-existing non-MPS CUDA compute client visible" >&2; nvidia-smi || true; exit 42
fi

KV_BYTES_PER_TOKEN=$(python - "$MODEL" <<'PY'
from transformers import AutoConfig
import sys
c=AutoConfig.from_pretrained(sys.argv[1]); hd=getattr(c,'head_dim',None) or c.hidden_size//c.num_attention_heads
print(int(c.num_hidden_layers*c.num_key_value_heads*hd*2*2))
PY
)

GNN_CHUNK_VOTE_CONFIG_JSON=$(python - <<'PY'
import json
from dataclasses import asdict
from Hierarchical_KV.shortq_placement.chunk_voting import selected_vote_config, selected_vote_policy_name
print(json.dumps({'policy':selected_vote_policy_name(),'config':asdict(selected_vote_config())},sort_keys=True))
PY
)

cat > "$RUN_DIR/run_config.json" <<JSON
{
  "profile":"l40_gnn_dynamic_vpc_q200_diskstress","model":"$MODEL","feature_model":"$FEATURE_MODEL","questions":$NUM_QUESTIONS,
  "tp":1,"gpu_memory_utilization":$GPU_UTIL,"max_num_seqs":$MAX_SEQS,"l1_gb":$L1_GB,"chunk_size":$LMCACHE_CHUNK_SIZE,
  "load_max_in_flight":$GET_PF,"lookup_max_in_flight":$LOOKUP_PF,"store_max_inflight_gb":$STORE_CAP_GB,
  "request_order":"$REQUEST_ORDER","request_order_seed":$REQUEST_ORDER_SEED,"warm_order":"$WARM_ORDER","warm_order_seed":$WARM_ORDER_SEED,
  "submission_batch_size":$SUBMISSION_BATCH_SIZE,"min_tokens":$MIN_TOKENS,"max_tokens":$MAX_TOKENS,"kv_bytes_per_token":$KV_BYTES_PER_TOKEN,
  "vpc":true,"gnn_aware_vpc":$KV_GNN_AWARE_VPC,"gnn_aware_vpc_window":$VLLM_GNN_AWARE_VPC_WINDOW,
  "gnn_l1_backing":$KV_GNN_L1_BACKING,"gnn_l2_backing":$KV_GNN_L2_BACKING,
  "vpc_sufficient_bypass":$KV_VPC_SUFFICIENT_BYPASS,"mutual_prefix":$KV_MUTUAL_PREFIX,"stage1_starvation_fallback":$KV_STAGE1_STARVATION_FALLBACK,
  "cold_warm_store_barrier":$KV_COLD_WARM_STORE_BARRIER,"placement":"shortq_dynamic_vpc_plus_lower_backing",
  "chunk_vote":$GNN_CHUNK_VOTE_CONFIG_JSON,"l2_dir":"$L2_DIR"
}
JSON

echo "===== SHORT-Q PRECOMPUTE ====="
python experiments/precompute_shortq_placements.py \
  --dataset_name "$DATASET_NAME" --max_questions "$NUM_QUESTIONS" --retrieval_top_k "$RETRIEVAL_TOP_K" \
  --request_order "$REQUEST_ORDER" --prefix_sort_depth 2 --request_order_seed "$REQUEST_ORDER_SEED" \
  --serving_model "$MODEL" --feature_model "$FEATURE_MODEL" --checkpoint "$CHECKPOINT" \
  --chunk_size "$LMCACHE_CHUNK_SIZE" --max_seq_len "$MAX_MODEL_LEN" --device cuda:0 --inference_batch_size "$GNN_INFERENCE_BATCH_SIZE" \
  --runtime_metadata "$RUNTIME_METADATA" --vpc_importance_sidecar "$VPC_IMPORTANCE_SIDECAR" \
  --placement_trace "$PLACEMENT_TRACE" --hash_prediction_trace "$HASH_PREDICTION_TRACE" --timing_trace "$GNN_TIMING_TRACE" \
  --summary "$PLACEMENT_SUMMARY" --smoke_force_min_unique_per_placement 0 > "$LOG_DIR/gnn_precompute.log" 2>&1

python - "$RUNTIME_METADATA" "$VPC_IMPORTANCE_SIDECAR" "$PLACEMENT_SUMMARY" <<'PY'
import json,sys
r=json.load(open(sys.argv[1])); s=json.load(open(sys.argv[2])); p=json.load(open(sys.argv[3]))
assert r and all(v in {'gpu','cpu','disk'} for v in r.values()); assert len(s)==p['requests']
print('placement_entries',len(r)); print('placement_counts',p['unique_runtime_placement_counts'])
PY

L2_JSON=$(python - "$L2_DIR" "$KV_FS_SHARED_WORKERS" "$KV_FS_LOOKUP_WORKERS" "$KV_FS_RETRIEVE_WORKERS" "$KV_FS_STORE_WORKERS" "$KV_FS_DELETE_WORKERS" <<'PY'
import json,sys
spec={'type':'fs_native','base_path':sys.argv[1],'num_workers':int(sys.argv[2]),'use_odirect':False}
names=('lookup','retrieve','store','delete'); counts=map(int,sys.argv[3:7]); per={n:c for n,c in zip(names,counts) if c>0}
if per: spec['per_op_workers']=per
print(json.dumps(spec))
PY
)

echo "===== START LMCACHE ====="
lmcache server --host localhost --port 5556 --chunk-size "$LMCACHE_CHUNK_SIZE" --l1-size-gb "$L1_GB" \
  --l2-load-max-in-flight "$GET_PF" --l2-lookup-max-in-flight "$LOOKUP_PF" --l2-store-max-inflight-gb "$STORE_CAP_GB" \
  --l2-store-policy gnn_dynamic --eviction-policy LRU --max-workers "$LMCACHE_MAX_WORKERS" --l2-adapter "$L2_JSON" \
  > "$LOG_DIR/lmcache.log" 2>&1 &
LMCACHE_PID=$!
wait_tcp 5556 "$LMCACHE_PID" 240 || { tail -200 "$LOG_DIR/lmcache.log"; exit 30; }
curl -sf http://127.0.0.1:8080/metrics > "$RUN_DIR/lmcache_metrics_before.txt" || true

KV_CONFIG=$(python - "$MQ_TIMEOUT" "$HEARTBEAT" "$KV_STAGE1_STARVATION_FALLBACK" "$KV_VPC_SUFFICIENT_BYPASS" "$KV_MUTUAL_PREFIX" <<'PY'
import json,sys
print(json.dumps({'kv_connector':'LMCacheMPConnector','kv_connector_module_path':'lmcache.integration.vllm.lmcache_mp_connector','kv_role':'kv_both','kv_load_failure_policy':'recompute','kv_connector_extra_config':{'lmcache.mp.host':'tcp://localhost','lmcache.mp.port':5556,'lmcache.mp.mq_timeout':float(sys.argv[1]),'lmcache.mp.heartbeat_interval':float(sys.argv[2]),'lmcache.mp.lookup_timeout':0.0,'lmcache.mp.starvation_fallback':bool(int(sys.argv[3])),'lmcache.mp.vpc_sufficient_bypass':bool(int(sys.argv[4])),'lmcache.mp.mutual_prefix':bool(int(sys.argv[5]))}}))
PY
)

echo "===== START VLLM ====="
vllm serve "$MODEL" --host 127.0.0.1 --port 8000 --dtype bfloat16 --tensor-parallel-size 1 \
  --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_UTIL" --max-num-seqs "$MAX_SEQS" \
  --enable-chunked-prefill --disable-hybrid-kv-cache-manager --enable-prefix-caching --kv-transfer-config "$KV_CONFIG" \
  > "$LOG_DIR/vllm.log" 2>&1 &
VLLM_PID=$!
wait_health "$VLLM_PID" 1500 || { tail -300 "$LOG_DIR/vllm.log"; exit 40; }
curl -sf http://127.0.0.1:8000/metrics > "$RUN_DIR/metrics_before.txt" || true

DARGS=(--dataset_name "$DATASET_NAME" --max_questions "$NUM_QUESTIONS" --retrieval_top_k "$RETRIEVAL_TOP_K" --llm_model "$MODEL"
  --request_order "$REQUEST_ORDER" --request_order_seed "$REQUEST_ORDER_SEED" --warm_order "$WARM_ORDER" --warm_order_seed "$WARM_ORDER_SEED"
  --submission_batch_size "$SUBMISSION_BATCH_SIZE" --server_url http://127.0.0.1:8000 --request_timeout_s 7200
  --temperature 0 --top_p 0.95 --min_tokens "$MIN_TOKENS" --max_tokens "$MAX_TOKENS"
  --metrics_after_cold_path "$RUN_DIR/metrics_after_cold.txt" --deterministic_request_ids --output_dir "$RESULT_DIR")
if [[ "$KV_COLD_WARM_STORE_BARRIER" == "1" ]]; then
  DARGS+=(--between_phases_store_barrier --lmcache_status_url http://127.0.0.1:8080/status \
    --store_barrier_timeout_s "$KV_STORE_BARRIER_TIMEOUT_S" --store_barrier_poll_s "$KV_STORE_BARRIER_POLL_S" \
    --store_barrier_stable_polls "$KV_STORE_BARRIER_STABLE_POLLS" --store_barrier_stall_s "$KV_STORE_BARRIER_STALL_S" \
    --store_barrier_result_path "$RUN_DIR/store_barrier.json" --metrics_after_barrier_path "$RUN_DIR/metrics_after_barrier.txt")
fi

echo "===== COLD + WARM ====="
CUDA_VISIBLE_DEVICES="" python experiments/mp_nognn_project.py "${DARGS[@]}" > "$LOG_DIR/driver.log" 2>&1 || { tail -300 "$LOG_DIR/driver.log"; exit 50; }
sleep 5
curl -sf http://127.0.0.1:8000/metrics > "$RUN_DIR/metrics_after.txt" || true
curl -sf http://127.0.0.1:8080/metrics > "$RUN_DIR/lmcache_metrics_after.txt" || true

python scripts/analyze_mp_congestion_chthm.py --lmcache-log "$LOG_DIR/lmcache.log" --vllm-log "$LOG_DIR/vllm.log" \
  --results-dir "$RESULT_DIR" --output "$RUN_DIR/mp_diag_summary.json" > "$RUN_DIR/mp_diag_summary.txt" 2>&1 || true
python scripts/analyze_vllm_phase_metrics.py --before "$RUN_DIR/metrics_before.txt" --after-cold "$RUN_DIR/metrics_after_cold.txt" \
  --after-warm "$RUN_DIR/metrics_after.txt" --output "$RUN_DIR/vllm_phase_metrics.json" > "$RUN_DIR/vllm_phase_metrics.txt" 2>&1 || true
if [[ -f scripts/analyze_l2_store_bandwidth.py ]]; then
  python scripts/analyze_l2_store_bandwidth.py --lmcache-log "$LOG_DIR/lmcache.log" --summary "$RESULT_DIR/summary.json" \
    --kv-bytes-per-token "$KV_BYTES_PER_TOKEN" --output-json "$RUN_DIR/l2_store_bandwidth.json" --output-txt "$RUN_DIR/l2_store_bandwidth.txt" || true
fi

python - "$RESULT_DIR/summary.json" "$PLACEMENT_SUMMARY" "$LOG_DIR/lmcache.log" "$LOG_DIR/vllm.log" "$KV_GNN_L2_BACKING" <<'PY'
import json,re,sys
summary=json.load(open(sys.argv[1])); placement=json.load(open(sys.argv[2])); lm=open(sys.argv[3],errors='replace').read(); vl=open(sys.argv[4],errors='replace').read()
assert summary['cold']['successful']==summary['cold']['requests']; assert summary['warm']['successful']==summary['warm']['requests']
assert '[GNN_DYNAMIC_STORE]' in lm and '[GNN_DYNAMIC_L2_POLICY]' in lm
assert '[GNN_PLACEMENT_METADATA_MISS]' not in lm
assert '[GNN_AWARE_VPC_INIT]' in vl
if sys.argv[5]=='0':
    assert '[GNN_DYNAMIC_L2_BACKING_INIT] enabled=True' not in lm
print('GNN_DYNAMIC_VPC_RUN_PASS',placement['unique_runtime_placement_counts'])
PY

{
  echo "l2_root=$LMCACHE_L2_ROOT"; echo "l2_dir=$L2_DIR"
  echo -n "file_count="; find "$L2_DIR" -type f 2>/dev/null | wc -l
  echo -n "disk_bytes="; du -sb "$L2_DIR" 2>/dev/null | awk '{print $1}'
} > "$RUN_DIR/l2_filesystem_summary.txt"

cat "$RESULT_DIR/summary.json"; echo "placement:"; cat "$PLACEMENT_SUMMARY"
[[ -f "$RUN_DIR/l2_store_bandwidth.txt" ]] && cat "$RUN_DIR/l2_store_bandwidth.txt" || true
echo "run_dir=$RUN_DIR"
