#!/usr/bin/env bash
set -Eeuo pipefail

PROFILE="${KV_GNN_DYNAMIC_PROFILE:?wrapper must set KV_GNN_DYNAMIC_PROFILE}"
REPO=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
VENV=/mnt/shared/gpfs/home/sriramc2/venvs/kvaware-vllm029
RUN_ROOT=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro
DATASET_NAME=medical
RETRIEVAL_TOP_K=5
REQUEST_ORDER=legacy_prefix_hash
PREFIX_SORT_DEPTH=2
REQUEST_ORDER_SEED=0
WARM_ORDER="${WARM_ORDER:-reverse}"
LMCACHE_CHUNK_SIZE=512
# Separate admission for fast lookup/discovery and slow L2 loads.  The old
# LMCACHE_L2_PREFETCH_MAX_IN_FLIGHT name is accepted here as a launcher-only
# compatibility alias; LMCache itself now uses explicit load/lookup names.
LMCACHE_L2_LOAD_MAX_IN_FLIGHT="${LMCACHE_L2_LOAD_MAX_IN_FLIGHT:-${LMCACHE_L2_PREFETCH_MAX_IN_FLIGHT:-2}}"
LMCACHE_L2_LOOKUP_MAX_IN_FLIGHT="${LMCACHE_L2_LOOKUP_MAX_IN_FLIGHT:-16}"
LMCACHE_MP_MAX_INFLIGHT_STORE_GB="${LMCACHE_MP_MAX_INFLIGHT_STORE_GB:-0}"
LMCACHE_MAX_WORKERS="${LMCACHE_MAX_WORKERS:-8}"
LMCACHE_MP_TIMEOUT="${LMCACHE_MP_TIMEOUT:-120}"
LMCACHE_MP_HEARTBEAT_INTERVAL="${LMCACHE_MP_HEARTBEAT_INTERVAL:-40}"
FEATURE_MODEL=meta-llama/Llama-3.1-8B-Instruct
CHECKPOINT="${REPO}/Hierarchical_KV/shortq_placement/model.pt"
GNN_INFERENCE_BATCH_SIZE="${GNN_INFERENCE_BATCH_SIZE:-1}"

# Independent ablation knobs. Standard vLLM prefix caching remains ON even when
# KV_GNN_AWARE_VPC=0; only the learned retention bias is disabled.
KV_GNN_AWARE_VPC="${KV_GNN_AWARE_VPC:-1}"
KV_GNN_L1_BACKING="${KV_GNN_L1_BACKING:-1}"
LMCACHE_L1_SIZE_GB="${LMCACHE_L1_SIZE_GB:-200}"
# Optional durability experiment: back every host-resident GNN prompt chunk to
# L2 while preserving its original VPC/L1 placement. Disabled by default so
# existing grid runs are unchanged.
KV_GNN_L2_BACKING="${KV_GNN_L2_BACKING:-0}"
VLLM_GNN_AWARE_VPC_WINDOW="${VLLM_GNN_AWARE_VPC_WINDOW:-256}"

# Optional true cold->warm persistence barrier. When enabled, the driver waits
# for LMCache StoreController + L1 writes to remain fully drained before warm.
KV_COLD_WARM_STORE_BARRIER="${KV_COLD_WARM_STORE_BARRIER:-0}"
KV_STORE_BARRIER_TIMEOUT_S="${KV_STORE_BARRIER_TIMEOUT_S:-7200}"
KV_STORE_BARRIER_POLL_S="${KV_STORE_BARRIER_POLL_S:-1}"
KV_STORE_BARRIER_STABLE_POLLS="${KV_STORE_BARRIER_STABLE_POLLS:-5}"

# Current scheduler/cache controls.  The retired occupancy/GPU-utilization
# fallback is intentionally absent; the remaining Stage-1 fallback is the
# validated two-consecutive-full-scan rule.
KV_VPC_SUFFICIENT_BYPASS="${KV_VPC_SUFFICIENT_BYPASS:-0}"
KV_MUTUAL_PREFIX="${KV_MUTUAL_PREFIX:-0}"
KV_STAGE1_STARVATION_FALLBACK="${KV_STAGE1_STARVATION_FALLBACK:-1}"

# Dedicated worker counts are self-enabling: a zero count means that operation
# uses the shared pool.  Selected production topology is 5 shared + 1 lookup
# + 2 retrieve workers.
KV_FS_SHARED_WORKERS="${KV_FS_SHARED_WORKERS:-5}"
KV_FS_LOOKUP_WORKERS="${KV_FS_LOOKUP_WORKERS:-1}"
KV_FS_RETRIEVE_WORKERS="${KV_FS_RETRIEVE_WORKERS:-2}"
KV_FS_STORE_WORKERS="${KV_FS_STORE_WORKERS:-0}"
KV_FS_DELETE_WORKERS="${KV_FS_DELETE_WORKERS:-0}"

for name in KV_GNN_AWARE_VPC KV_GNN_L1_BACKING KV_GNN_L2_BACKING \
  KV_COLD_WARM_STORE_BARRIER KV_VPC_SUFFICIENT_BYPASS KV_MUTUAL_PREFIX \
  KV_STAGE1_STARVATION_FALLBACK; do
  value="${!name}"
  [[ "$value" == "0" || "$value" == "1" ]] || {
    echo "ERROR: $name must be 0 or 1, got $value" >&2
    exit 2
  }
done

"$VENV/bin/python" - "$LMCACHE_L2_LOAD_MAX_IN_FLIGHT" \
  "$LMCACHE_L2_LOOKUP_MAX_IN_FLIGHT" "$LMCACHE_MP_MAX_INFLIGHT_STORE_GB" \
  "$KV_FS_SHARED_WORKERS" "$KV_FS_LOOKUP_WORKERS" "$KV_FS_RETRIEVE_WORKERS" \
  "$KV_FS_STORE_WORKERS" "$KV_FS_DELETE_WORKERS" <<'PY'
import sys
load_pf, lookup_pf = map(int, sys.argv[1:3])
store_cap_gb = float(sys.argv[3])
shared = int(sys.argv[4])
workers = list(map(int, sys.argv[5:]))
assert load_pf > 0 and lookup_pf > 0, (load_pf, lookup_pf)
assert store_cap_gb >= 0, store_cap_gb
assert shared > 0, shared
assert all(x >= 0 for x in workers), workers
PY

case "$PROFILE" in
  h100_smoke)
    CUDA_HOME=/usr/local/cuda-13.1
    MODEL=meta-llama/Llama-3.3-70B-Instruct
    TP=2; QUANTIZATION=fp8; GPU_UTIL=0.65; MAX_SEQS="${MAX_NUM_SEQS:-16}"
    L1_GB="$LMCACHE_L1_SIZE_GB"
    NUM_QUESTIONS="${NUM_QUESTIONS:-96}"
    SUBMISSION_BATCH_SIZE="${SUBMISSION_BATCH_SIZE:-$NUM_QUESTIONS}"
    MIN_TOKENS="${MIN_TOKENS:-32}"; MAX_TOKENS="${MAX_TOKENS:-128}"
    RUN_LABEL=h100_shortq_gnn_dynamic_vpc_smoke
    L2_DIR="/scratch2/sriramc2/lmcache_mp_l2/gnn_dynamic_vpc_smoke_${SLURM_JOB_ID}"
    SMOKE_FORCE_MIN_UNIQUE_PER_PLACEMENT="${SMOKE_FORCE_MIN_UNIQUE_PER_PLACEMENT:-2}"
    ;;
  h100_q650)
    CUDA_HOME=/usr/local/cuda-13.1
    MODEL=meta-llama/Llama-3.3-70B-Instruct
    TP=2; QUANTIZATION=fp8; GPU_UTIL=0.65; MAX_SEQS="${MAX_NUM_SEQS:-16}"
    L1_GB="$LMCACHE_L1_SIZE_GB"
    NUM_QUESTIONS=650; SUBMISSION_BATCH_SIZE="${SUBMISSION_BATCH_SIZE:-650}"
    MIN_TOKENS=128; MAX_TOKENS=512
    RUN_LABEL=h100_shortq_gnn_dynamic_vpc_q650
    L2_DIR="/scratch2/sriramc2/lmcache_mp_l2/gnn_dynamic_vpc_q650_${SLURM_JOB_ID}"
    SMOKE_FORCE_MIN_UNIQUE_PER_PLACEMENT=0
    ;;
  *)
    echo "unknown KV_GNN_DYNAMIC_PROFILE=$PROFILE" >&2
    exit 2
    ;;
esac

EXPERIMENT_LABEL="${EXPERIMENT_LABEL:-}"
RUN_LABEL_SUFFIX="${EXPERIMENT_LABEL:+_${EXPERIMENT_LABEL}}"

# Interactive reruns inside one srun allocation reuse SLURM_JOB_ID. Reserve a
# unique result directory atomically while keeping the job ID at the very end
# of the basename so existing folder-discovery aliases continue to work:
#   run_name_20908
#   2_run_name_20908
#   3_run_name_20908
reserve_unique_run_dir() {
  local base="$1"
  local parent name candidate n=1
  parent="$(dirname -- "$base")"
  name="$(basename -- "$base")"
  mkdir -p "$parent"
  while :; do
    if (( n == 1 )); then
      candidate="$base"
    else
      candidate="${parent}/${n}_${name}"
    fi
    if mkdir "$candidate" 2>/dev/null; then
      printf '%s\n' "$candidate"
      return 0
    fi
    ((n += 1))
  done
}

RUN_DIR="$(reserve_unique_run_dir "${RUN_ROOT}/${RUN_LABEL}${RUN_LABEL_SUFFIX}_${SLURM_JOB_ID}")"
LOG_DIR="${RUN_DIR}/logs"
RESULT_DIR="${RUN_DIR}/results"
PLACEMENT_DIR="${RUN_DIR}/placement"
RUNTIME_METADATA="${PLACEMENT_DIR}/runtime_hash_to_placement.json"
VPC_IMPORTANCE_SIDECAR="${PLACEMENT_DIR}/vpc_importance_by_request.json"
PLACEMENT_TRACE="${PLACEMENT_DIR}/gnn_chunk_placements.jsonl"
HASH_PREDICTION_TRACE="${PLACEMENT_DIR}/gnn_hash_prediction_occurrences.jsonl"
GNN_TIMING_TRACE="${PLACEMENT_DIR}/gnn_prediction_timing.jsonl"
PLACEMENT_SUMMARY="${PLACEMENT_DIR}/gnn_placement_summary.json"
CLEAN_OLD_LMCACHE="${CLEAN_OLD_LMCACHE:-1}"
FOREIGN_MPS_MAX_MIB="${FOREIGN_MPS_MAX_MIB:-64}"

clean_old_lmcache() {
  if [[ "$CLEAN_OLD_LMCACHE" != "1" ]]; then
    echo "Skipping old LMCache cleanup: CLEAN_OLD_LMCACHE=$CLEAN_OLD_LMCACHE"
    return
  fi
  local root=/scratch2/sriramc2
  [[ "$root" == "/scratch2/sriramc2" ]] || exit 12
  echo "WARNING: deleting EVERYTHING under $root"
  mkdir -p "$root"
  find "$root" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
}

clean_old_lmcache
mkdir -p "$LOG_DIR" "$RESULT_DIR" "$PLACEMENT_DIR" "$L2_DIR"

cd "$REPO"
source "${VENV}/bin/activate"
export CUDA_HOME PATH="${CUDA_HOME}/bin:${PATH}" LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_MPS_PIPE_DIRECTORY="/tmp/kvaware-no-mps-${USER}-${SLURM_JOB_ID}-DO-NOT-CREATE"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HOME=/mnt/shared/gpfs/home/sriramc2/.cache/huggingface

check_gpu_exclusivity() {
  local current_job="${SLURM_JOB_ID:-}"
  [[ -n "$current_job" ]] || { echo "ERROR: SLURM_JOB_ID is not set"; return 1; }
  [[ ! -e "$CUDA_MPS_PIPE_DIRECTORY" ]] || {
    echo "ERROR: MPS bypass path unexpectedly exists: $CUDA_MPS_PIPE_DIRECTORY"
    return 44
  }

  local bad=0
  while IFS=',' read -r pid process gpu_uuid used_mem; do
    pid="$(xargs <<<"$pid")"; process="$(xargs <<<"$process")"
    used_mem="$(xargs <<<"$used_mem")"
    [[ -n "$pid" ]] || continue
    local owner cgroup
    owner="$(ps -o user= -p "$pid" 2>/dev/null | xargs || true)"
    cgroup="$(cat "/proc/$pid/cgroup" 2>/dev/null || true)"
    echo "GPU process pid=$pid owner=${owner:-UNKNOWN} process=$process gpu=$gpu_uuid memory=${used_mem}MiB"
    if grep -q "/job_${current_job}/" <<<"$cgroup"; then
      echo "  verdict: current job"
    elif [[ "$process" == *"nvidia-cuda-mps-server"* ]] && \
         [[ "$used_mem" =~ ^[0-9]+$ ]] && (( used_mem <= FOREIGN_MPS_MAX_MIB )); then
      echo "  verdict: small bypassed MPS residue"
    elif [[ "$owner" == "root" ]]; then
      echo "  verdict: root/system process"
    else
      echo "  verdict: FAIL foreign CUDA client"
      bad=1
    fi
  done < <(nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory --format=csv,noheader,nounits 2>/dev/null || true)
  [[ "$bad" == "0" ]] || return 42
}

check_gpu_exclusivity || exit $?

GNN_CHUNK_VOTE_CONFIG_JSON="$(python - <<'PY'
import json
from dataclasses import asdict
from Hierarchical_KV.shortq_placement.chunk_voting import (
    selected_vote_config,
    selected_vote_policy_name,
)
print(json.dumps({
    "policy": selected_vote_policy_name(),
    "config": asdict(selected_vote_config()),
}, sort_keys=True))
PY
)"
export VLLM_GNN_AWARE_VPC="$KV_GNN_AWARE_VPC"
export VLLM_GNN_AWARE_VPC_WINDOW
export VLLM_KV_IMPORTANCE_PLACEMENTS="$VPC_IMPORTANCE_SIDECAR"
export LMCACHE_GNN_DYNAMIC_STORE=1
export LMCACHE_GNN_L1_BACKING="$KV_GNN_L1_BACKING"
export LMCACHE_GNN_L2_BACKING="$KV_GNN_L2_BACKING"
export LMCACHE_GNN_PLACEMENT_METADATA="$RUNTIME_METADATA"

export VLLM_KV_RECOMPUTE_DEBUG=1
export LMCACHE_KV_ACCOUNTING_DEBUG=1
export LMCACHE_PREFETCH_LIFETIME_DEBUG=1 LMCACHE_PREFETCH_LIFETIME_WARN_S=30
export LMCACHE_MP_STAGE1_FRESHNESS_GUARD_S=270
export LMCACHE_MP_CONGESTION_DEBUG=1 LMCACHE_MP_CONGESTION_LOG_EVERY=25 LMCACHE_MP_CONGESTION_SLOW_S=5
export LMCACHE_MP_CHTHM_DEBUG=1
export LMCACHE_MP_PREFIX_DIAGNOSTICS="${LMCACHE_MP_PREFIX_DIAGNOSTICS:-0}"
export LMCACHE_CHUNK_SIZE

LMCACHE_PID=""; VLLM_PID=""
cleanup() {
  set +e
  for pid in "$VLLM_PID" "$LMCACHE_PID"; do
    [[ -n "$pid" ]] || continue
    kill -TERM "$pid" 2>/dev/null || true
    sleep 2
    kill -KILL "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

wait_tcp() {
  local port=$1 pid=$2 limit=$3 start=$SECONDS
  while (( SECONDS-start < limit )); do
    kill -0 "$pid" 2>/dev/null || return 1
    python - "$port" <<'PY' >/dev/null 2>&1 && return 0
import socket, sys
s=socket.socket(); s.settimeout(.25)
r=s.connect_ex(('127.0.0.1', int(sys.argv[1]))); s.close()
raise SystemExit(0 if r == 0 else 1)
PY
    sleep 1
  done
  return 1
}

wait_health() {
  local pid=$1 limit=$2 start=$SECONDS
  while (( SECONDS-start < limit )); do
    kill -0 "$pid" 2>/dev/null || return 1
    curl -sf http://127.0.0.1:8000/health >/dev/null && return 0
    sleep 2
  done
  return 1
}

cat > "${RUN_DIR}/run_config.json" <<JSON
{
  "profile": "$PROFILE",
  "model": "$MODEL",
  "feature_model": "$FEATURE_MODEL",
  "questions": $NUM_QUESTIONS,
  "tp": $TP,
  "gpu_memory_utilization": $GPU_UTIL,
  "max_num_seqs": $MAX_SEQS,
  "l1_gb": $L1_GB,
  "chunk_size": $LMCACHE_CHUNK_SIZE,
  "load_max_in_flight": $LMCACHE_L2_LOAD_MAX_IN_FLIGHT,
  "lookup_max_in_flight": $LMCACHE_L2_LOOKUP_MAX_IN_FLIGHT,
  "store_max_inflight_gb": $LMCACHE_MP_MAX_INFLIGHT_STORE_GB,
  "vpc": true,
  "gnn_aware_vpc": $KV_GNN_AWARE_VPC,
  "gnn_aware_vpc_window": $VLLM_GNN_AWARE_VPC_WINDOW,
  "vpc_sufficient_bypass": $KV_VPC_SUFFICIENT_BYPASS,
  "mutual_prefix": $KV_MUTUAL_PREFIX,
  "experiment_label": "$EXPERIMENT_LABEL",
  "prefix_diagnostics": $LMCACHE_MP_PREFIX_DIAGNOSTICS,
  "stage1_starvation_fallback": $KV_STAGE1_STARVATION_FALLBACK,
  "fs_shared_workers": $KV_FS_SHARED_WORKERS,
  "fs_lookup_workers": $KV_FS_LOOKUP_WORKERS,
  "fs_retrieve_workers": $KV_FS_RETRIEVE_WORKERS,
  "fs_store_workers": $KV_FS_STORE_WORKERS,
  "fs_delete_workers": $KV_FS_DELETE_WORKERS,
  "gnn_l1_backing": $KV_GNN_L1_BACKING,
  "gnn_l2_backing": $KV_GNN_L2_BACKING,
  "cold_warm_store_barrier": $KV_COLD_WARM_STORE_BARRIER,
  "store_barrier_timeout_s": $KV_STORE_BARRIER_TIMEOUT_S,
  "store_barrier_poll_s": $KV_STORE_BARRIER_POLL_S,
  "store_barrier_stable_polls": $KV_STORE_BARRIER_STABLE_POLLS,
  "placement": "shortq_dynamic_vpc_plus_lower_backing",
  "chunk_vote": $GNN_CHUNK_VOTE_CONFIG_JSON,
  "runtime_metadata": "$RUNTIME_METADATA",
  "vpc_importance_sidecar": "$VPC_IMPORTANCE_SIDECAR",
  "gnn_inference_batch_size": $GNN_INFERENCE_BATCH_SIZE,
  "smoke_force_min_unique_per_placement": $SMOKE_FORCE_MIN_UNIQUE_PER_PLACEMENT
}
JSON

echo "===== SHORT-Q PRECOMPUTE ($PROFILE) ====="
python experiments/precompute_shortq_placements.py \
  --dataset_name "$DATASET_NAME" --max_questions "$NUM_QUESTIONS" --retrieval_top_k "$RETRIEVAL_TOP_K" \
  --request_order "$REQUEST_ORDER" --prefix_sort_depth "$PREFIX_SORT_DEPTH" --request_order_seed "$REQUEST_ORDER_SEED" \
  --serving_model "$MODEL" --feature_model "$FEATURE_MODEL" --checkpoint "$CHECKPOINT" \
  --chunk_size "$LMCACHE_CHUNK_SIZE" --max_seq_len 8000 --device cuda:0 \
  --inference_batch_size "$GNN_INFERENCE_BATCH_SIZE" \
  --runtime_metadata "$RUNTIME_METADATA" --vpc_importance_sidecar "$VPC_IMPORTANCE_SIDECAR" \
  --placement_trace "$PLACEMENT_TRACE" --hash_prediction_trace "$HASH_PREDICTION_TRACE" \
  --timing_trace "$GNN_TIMING_TRACE" --summary "$PLACEMENT_SUMMARY" \
  --smoke_force_min_unique_per_placement "$SMOKE_FORCE_MIN_UNIQUE_PER_PLACEMENT" \
  > "${LOG_DIR}/gnn_precompute.log" 2>&1

python - "$RUNTIME_METADATA" "$VPC_IMPORTANCE_SIDECAR" "$PLACEMENT_SUMMARY" <<'PY'
import json, sys
runtime=json.load(open(sys.argv[1])); sidecar=json.load(open(sys.argv[2])); summary=json.load(open(sys.argv[3]))
assert runtime and all(v in {'gpu','cpu','disk'} for v in runtime.values())
assert len(sidecar) == summary['requests']
valid={'gpu','cpu','disk'}
assert all(all(v in valid for v in blocks.values()) for blocks in sidecar.values())
print('placement_entries', len(runtime))
print('placement_counts', summary['unique_runtime_placement_counts'])
print('vpc_sidecar_requests', len(sidecar))
PY

L2_JSON=$(python - "$L2_DIR" "$KV_FS_SHARED_WORKERS" \
  "$KV_FS_LOOKUP_WORKERS" "$KV_FS_RETRIEVE_WORKERS" \
  "$KV_FS_STORE_WORKERS" "$KV_FS_DELETE_WORKERS" <<'PY'
import json,sys
spec = {
    'type': 'fs_native',
    'base_path': sys.argv[1],
    'num_workers': int(sys.argv[2]),
    'use_odirect': False,
}
names = ('lookup', 'retrieve', 'store', 'delete')
counts = map(int, sys.argv[3:7])
per_op = {name: count for name, count in zip(names, counts) if count > 0}
if per_op:
    spec['per_op_workers'] = per_op
print(json.dumps(spec))
PY
)
echo "fs_native adapter: $L2_JSON"

echo "===== START LMCACHE ====="
lmcache server --host localhost --port 5556 --chunk-size "$LMCACHE_CHUNK_SIZE" \
  --l1-size-gb "$L1_GB" \
  --l2-load-max-in-flight "$LMCACHE_L2_LOAD_MAX_IN_FLIGHT" \
  --l2-lookup-max-in-flight "$LMCACHE_L2_LOOKUP_MAX_IN_FLIGHT" \
  --l2-store-max-inflight-gb "$LMCACHE_MP_MAX_INFLIGHT_STORE_GB" \
  --l2-store-policy gnn_dynamic --eviction-policy LRU --max-workers "$LMCACHE_MAX_WORKERS" \
  --l2-adapter "$L2_JSON" > "${LOG_DIR}/lmcache.log" 2>&1 &
LMCACHE_PID=$!
wait_tcp 5556 "$LMCACHE_PID" 240 || { tail -200 "${LOG_DIR}/lmcache.log"; exit 30; }
curl -sf http://127.0.0.1:8080/metrics > "${RUN_DIR}/lmcache_metrics_before.txt" || true

KV_CONFIG=$(python - "$LMCACHE_MP_TIMEOUT" "$LMCACHE_MP_HEARTBEAT_INTERVAL" \
  "$KV_STAGE1_STARVATION_FALLBACK" "$KV_VPC_SUFFICIENT_BYPASS" \
  "$KV_MUTUAL_PREFIX" <<'PY'
import json,sys
timeout=float(sys.argv[1])
heartbeat_interval=float(sys.argv[2])
print(json.dumps({
  'kv_connector':'LMCacheMPConnector',
  'kv_connector_module_path':'lmcache.integration.vllm.lmcache_mp_connector',
  'kv_role':'kv_both',
  'kv_load_failure_policy':'recompute',
  'kv_connector_extra_config':{
    'lmcache.mp.host':'tcp://localhost',
    'lmcache.mp.port':5556,
    'lmcache.mp.mq_timeout':timeout,
    'lmcache.mp.heartbeat_interval':heartbeat_interval,
    'lmcache.mp.lookup_timeout':0.0,
    'lmcache.mp.starvation_fallback':bool(int(sys.argv[3])),
    'lmcache.mp.vpc_sufficient_bypass':bool(int(sys.argv[4])),
    'lmcache.mp.mutual_prefix':bool(int(sys.argv[5])),
  },
}))
PY
)

VARGS=(
  serve "$MODEL" --host 127.0.0.1 --port 8000 --dtype bfloat16
  --tensor-parallel-size "$TP" --max-model-len 8000
  --gpu-memory-utilization "$GPU_UTIL" --max-num-seqs "$MAX_SEQS"
  --enable-chunked-prefill --disable-hybrid-kv-cache-manager
  --enable-prefix-caching --kv-transfer-config "$KV_CONFIG"
)
[[ -z "$QUANTIZATION" ]] || VARGS+=(--quantization "$QUANTIZATION")

echo "===== START VLLM (VPC ON, GPU_UTIL=$GPU_UTIL) ====="
vllm "${VARGS[@]}" > "${LOG_DIR}/vllm.log" 2>&1 &
VLLM_PID=$!
wait_health "$VLLM_PID" 1500 || { tail -300 "${LOG_DIR}/vllm.log"; exit 40; }
curl -sf http://127.0.0.1:8000/metrics > "${RUN_DIR}/metrics_before.txt" || true

DARGS=(
  --dataset_name "$DATASET_NAME" --max_questions "$NUM_QUESTIONS"
  --retrieval_top_k "$RETRIEVAL_TOP_K" --llm_model "$MODEL"
  --request_order "$REQUEST_ORDER" --prefix_sort_depth "$PREFIX_SORT_DEPTH"
  --request_order_seed "$REQUEST_ORDER_SEED" --warm_order "$WARM_ORDER"
  --submission_batch_size "$SUBMISSION_BATCH_SIZE"
  --server_url http://127.0.0.1:8000 --request_timeout_s 7200
  --temperature 0 --top_p 0.95 --min_tokens "$MIN_TOKENS" --max_tokens "$MAX_TOKENS"
  --metrics_after_cold_path "${RUN_DIR}/metrics_after_cold.txt"
  --deterministic_request_ids --output_dir "$RESULT_DIR"
)

if [[ "$KV_COLD_WARM_STORE_BARRIER" == "1" ]]; then
  DARGS+=(
    --between_phases_store_barrier
    --lmcache_status_url http://127.0.0.1:8080/status
    --store_barrier_timeout_s "$KV_STORE_BARRIER_TIMEOUT_S"
    --store_barrier_poll_s "$KV_STORE_BARRIER_POLL_S"
    --store_barrier_stable_polls "$KV_STORE_BARRIER_STABLE_POLLS"
    --store_barrier_result_path "${RUN_DIR}/store_barrier.json"
    --metrics_after_barrier_path "${RUN_DIR}/metrics_after_barrier.txt"
  )
fi

echo "===== COLD + WARM WORKLOAD ====="
CUDA_VISIBLE_DEVICES="" python experiments/mp_nognn_project.py "${DARGS[@]}" \
  > "${LOG_DIR}/driver.log" 2>&1 || { tail -300 "${LOG_DIR}/driver.log"; exit 50; }

sleep 5
curl -sf http://127.0.0.1:8000/metrics > "${RUN_DIR}/metrics_after.txt" || true
curl -sf http://127.0.0.1:8080/metrics > "${RUN_DIR}/lmcache_metrics_after.txt" || true

python scripts/analyze_mp_congestion_chthm.py \
  --lmcache-log "${LOG_DIR}/lmcache.log" --vllm-log "${LOG_DIR}/vllm.log" \
  --results-dir "$RESULT_DIR" --output "${RUN_DIR}/mp_diag_summary.json" \
  > "${RUN_DIR}/mp_diag_summary.txt" 2>&1 || true
python scripts/analyze_vllm_phase_metrics.py \
  --before "${RUN_DIR}/metrics_before.txt" --after-cold "${RUN_DIR}/metrics_after_cold.txt" \
  --after-warm "${RUN_DIR}/metrics_after.txt" --output "${RUN_DIR}/vllm_phase_metrics.json" \
  > "${RUN_DIR}/vllm_phase_metrics.txt" 2>&1 || true

python - "$RESULT_DIR" "$PLACEMENT_SUMMARY" "$LOG_DIR" "$PROFILE" \
  "$KV_GNN_AWARE_VPC" "$KV_GNN_L1_BACKING" "$KV_GNN_L2_BACKING" \
  "$KV_COLD_WARM_STORE_BARRIER" <<'PY'
import json, pathlib, re, sys
results=pathlib.Path(sys.argv[1]); placement=json.load(open(sys.argv[2])); logs=pathlib.Path(sys.argv[3])
profile=sys.argv[4]; aware=sys.argv[5]=='1'; backing=sys.argv[6]=='1'
l2_backing=sys.argv[7]=='1'; store_barrier=sys.argv[8]=='1'
summary=json.load(open(results/'summary.json'))
assert summary['cold']['successful']==summary['cold']['requests']
assert summary['warm']['successful']==summary['warm']['requests']
assert bool(summary.get('store_barrier', {}).get('enabled', False)) == store_barrier, \
    ('store barrier gate/result mismatch', store_barrier, summary.get('store_barrier'))
lm=(logs/'lmcache.log').read_text(errors='replace')
vl=(logs/'vllm.log').read_text(errors='replace')
assert '[GNN_DYNAMIC_STORE]' in lm, 'dynamic GNN store path never executed'
assert '[GNN_DYNAMIC_L2_POLICY]' in lm, 'dynamic GNN L2 policy never executed'
if l2_backing:
    assert '[GNN_DYNAMIC_L2_BACKING_INIT] enabled=True' in lm, \
        'GNN L2 backing was requested but policy did not enable it'
    assert re.search(r'\[GNN_DYNAMIC_L2_POLICY\].*backing_targets=[1-9]\d*.*l2_backing=True', lm), \
        'GNN L2 backing enabled but no GPU/L1 placement was targeted for L2 backing'
assert '[GNN_PLACEMENT_METADATA_MISS]' not in lm, 'placement metadata miss detected'
if aware:
    assert '[GNN_AWARE_VPC_INIT]' in vl, 'GNN-aware VPC was requested but not initialized'
else:
    assert '[GNN_AWARE_VPC_INIT]' not in vl, 'GNN-aware VPC initialized while gate is off'

if profile.endswith('_smoke'):
    counts=placement['unique_runtime_placement_counts']
    assert all(counts.get(t,0) >= 2 for t in ('gpu','cpu','disk')), counts
    if aware:
        for tier in ('gpu','cpu','disk'):
            assert f'[GNN_AWARE_VPC_LABEL] first_placement={tier}' in vl, f'missing VPC label {tier}'
    if backing:
        assert re.search(r'\[GNN_DYNAMIC_STORE\].*l1_backing=True.*reserved_gpu=[1-9]\d*', lm), \
            'smoke never physically created an L1 backing copy for a GPU-labelled chunk'
    else:
        assert not re.search(r'\[GNN_DYNAMIC_STORE\].*reserved_gpu=[1-9]\d*', lm), \
            'GPU-labelled chunks reached L1 while backing gate was off'
    assert re.search(r'\[GNN_DYNAMIC_STORE\].*reserved_cpu=[1-9]\d*', lm), \
        'smoke never persisted a CPU-labelled chunk in L1'
    assert '[GNN_DYNAMIC_L2_COMMIT]' in lm, 'smoke never completed an L2 commit'
    print('GNN_DYNAMIC_VPC_SMOKE_PASS')
print('GNN_DYNAMIC_VPC_RUN_PASS')
PY

echo "===== RESULTS ====="
cat "${RESULT_DIR}/summary.json"
echo "placement:"; cat "$PLACEMENT_SUMMARY"
echo "run_dir=$RUN_DIR"
