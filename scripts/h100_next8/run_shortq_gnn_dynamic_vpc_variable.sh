#!/usr/bin/env bash
set -Eeuo pipefail

PROFILE="${KV_GNN_DYNAMIC_PROFILE:?wrapper must set KV_GNN_DYNAMIC_PROFILE}"
REPO=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
VENV=/mnt/shared/gpfs/home/sriramc2/venvs/kvaware-vllm029
RUN_ROOT=/mnt/shared/gpfs/home/sriramc2/runs/kvaware
DATASET_NAME="${NEXT8_DATASET:-medical}"
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
LMCACHE_L2_SIZE_GB="${LMCACHE_L2_SIZE_GB:-100}"
# CE28G: these are the exact physical L1 eviction controls passed to LMCache.
# Previously the server used 0.80 / 0.20 implicit CLI defaults.
LMCACHE_L1_EVICTION_TRIGGER="${LMCACHE_L1_EVICTION_TRIGGER:-0.80}"
LMCACHE_L1_EVICTION_RATIO="${LMCACHE_L1_EVICTION_RATIO:-0.20}"
LMCACHE_L2_EVICTION_TRIGGER="${LMCACHE_L2_EVICTION_TRIGGER:-0.90}"
LMCACHE_L2_EVICTION_RATIO="${LMCACHE_L2_EVICTION_RATIO:-0.10}"
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
KV_STORE_BARRIER_STALL_S="${KV_STORE_BARRIER_STALL_S:-120}"

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
  "$KV_FS_STORE_WORKERS" "$KV_FS_DELETE_WORKERS" \
  "$LMCACHE_L2_SIZE_GB" "$LMCACHE_L2_EVICTION_TRIGGER" "$LMCACHE_L2_EVICTION_RATIO" <<'PY'
import sys
load_pf, lookup_pf = map(int, sys.argv[1:3])
store_cap_gb = float(sys.argv[3])
shared = int(sys.argv[4])
# The four per-operation worker counts precede the three L2 capacity/eviction
# arguments. Do not try to parse float watermarks (e.g. 0.90) as worker counts.
assert len(sys.argv) == 12, f"Unexpected L2 validation argument count: {len(sys.argv) - 1}"
workers = list(map(int, sys.argv[5:9]))
l2_size_gb, trigger, ratio = map(float, sys.argv[9:12])
assert load_pf > 0 and lookup_pf > 0, (load_pf, lookup_pf)
assert store_cap_gb >= 0, store_cap_gb
assert shared > 0, shared
assert all(x >= 0 for x in workers), workers
assert l2_size_gb > 0, l2_size_gb
assert 0 < trigger <= 1 and 0 < ratio <= 1, (trigger, ratio)
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
    NUM_QUESTIONS="${NEXT8_Q:-650}"; SUBMISSION_BATCH_SIZE="${SUBMISSION_BATCH_SIZE:-650}"
    MIN_TOKENS="${MIN_TOKENS:-128}"; MAX_TOKENS="${MAX_TOKENS:-512}"
    RUN_LABEL=h100_shortq_gnn_dynamic_vpc_q650
    L2_DIR="/scratch2/sriramc2/lmcache_mp_l2/gnn_dynamic_vpc_q650_${SLURM_JOB_ID}"
    SMOKE_FORCE_MIN_UNIQUE_PER_PLACEMENT=0
    ;;
  *)
    echo "unknown KV_GNN_DYNAMIC_PROFILE=$PROFILE" >&2
    exit 2
    ;;
esac

# [SC] Keep the historical defaults above, but allow controlled benchmark
# runs to choose an independent cold permutation without rewriting legacy
# launch behavior.  WARM_ORDER already has its historical env override.
REQUEST_ORDER="${KV_REQUEST_ORDER:-$REQUEST_ORDER}"
REQUEST_ORDER_SEED="${KV_REQUEST_ORDER_SEED:-$REQUEST_ORDER_SEED}"
PASSAGE_ORDER_POLICY="${KV_PASSAGE_ORDER_POLICY:-original}"
case "$PASSAGE_ORDER_POLICY" in
  original|canonical) ;;
  *) echo "ERROR: KV_PASSAGE_ORDER_POLICY must be original or canonical" >&2; exit 2 ;;
esac
WARM_ORDER_SEED="${WARM_ORDER_SEED:-$REQUEST_ORDER_SEED}"

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
# Both flags are opt-in; tracing does not change upstream Prometheus definitions.
export VLLM_PHASE_TRACE="${VLLM_PHASE_TRACE:-0}"
export VLLM_PHASE_TRACE_DEVICE_TIMING="${VLLM_PHASE_TRACE_DEVICE_TIMING:-0}"
export VLLM_PHASE_TRACE_DIR="${RUN_DIR}/logs/phase_trace"
for trace_switch in VLLM_PHASE_TRACE VLLM_PHASE_TRACE_DEVICE_TIMING; do
  case "${!trace_switch}" in 0|1) ;; *) echo "bad $trace_switch" >&2; exit 2 ;; esac
done
RUNTIME_METADATA="${PLACEMENT_DIR}/runtime_hash_to_placement.json"
VPC_IMPORTANCE_SIDECAR="${PLACEMENT_DIR}/vpc_importance_by_request.json"
PLACEMENT_TRACE="${PLACEMENT_DIR}/gnn_chunk_placements.jsonl"
HASH_PREDICTION_TRACE="${PLACEMENT_DIR}/gnn_hash_prediction_occurrences.jsonl"
GNN_TIMING_TRACE="${PLACEMENT_DIR}/gnn_prediction_timing.jsonl"
PLACEMENT_SUMMARY="${PLACEMENT_DIR}/gnn_placement_summary.json"
KV_BAYES_ENABLE="${KV_BAYES_ENABLE:-0}"
# [SC CE28G] One switch; no separate Bayes watermark values.
KV_BAYES_USE_WATERMARK_CAPACITY="${KV_BAYES_USE_WATERMARK_CAPACITY:-0}"
[[ "$KV_BAYES_USE_WATERMARK_CAPACITY" == 0 || "$KV_BAYES_USE_WATERMARK_CAPACITY" == 1 ]] || {
  echo "KV_BAYES_USE_WATERMARK_CAPACITY must be 0 or 1" >&2; exit 2;
}
KV_BAYES_PRIOR_WEIGHT="${KV_BAYES_PRIOR_WEIGHT:-1}"
KV_BAYES_PRIOR_FILE="${KV_BAYES_PRIOR_FILE:-${REPO}/Hierarchical_KV/shortq_placement/priors/rank_example.json}"
KV_BAYES_SCORE_FILE="${PLACEMENT_DIR}/gnn_chunk_scores.jsonl"
KV_BAYES_SCORE_FORMAT="${KV_BAYES_SCORE_FORMAT:-mean_32_block_rank_scores}"
case "$KV_BAYES_SCORE_FORMAT" in
  mean_32_block_rank_scores|expected_ordinal_0_100) ;;
  *) echo "Unsupported KV_BAYES_SCORE_FORMAT=$KV_BAYES_SCORE_FORMAT" >&2; exit 2 ;;
esac
KV_BAYES_GPU_GB="${KV_BAYES_GPU_GB:-10}"
KV_BAYES_DISK_GB="${KV_BAYES_DISK_GB:-$LMCACHE_L2_SIZE_GB}"
if [[ "$KV_BAYES_ENABLE" == "1" && "$KV_BAYES_DISK_GB" != "$LMCACHE_L2_SIZE_GB" ]]; then
  echo "ERROR: mapper disk budget must match physical L2 size for this experiment" >&2; exit 2
fi
# [SC CE28G] Physical eviction knobs are authoritative. Reject values before
# any service starts, and compute mapper budgets from those exact knobs.
CE28G_CAP_VALUES="$(python - "$L1_GB" "$KV_BAYES_DISK_GB" \
    "$LMCACHE_L1_EVICTION_TRIGGER" "$LMCACHE_L1_EVICTION_RATIO" \
    "$LMCACHE_L2_EVICTION_TRIGGER" "$LMCACHE_L2_EVICTION_RATIO" \
    "$KV_BAYES_USE_WATERMARK_CAPACITY" <<'PY_CE28G'
import math,sys
cpu,disk,l1wm,l1ratio,l2wm,l2ratio,enabled=map(float,sys.argv[1:])
assert cpu>0 and disk>0 and all(math.isfinite(v) for v in (cpu,disk,l1wm,l1ratio,l2wm,l2ratio))
assert all(0<x<=1 for x in (l1wm,l1ratio,l2wm,l2ratio))
assert enabled in (0,1)
print(f'{cpu*l1wm if enabled else cpu:.8f} {disk*l2wm if enabled else disk:.8f}')
PY_CE28G
)"
read -r BAYES_CPU_EFFECTIVE_GB BAYES_DISK_EFFECTIVE_GB <<< "$CE28G_CAP_VALUES"
echo "[CE28G_CAPACITY] physical_cpu=$L1_GB physical_disk=$KV_BAYES_DISK_GB "\
  "l1_watermark=$LMCACHE_L1_EVICTION_TRIGGER l1_evict_ratio=$LMCACHE_L1_EVICTION_RATIO "\
  "l2_watermark=$LMCACHE_L2_EVICTION_TRIGGER l2_evict_ratio=$LMCACHE_L2_EVICTION_RATIO "\
  "apply=$KV_BAYES_USE_WATERMARK_CAPACITY effective_cpu=$BAYES_CPU_EFFECTIVE_GB "\
  "effective_disk=$BAYES_DISK_EFFECTIVE_GB"

KV_BAYES_USE_COST_PROXIES="${KV_BAYES_USE_COST_PROXIES:-0}"
KV_BAYES_DIRECT_COST_FEEDBACK="${KV_BAYES_DIRECT_COST_FEEDBACK:-0}"
KV_BAYES_POSTERIOR_MODE="${KV_BAYES_POSTERIOR_MODE:-fixed}"
KV_BAYES_SCORE_PRIOR_COUNT="${KV_BAYES_SCORE_PRIOR_COUNT:-10}"
KV_BAYES_COST_PRIOR_COUNT="${KV_BAYES_COST_PRIOR_COUNT:-5}"
KV_BAYES_READMISSION="${KV_BAYES_READMISSION:-1}"
KV_BAYES_RANDOM_SCORE_SEED="${KV_BAYES_RANDOM_SCORE_SEED:-}"
if [[ "$KV_BAYES_ENABLE" == "1" ]]; then
  export LMCACHE_BAYES_TIMING="$KV_BAYES_DIRECT_COST_FEEDBACK"
  export LMCACHE_BAYES_READMISSION="$KV_BAYES_READMISSION"
fi
if [[ "$KV_BAYES_ENABLE" == "1" ]]; then
  export LMCACHE_GNN_PLACEMENT_DYNAMIC=1 VLLM_KV_IMPORTANCE_DYNAMIC=1
  SMOKE_FORCE_MIN_UNIQUE_PER_PLACEMENT=0
  [[ -f "$KV_BAYES_PRIOR_FILE" ]] || {
    echo "Missing Bayesian prior JSON: $KV_BAYES_PRIOR_FILE" >&2; exit 2;
  }
  # Never silently interpret the old negative rank-score histogram as 0-100.
  python - "$KV_BAYES_PRIOR_FILE" "$KV_BAYES_SCORE_FORMAT" <<'PY'
import json, sys
prior = json.load(open(sys.argv[1]))
actual, expected = prior.get('score_format'), sys.argv[2]
if actual != expected:
    raise SystemExit(f"Bayesian prior score_format={actual!r} != selected {expected!r}; "
                     "regenerate or select the compatible score histogram")
print('[BAYES_SCORE_PRIOR_FORMAT]', actual, flush=True)
PY
fi
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
import os
from dataclasses import asdict
from Hierarchical_KV.shortq_placement.chunk_voting import (
    selected_vote_config,
    selected_vote_policy_name,
)
print(json.dumps({
    "policy": selected_vote_policy_name(),
    "config": asdict(selected_vote_config()),
    "random_chunk_seed": os.getenv("GNN_RANDOM_CHUNK_SEED", "0"),
    "random_match_distribution": os.getenv("GNN_RANDOM_MATCH_DISTRIBUTION", ""),
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

{ echo "job_id=${SLURM_JOB_ID:-unset}";
  echo "cpus_per_task=${SLURM_CPUS_PER_TASK:-unset}";
  echo "nproc=$(nproc)";
  taskset -pc $$ || true;
  scontrol show job "${SLURM_JOB_ID:-}" 2>/dev/null | grep -E 'NumCPUs=|ReqTRES=|AllocTRES=' || true;
  git rev-parse HEAD 2>/dev/null || true;
  git status --short 2>/dev/null || true;
} > "${RUN_DIR}/slurm_and_source.txt"

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
  "request_order": "$REQUEST_ORDER",
  "passage_order_policy": "$PASSAGE_ORDER_POLICY",
  "phase_trace": $VLLM_PHASE_TRACE,
  "phase_trace_device_timing": $VLLM_PHASE_TRACE_DEVICE_TIMING,
  "request_order_seed": $REQUEST_ORDER_SEED,
  "warm_order": "$WARM_ORDER",
  "warm_order_seed": $WARM_ORDER_SEED,
  "min_tokens": $MIN_TOKENS,
  "max_tokens": $MAX_TOKENS,
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
  "store_barrier_stall_s": $KV_STORE_BARRIER_STALL_S,
  "placement": "shortq_dynamic_vpc_plus_lower_backing",
  "chunk_vote": $GNN_CHUNK_VOTE_CONFIG_JSON,
  "runtime_metadata": "$RUNTIME_METADATA",
  "vpc_importance_sidecar": "$VPC_IMPORTANCE_SIDECAR",
  "bayesian_enabled": $KV_BAYES_ENABLE,
  "bayes_prior_weight": $KV_BAYES_PRIOR_WEIGHT,
  "bayes_prior_file": "$KV_BAYES_PRIOR_FILE",
  "bayes_score_file": "$KV_BAYES_SCORE_FILE",
  "bayes_score_format": "$KV_BAYES_SCORE_FORMAT",
  "bayes_gpu_gb": $KV_BAYES_GPU_GB,
  "bayes_disk_gb": $KV_BAYES_DISK_GB,
  "lmcache_l2_size_gb": $LMCACHE_L2_SIZE_GB,
  "lmcache_l1_eviction_trigger": $LMCACHE_L1_EVICTION_TRIGGER,
  "lmcache_l1_eviction_ratio": $LMCACHE_L1_EVICTION_RATIO,
  "bayes_use_watermark_capacity": $KV_BAYES_USE_WATERMARK_CAPACITY,
  "bayes_cpu_effective_gb": $BAYES_CPU_EFFECTIVE_GB,
  "bayes_disk_effective_gb": $BAYES_DISK_EFFECTIVE_GB,
  "lmcache_l2_eviction_trigger": $LMCACHE_L2_EVICTION_TRIGGER,
  "lmcache_l2_eviction_ratio": $LMCACHE_L2_EVICTION_RATIO,
  "bayes_cost_observation_proxies": $KV_BAYES_USE_COST_PROXIES,
  "bayes_direct_timing": $KV_BAYES_DIRECT_COST_FEEDBACK,
  "bayes_readmission": $KV_BAYES_READMISSION,
  "bayes_posterior_mode": "$KV_BAYES_POSTERIOR_MODE",
  "bayes_score_prior_count": $KV_BAYES_SCORE_PRIOR_COUNT,
  "bayes_cost_prior_count": $KV_BAYES_COST_PRIOR_COUNT,
  "bayes_random_score_seed": "$KV_BAYES_RANDOM_SCORE_SEED",
  "gnn_inference_batch_size": $GNN_INFERENCE_BATCH_SIZE,
  "smoke_force_min_unique_per_placement": $SMOKE_FORCE_MIN_UNIQUE_PER_PLACEMENT
}
JSON

BAYES_PRECOMPUTE_ARGS=()
if [[ "$KV_BAYES_ENABLE" == "1" ]]; then
  BAYES_PRECOMPUTE_ARGS=(--bayes_scores_only --bayes_score_file "$KV_BAYES_SCORE_FILE"
                         --bayes_score_format "$KV_BAYES_SCORE_FORMAT")
fi
echo "===== SHORT-Q PRECOMPUTE ($PROFILE) ====="
python experiments/precompute_shortq_placements.py \
  --dataset_name "$DATASET_NAME" --max_questions "$NUM_QUESTIONS" --retrieval_top_k "$RETRIEVAL_TOP_K" \
  --request_order "$REQUEST_ORDER" --prefix_sort_depth "$PREFIX_SORT_DEPTH" --request_order_seed "$REQUEST_ORDER_SEED" \
  --passage_order_policy "$PASSAGE_ORDER_POLICY" \
  --serving_model "$MODEL" --feature_model "$FEATURE_MODEL" --checkpoint "$CHECKPOINT" \
  --chunk_size "$LMCACHE_CHUNK_SIZE" --max_seq_len 8000 --device cuda:0 \
  --inference_batch_size "$GNN_INFERENCE_BATCH_SIZE" \
  --runtime_metadata "$RUNTIME_METADATA" --vpc_importance_sidecar "$VPC_IMPORTANCE_SIDECAR" \
  --placement_trace "$PLACEMENT_TRACE" --hash_prediction_trace "$HASH_PREDICTION_TRACE" \
  --timing_trace "$GNN_TIMING_TRACE" --summary "$PLACEMENT_SUMMARY" \
  --smoke_force_min_unique_per_placement "$SMOKE_FORCE_MIN_UNIQUE_PER_PLACEMENT" \
  "${BAYES_PRECOMPUTE_ARGS[@]}" \
  > "${LOG_DIR}/gnn_precompute.log" 2>&1

if [[ "$KV_BAYES_ENABLE" == "1" ]]; then
  python - "$KV_BAYES_SCORE_FILE" "$NUM_QUESTIONS" "$KV_BAYES_SCORE_FORMAT" <<'PY'
import json, sys
rows=[json.loads(x) for x in open(sys.argv[1]) if x.strip()]
assert len(rows)==int(sys.argv[2]), (len(rows),sys.argv[2])
assert all(row.get('score_format') == sys.argv[3] for row in rows), 'Wrong GNN score format'
assert all(isinstance(row['chunks'],list) and row['chunks'] for row in rows)
print('[BAYES_SCORES_READY] requests=',len(rows),
      'chunks=',sum(len(row['chunks']) for row in rows))
PY
else
python - "$RUNTIME_METADATA" "$VPC_IMPORTANCE_SIDECAR" "$PLACEMENT_SUMMARY" <<'PY'
import json, sys
runtime=json.load(open(sys.argv[1])); sidecar=json.load(open(sys.argv[2])); summary=json.load(open(sys.argv[3]))
assert runtime and all(v in {'gpu','cpu','disk','drop'} for v in runtime.values())
assert len(sidecar) == summary['requests']
valid={'gpu','cpu','disk'}
assert all(all(v in valid for v in blocks.values()) for blocks in sidecar.values())
print('placement_entries', len(runtime))
print('placement_counts', summary['unique_runtime_placement_counts'])
print('vpc_sidecar_requests', len(sidecar))
PY
fi

L2_JSON=$(python - "$L2_DIR" "$KV_FS_SHARED_WORKERS" \
  "$KV_FS_LOOKUP_WORKERS" "$KV_FS_RETRIEVE_WORKERS" \
  "$KV_FS_STORE_WORKERS" "$KV_FS_DELETE_WORKERS" \
  "$LMCACHE_L2_SIZE_GB" "$LMCACHE_L2_EVICTION_TRIGGER" "$LMCACHE_L2_EVICTION_RATIO" <<'PY'
import json,sys
spec = {
    'type': 'fs_native',
    'base_path': sys.argv[1],
    'num_workers': int(sys.argv[2]),
    'use_odirect': False,
}
size, trigger, ratio = map(float, sys.argv[7:10])
if size <= 0 or not 0 < trigger <= 1 or not 0 < ratio <= 1:
    raise ValueError('L2 capacity must be positive; watermarks must be in (0,1]')
spec['max_capacity_gb'] = size
spec['eviction'] = {'eviction_policy': 'LRU', 'trigger_watermark': trigger,
                    'eviction_ratio': ratio}
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
  --eviction-trigger-watermark "$LMCACHE_L1_EVICTION_TRIGGER" \
  --eviction-ratio "$LMCACHE_L1_EVICTION_RATIO" \
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
  --request_order_seed "$REQUEST_ORDER_SEED" --passage_order_policy "$PASSAGE_ORDER_POLICY" --warm_order "$WARM_ORDER"
  --warm_order_seed "$WARM_ORDER_SEED"
  --submission_batch_size "$SUBMISSION_BATCH_SIZE"
  --server_url http://127.0.0.1:8000 --request_timeout_s 7200
  --temperature 0 --top_p 0.95 --min_tokens "$MIN_TOKENS" --max_tokens "$MAX_TOKENS"
  --metrics_after_cold_path "${RUN_DIR}/metrics_after_cold.txt"
  --deterministic_request_ids --output_dir "$RESULT_DIR"
)

if [[ "$KV_BAYES_ENABLE" == "1" ]]; then
  DARGS+=(
    --bayes_score_file "$KV_BAYES_SCORE_FILE"
    --bayes_prior_file "$KV_BAYES_PRIOR_FILE"
    --bayes_prior_weight "$KV_BAYES_PRIOR_WEIGHT"
    --bayes_posterior_mode "$KV_BAYES_POSTERIOR_MODE"
    --bayes_score_prior_count "$KV_BAYES_SCORE_PRIOR_COUNT"
    --bayes_cost_prior_count "$KV_BAYES_COST_PRIOR_COUNT"
    --bayes_gpu_gb "$KV_BAYES_GPU_GB"
    --bayes_cpu_gb "$BAYES_CPU_EFFECTIVE_GB"
    --bayes_disk_gb "$BAYES_DISK_EFFECTIVE_GB"
    --bayes_metadata_file "$RUNTIME_METADATA"
    --bayes_vpc_file "$VPC_IMPORTANCE_SIDECAR"
    --bayes_lmcache_log "${LOG_DIR}/lmcache.log"
  )
  if [[ "$KV_BAYES_USE_COST_PROXIES" == "1" ]]; then
    DARGS+=(--bayes_use_cost_proxies)
  fi
  if [[ "$KV_BAYES_DIRECT_COST_FEEDBACK" == "1" ]]; then
    DARGS+=(--bayes_direct_cost_feedback)
  fi
  if [[ "$KV_BAYES_READMISSION" == "1" ]]; then
    DARGS+=(--bayes_readmission)
  fi
  if [[ -n "$KV_BAYES_RANDOM_SCORE_SEED" ]]; then
    DARGS+=(--bayes_random_score_seed "$KV_BAYES_RANDOM_SCORE_SEED")
  fi
fi

if [[ "$KV_COLD_WARM_STORE_BARRIER" == "1" ]]; then
  DARGS+=(
    --between_phases_store_barrier
    --lmcache_status_url http://127.0.0.1:8080/status
    --store_barrier_timeout_s "$KV_STORE_BARRIER_TIMEOUT_S"
    --store_barrier_poll_s "$KV_STORE_BARRIER_POLL_S"
    --store_barrier_stable_polls "$KV_STORE_BARRIER_STABLE_POLLS"
    --store_barrier_stall_s "$KV_STORE_BARRIER_STALL_S"
    --store_barrier_result_path "${RUN_DIR}/store_barrier.json"
    --metrics_after_barrier_path "${RUN_DIR}/metrics_after_barrier.txt"
  )
fi

echo "===== COLD + WARM WORKLOAD ====="
CUDA_VISIBLE_DEVICES="" python experiments/mp_nognn_project.py "${DARGS[@]}" \
  > "${LOG_DIR}/driver.log" 2>&1 || { tail -300 "${LOG_DIR}/driver.log"; exit 50; }

sleep 5
if [[ "$VLLM_PHASE_TRACE" == "1" ]]; then
  python scripts/analyze_phase_trace.py \
    --trace-dir "$VLLM_PHASE_TRACE_DIR" \
    --output "${RUN_DIR}/phase_trace_summary.json" \
    > "${RUN_DIR}/phase_trace_summary.txt" 2>&1 || true
fi
curl -sf http://127.0.0.1:8000/metrics > "${RUN_DIR}/metrics_after.txt" || true
curl -sf http://127.0.0.1:8080/metrics > "${RUN_DIR}/lmcache_metrics_after.txt" || true

python scripts/analyze_mp_congestion_chthm.py \
  --lmcache-log "${LOG_DIR}/lmcache.log" --vllm-log "${LOG_DIR}/vllm.log" \
  --results-dir "$RESULT_DIR" --strict-phase \
  --output "${RUN_DIR}/mp_diag_summary.json" \
  > "${RUN_DIR}/mp_diag_summary.txt" 2>&1 || {
    cat "${RUN_DIR}/mp_diag_summary.txt" >&2
    echo "Raw CHTHM analysis failed; refusing to mark run as validated" >&2
    exit 51
  }
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

if profile.endswith('_smoke') and __import__('os').environ.get('KV_BAYES_ENABLE') != '1':
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
if __import__('os').environ.get('KV_BAYES_ENABLE') == '1':
    from pathlib import Path
    dyn=json.loads(Path(__import__('os').environ['LMCACHE_GNN_PLACEMENT_METADATA']).read_text())
    assert dyn and all(t in ('gpu','cpu','disk','drop') for t in dyn.values())
    assert (results/'bayes_wave_history.json').exists()
    print('BAYES_WAVE_RUN_PASS', len(dyn))
print('GNN_DYNAMIC_VPC_RUN_PASS')
PY

echo "===== RESULTS ====="
cat "${RESULT_DIR}/summary.json"
echo "placement:"; cat "$PLACEMENT_SUMMARY"
echo "run_dir=$RUN_DIR"
