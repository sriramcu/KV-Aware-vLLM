#!/usr/bin/env bash
#SBATCH --job-name=kvaware-l0-smoke
#SBATCH --partition=gpu-h100-dchen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.out
#SBATCH --error=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.err

set -Eeuo pipefail

REPO=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
VENV=/mnt/shared/gpfs/home/sriramc2/venvs/kvaware-vllm029
CUDA_HOME=/usr/local/cuda-13.1
LINEARRAG_ROOT="${REPO}/Hierarchical_KV/LinearRAG"
RUN_ROOT=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro
RUN_DIR="${RUN_ROOT}/h100_l0_smoke_${SLURM_JOB_ID}"
LOG_DIR="${RUN_DIR}/logs"
RESULT_DIR="${RUN_DIR}/results"

SCRATCH_USER_ROOT=/scratch2/sriramc2
LMCACHE_L2_ROOT="${LMCACHE_L2_ROOT:-${SCRATCH_USER_ROOT}/lmcache_mp_l2}"
CACHE_TAG="${CACHE_TAG:-l0_smoke_${SLURM_JOB_ID}}"
LMCACHE_L2_DIR="${LMCACHE_L2_ROOT}/${CACHE_TAG}"

MODEL="${MODEL:-meta-llama/Llama-3.3-70B-Instruct}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8000}"
# Smoke-only budget: reduce native-vLLM share while reserving explicit Q for L0.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.60}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-6}"

LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-512}"
LMCACHE_L0_CAPACITY_GB="${LMCACHE_L0_CAPACITY_GB:-4}"
LMCACHE_L1_SIZE_GB="${LMCACHE_L1_SIZE_GB:-2}"
LMCACHE_MAX_WORKERS="${LMCACHE_MAX_WORKERS:-4}"
LMCACHE_L2_WORKERS="${LMCACHE_L2_WORKERS:-4}"
LMCACHE_MP_TIMEOUT="${LMCACHE_MP_TIMEOUT:-30}"
LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-18080}"
LMCACHE_PROM_PORT="${LMCACHE_PROM_PORT:-19090}"

# Deterministic smoke placement: chunks 0,1,4,5,... -> L0; 2,3,6,7,... -> L1/L2.
L0_SMOKE_MOD="${L0_SMOKE_MOD:-4}"
L0_SMOKE_TAKE="${L0_SMOKE_TAKE:-2}"
SMOKE_EXPECT_EVICTION="${SMOKE_EXPECT_EVICTION:-0}"

DATASET_NAME="${DATASET_NAME:-medical}"
NUM_QUESTIONS="${NUM_QUESTIONS:-6}"
RETRIEVAL_TOP_K="${RETRIEVAL_TOP_K:-5}"
REQUEST_ORDER="${REQUEST_ORDER:-legacy_prefix_hash}"
PREFIX_SORT_DEPTH="${PREFIX_SORT_DEPTH:-2}"
REQUEST_ORDER_SEED="${REQUEST_ORDER_SEED:-0}"
WARM_ORDER="${WARM_ORDER:-reverse}"
SUBMISSION_BATCH_SIZE="${SUBMISSION_BATCH_SIZE:-6}"
MIN_TOKENS="${MIN_TOKENS:-32}"
MAX_TOKENS="${MAX_TOKENS:-64}"
TOP_P="${TOP_P:-0.95}"
REQUEST_TIMEOUT_S="${REQUEST_TIMEOUT_S:-1800}"
# IMPORTANT: this integration smoke must force Stage-2 external retrieval.
# If native vLLM prefix caching remains enabled, the immediate warm replay can
# satisfy the same prompts from vLLM's own GPU block cache. LMCache Stage-1 may
# still report L0/L1 hits, but the scheduler then computes need_to_load=0 and no
# L0 -> vLLM D2D is exercised. Keep VPC disabled by default for this smoke.
ENABLE_VPC="${ENABLE_VPC:-0}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"

mkdir -p "$LOG_DIR" "$RESULT_DIR" "$LMCACHE_L2_ROOT"
rm -rf -- "$LMCACHE_L2_DIR"
mkdir -p "$LMCACHE_L2_DIR"

cd "$REPO"
source "${VENV}/bin/activate"

export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_MPS_PIPE_DIRECTORY="/tmp/kvaware-no-mps-${USER}-${SLURM_JOB_ID}-DO-NOT-CREATE"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/mnt/shared/gpfs/home/sriramc2/.cache/huggingface}"

# The smoke-only patch reads these in the LMCache MP server process.
export LMCACHE_L0_SMOKE_STORE=1
export LMCACHE_L0_SMOKE_MOD="$L0_SMOKE_MOD"
export LMCACHE_L0_SMOKE_TAKE="$L0_SMOKE_TAKE"
export LMCACHE_L0_SMOKE_TRACE_D2D=1

# Keep the current recovery instrumentation and production-style fallback.
export VLLM_KV_RECOMPUTE_DEBUG="${VLLM_KV_RECOMPUTE_DEBUG:-1}"
export LMCACHE_KV_ACCOUNTING_DEBUG="${LMCACHE_KV_ACCOUNTING_DEBUG:-1}"
export LMCACHE_PREFETCH_LIFETIME_DEBUG="${LMCACHE_PREFETCH_LIFETIME_DEBUG:-1}"
export LMCACHE_PREFETCH_LIFETIME_WARN_S="${LMCACHE_PREFETCH_LIFETIME_WARN_S:-30}"

LMCACHE_PID=""
VLLM_PID=""
cleanup() {
    set +e
    echo "[$(date -Is)] cleanup starting"
    if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        kill -TERM "$VLLM_PID" 2>/dev/null || true
        for _ in $(seq 1 30); do kill -0 "$VLLM_PID" 2>/dev/null || break; sleep 1; done
        kill -KILL "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
    if [[ -n "$LMCACHE_PID" ]] && kill -0 "$LMCACHE_PID" 2>/dev/null; then
        kill -TERM "$LMCACHE_PID" 2>/dev/null || true
        for _ in $(seq 1 30); do kill -0 "$LMCACHE_PID" 2>/dev/null || break; sleep 1; done
        kill -KILL "$LMCACHE_PID" 2>/dev/null || true
        wait "$LMCACHE_PID" 2>/dev/null || true
    fi
    echo "[$(date -Is)] cleanup finished"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

wait_for_tcp() {
    local host="$1" port="$2" pid="$3" timeout_s="$4" start
    start=$(date +%s)
    while true; do
        kill -0 "$pid" 2>/dev/null || return 1
        if python - "$host" "$port" <<'PY' >/dev/null 2>&1
import socket, sys
s = socket.socket(); s.settimeout(0.25)
rc = s.connect_ex((sys.argv[1], int(sys.argv[2]))); s.close()
raise SystemExit(0 if rc == 0 else 1)
PY
        then return 0; fi
        (( $(date +%s) - start < timeout_s )) || return 1
        sleep 1
    done
}

wait_for_http() {
    local url="$1" pid="$2" timeout_s="$3" start
    start=$(date +%s)
    while true; do
        kill -0 "$pid" 2>/dev/null || return 1
        curl -sf "$url" >/dev/null 2>&1 && return 0
        (( $(date +%s) - start < timeout_s )) || return 1
        sleep 1
    done
}

wait_for_vllm() {
    local pid="$1" timeout_s="$2" start
    start=$(date +%s)
    while true; do
        kill -0 "$pid" 2>/dev/null || return 1
        curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && return 0
        (( $(date +%s) - start < timeout_s )) || return 1
        sleep 2
    done
}

cat > "${RUN_DIR}/run_config.json" <<EOF_JSON
{
  "job_id": "${SLURM_JOB_ID}",
  "model": "${MODEL}",
  "tp": ${TENSOR_PARALLEL_SIZE},
  "gpu_memory_utilization": ${GPU_MEMORY_UTILIZATION},
  "l0_capacity_gb_per_rank": ${LMCACHE_L0_CAPACITY_GB},
  "l1_size_gb": ${LMCACHE_L1_SIZE_GB},
  "l0_smoke_mod": ${L0_SMOKE_MOD},
  "l0_smoke_take": ${L0_SMOKE_TAKE},
  "questions": ${NUM_QUESTIONS},
  "smoke_expect_eviction": ${SMOKE_EXPECT_EVICTION},
  "enable_vpc": ${ENABLE_VPC},
  "l2_dir": "${LMCACHE_L2_DIR}"
}
EOF_JSON

echo "===== L0 SMOKE CONFIG ====="
cat "${RUN_DIR}/run_config.json"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free,compute_cap --format=csv,noheader

echo "===== PREFLIGHT ====="
python -m py_compile \
    third_party/LMCache/lmcache/v1/distributed/l0_manager.py \
    third_party/LMCache/lmcache/v1/distributed/storage_manager.py \
    third_party/LMCache/lmcache/v1/multiprocess/modules/lmcache_driven_transfer.py \
    third_party/LMCache/lmcache/v1/multiprocess/object_group_transfer.py \
    experiments/mp_nognn_project.py

grep -q 'L0_SMOKE_STORE' third_party/LMCache/lmcache/v1/multiprocess/modules/lmcache_driven_transfer.py || {
    echo "ERROR: l0 smoke-store patch is not applied"; exit 20;
}

for p in 5556 8000 "$LMCACHE_HTTP_PORT" "$LMCACHE_PROM_PORT"; do
    if ss -ltn | grep -Eq ":${p}[[:space:]]"; then
        echo "ERROR: port ${p} already occupied"; exit 21
    fi
done

df -h "$SCRATCH_USER_ROOT" || true

L2_ADAPTER_JSON=$(python - "$LMCACHE_L2_DIR" "$LMCACHE_L2_WORKERS" <<'PY'
import json, sys
print(json.dumps({
    "type": "fs_native",
    "base_path": sys.argv[1],
    "num_workers": int(sys.argv[2]),
    "use_odirect": False,
}))
PY
)

echo "===== START LMCACHE ====="
lmcache server \
    --host localhost \
    --port 5556 \
    --http-host 127.0.0.1 \
    --http-port "$LMCACHE_HTTP_PORT" \
    --prometheus-port "$LMCACHE_PROM_PORT" \
    --chunk-size "$LMCACHE_CHUNK_SIZE" \
    --l0-enable \
    --l0-capacity-gb "$LMCACHE_L0_CAPACITY_GB" \
    --l1-size-gb "$LMCACHE_L1_SIZE_GB" \
    --eviction-policy LRU \
    --max-workers "$LMCACHE_MAX_WORKERS" \
    --l2-adapter "$L2_ADAPTER_JSON" \
    > "${LOG_DIR}/lmcache.log" 2>&1 &
LMCACHE_PID=$!

if ! wait_for_tcp 127.0.0.1 5556 "$LMCACHE_PID" 180; then
    echo "LMCache ZMQ endpoint failed"; tail -n 250 "${LOG_DIR}/lmcache.log"; exit 30
fi
if ! wait_for_http "http://127.0.0.1:${LMCACHE_HTTP_PORT}/status" "$LMCACHE_PID" 180; then
    echo "LMCache HTTP endpoint failed"; tail -n 250 "${LOG_DIR}/lmcache.log"; exit 31
fi

curl -sf "http://127.0.0.1:${LMCACHE_HTTP_PORT}/status" \
    > "${RUN_DIR}/lmcache_status_before.json"
curl -sf "http://127.0.0.1:${LMCACHE_PROM_PORT}/metrics" \
    > "${RUN_DIR}/lmcache_metrics_before.txt" || true

KV_TRANSFER_CONFIG=$(python - "$LMCACHE_MP_TIMEOUT" <<'PY'
import json, sys
print(json.dumps({
    "kv_connector": "LMCacheMPConnector",
    "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector",
    "kv_role": "kv_both",
    "kv_load_failure_policy": "recompute",
    "kv_connector_extra_config": {
        "lmcache.mp.host": "tcp://localhost",
        "lmcache.mp.port": 5556,
        "lmcache.mp.mq_timeout": float(sys.argv[1]),
        "lmcache.mp.lookup_timeout": 0.0,
        "lmcache.mp.starvation_fallback": True,
    },
}))
PY
)

VLLM_ARGS=(
    serve "$MODEL"
    --host 127.0.0.1
    --port 8000
    --dtype bfloat16
    --quantization fp8
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-num-seqs "$MAX_NUM_SEQS"
    --enable-chunked-prefill
    --disable-hybrid-kv-cache-manager
    --kv-transfer-config "$KV_TRANSFER_CONFIG"
)
if [[ "$ENABLE_VPC" == "1" ]]; then VLLM_ARGS+=(--enable-prefix-caching); else VLLM_ARGS+=(--no-enable-prefix-caching); fi
if [[ "$ENFORCE_EAGER" == "1" ]]; then VLLM_ARGS+=(--enforce-eager); fi

echo "===== START VLLM ====="
vllm "${VLLM_ARGS[@]}" > "${LOG_DIR}/vllm.log" 2>&1 &
VLLM_PID=$!
if ! wait_for_vllm "$VLLM_PID" 1500; then
    echo "vLLM startup failed"; tail -n 300 "${LOG_DIR}/vllm.log"; exit 40
fi

# Tiny sanity request is sub-chunk-sized and should not materially populate L0.
curl -fsS http://127.0.0.1:8000/v1/completions \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${MODEL}\",\"prompt\":\"Reply with OK.\",\"temperature\":0,\"max_tokens\":4}" \
    > "${RUN_DIR}/sanity_response.json"

echo "===== GPU MEMORY BEFORE WORKLOAD ====="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv \
    > "${RUN_DIR}/gpu_processes_before.txt" || true
cat "${RUN_DIR}/gpu_processes_before.txt" || true

echo "===== RUN COLD + WARM L0 SMOKE ====="
CUDA_VISIBLE_DEVICES="" python experiments/mp_nognn_project.py \
    --dataset_name "$DATASET_NAME" \
    --max_questions "$NUM_QUESTIONS" \
    --retrieval_top_k "$RETRIEVAL_TOP_K" \
    --llm_model "$MODEL" \
    --request_order "$REQUEST_ORDER" \
    --prefix_sort_depth "$PREFIX_SORT_DEPTH" \
    --request_order_seed "$REQUEST_ORDER_SEED" \
    --warm_order "$WARM_ORDER" \
    --submission_batch_size "$SUBMISSION_BATCH_SIZE" \
    --server_url http://127.0.0.1:8000 \
    --request_timeout_s "$REQUEST_TIMEOUT_S" \
    --temperature 0 \
    --top_p "$TOP_P" \
    --min_tokens "$MIN_TOKENS" \
    --max_tokens "$MAX_TOKENS" \
    --output_dir "$RESULT_DIR" \
    > "${LOG_DIR}/driver.log" 2>&1

sleep 5
curl -sf "http://127.0.0.1:${LMCACHE_HTTP_PORT}/status" \
    > "${RUN_DIR}/lmcache_status_after.json"
curl -sf "http://127.0.0.1:${LMCACHE_PROM_PORT}/metrics" \
    > "${RUN_DIR}/lmcache_metrics_after.txt" || true
curl -sf http://127.0.0.1:8000/metrics \
    > "${RUN_DIR}/vllm_metrics_after.txt" || true
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv \
    > "${RUN_DIR}/gpu_processes_after.txt" || true

# Compact evidence files for easy upload/review.
grep -E '\[L0_SMOKE_(STORE|D2D|EVICT)\]|Initialized LMCache L0 arena|Prefetch request completed \(L0\+L1\+L2\)' \
    "${LOG_DIR}/lmcache.log" > "${RUN_DIR}/l0_smoke_events.txt" || true
grep -E 'lmcache_mp_l0_|lmcache_mp_lookup_hit_l0_union' \
    "${RUN_DIR}/lmcache_metrics_after.txt" > "${RUN_DIR}/l0_smoke_metrics.txt" || true

python - \
    "${RESULT_DIR}/summary.json" \
    "${RUN_DIR}/lmcache_status_after.json" \
    "${RUN_DIR}/l0_smoke_events.txt" \
    "${RUN_DIR}/vllm_metrics_after.txt" \
    "$NUM_QUESTIONS" \
    "$SMOKE_EXPECT_EVICTION" <<'PY'
import json, re, sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
status = json.loads(Path(sys.argv[2]).read_text())
events = Path(sys.argv[3]).read_text()
vllm_metrics = Path(sys.argv[4]).read_text()
expected_questions = int(sys.argv[5])
expect_eviction = bool(int(sys.argv[6]))

assert summary["cold"]["successful"] == expected_questions, summary["cold"]
assert summary["warm"]["successful"] == expected_questions, summary["warm"]

l0 = status["storage_manager"]["l0_manager"]
assert l0["enabled"] is True, l0
assert l0["initialized_capacity_bytes"] > 0, l0
assert len(l0["ranks"]) == 2, l0
assert l0["objects"] > 0, l0

store_markers = events.count("[L0_SMOKE_STORE]")
d2d_store = events.count("[L0_SMOKE_D2D] direction=D2H")
d2d_load = events.count("[L0_SMOKE_D2D] direction=H2D")
evictions = events.count("[L0_SMOKE_EVICT]")
assert store_markers > 0, "no L0 smoke stores observed"
assert d2d_store > 0, "no device-backed L0 store D2D observed"
assert d2d_load > 0, (
    "no device-backed L0 retrieve D2D observed; if Stage-1 L0 hits exist but "
    "this is zero, verify native vLLM prefix caching is disabled for the smoke"
)

# A real Stage-2 load must also be visible to the scheduler metrics. This
# distinguishes an L0/L1 Stage-1 lookup from an actual external KV transfer.
m = re.search(
    r'vllm:external_prefix_cache_hits_total\{[^}]*\}\s+([0-9.eE+-]+)',
    vllm_metrics,
)
assert m is not None, "external-prefix-hit metric missing from vLLM metrics"
external_hit_tokens = float(m.group(1))
assert external_hit_tokens > 0, (
    "LMCache lookup ran but vLLM consumed zero external KV tokens; "
    "the smoke did not exercise Stage-2 retrieval"
)

if expect_eviction:
    assert evictions > 0, "eviction smoke requested but no L0 eviction observed"

pat = re.compile(r"\((\d+) L0, (\d+) L1, (\d+) L2\)")
triples = [tuple(map(int, m.groups())) for m in pat.finditer(events)]
assert triples, "no completed L0+L1+L2 union lookups logged"
assert any(l0_hits > 0 for l0_hits, _, _ in triples), triples
assert any(l0_hits > 0 and (l1_hits > 0 or l2_hits > 0)
           for l0_hits, l1_hits, l2_hits in triples), (
    "no lookup actually mixed L0 with L1/L2", triples
)

print("L0_SMOKE_PASS")
print("cold_successful=", summary["cold"]["successful"])
print("warm_successful=", summary["warm"]["successful"])
print("l0_objects=", l0["objects"])
print("l0_logical_chunks=", l0["logical_chunks"])
print("l0_used_bytes=", l0["used_bytes"])
print("l0_initialized_capacity_bytes=", l0["initialized_capacity_bytes"])
print("store_markers=", store_markers)
print("d2d_store_markers=", d2d_store)
print("d2d_load_markers=", d2d_load)
print("external_hit_tokens=", external_hit_tokens)
print("eviction_markers=", evictions)
print("union_source_counts=", triples)
PY

echo "===== L0 SMOKE EVENTS ====="
cat "${RUN_DIR}/l0_smoke_events.txt" || true
echo "===== L0 METRICS ====="
cat "${RUN_DIR}/l0_smoke_metrics.txt" || true
echo "===== SUMMARY ====="
cat "${RESULT_DIR}/summary.json"
echo "===== FINAL L0 STATUS ====="
python -m json.tool "${RUN_DIR}/lmcache_status_after.json" | sed -n '/"l0_manager"/,/"l1_manager"/p'
echo "===== COMPLETE ====="
echo "$RUN_DIR"
