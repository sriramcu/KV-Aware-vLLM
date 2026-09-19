#!/usr/bin/env bash
#SBATCH --job-name=kvaware-mp-nognn-med-q250
#SBATCH --partition=gpu-h100-dchen
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --output=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.out
#SBATCH --error=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.err

set -Eeuo pipefail

# ============================================================================
# Paths
# ============================================================================

REPO=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
VENV=/mnt/shared/gpfs/home/sriramc2/venvs/kvaware
CUDA_HOME=/mnt/shared/gpfs/home/sriramc2/cuda-12.8

LINEARRAG_ROOT="${REPO}/Hierarchical_KV/LinearRAG"

RUN_ROOT=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro
RUN_DIR="${RUN_ROOT}/h100_mp_nognn_${SLURM_JOB_ID}"
LOG_DIR="${RUN_DIR}/logs"
RESULT_DIR="${RUN_DIR}/results"

# Shared-NFS L2 storage. CACHE_TAG names the current run namespace.
# like the legacy "start with an empty LMCache" control.
LMCACHE_L2_ROOT="${LMCACHE_L2_ROOT:-/scratch2/sriramc2/lmcache_mp_l2}"
CACHE_TAG="${CACHE_TAG:-medical_q250_nognn_scratch2}"
LMCACHE_L2_DIR="${LMCACHE_L2_ROOT}/${CACHE_TAG}"

# ============================================================================
# Experiment defaults
# ============================================================================

# H100 baseline: Llama-3.3-70B-Instruct on TP2 with FP8 weight quantization.
MODEL="${MODEL:-meta-llama/Llama-3.3-70B-Instruct}"

TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-8000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.65}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"

# Match the legacy project chunk size rather than the 256-token smoke-test size.
LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-512}"
LMCACHE_L1_SIZE_GB="${LMCACHE_L1_SIZE_GB:-32}"
LMCACHE_MAX_WORKERS="${LMCACHE_MAX_WORKERS:-8}"
LMCACHE_L2_WORKERS="${LMCACHE_L2_WORKERS:-8}"
LMCACHE_MP_TIMEOUT="${LMCACHE_MP_TIMEOUT:-30}"

# Fresh state across ALL prior MP cache tags unless explicitly disabled.
CLEAN_OLD_LMCACHE="${CLEAN_OLD_LMCACHE:-1}"

# Dataset-first interface. No QUESTIONS_JSON is needed for normal use.
DATASET_NAME="${DATASET_NAME:-medical}"
NUM_QUESTIONS="${NUM_QUESTIONS:-250}"
RETRIEVAL_TOP_K="${RETRIEVAL_TOP_K:-5}"

REQUEST_ORDER="${REQUEST_ORDER:-legacy_prefix_hash}"
PREFIX_SORT_DEPTH="${PREFIX_SORT_DEPTH:-2}"
REQUEST_ORDER_SEED="${REQUEST_ORDER_SEED:-0}"
WARM_ORDER="${WARM_ORDER:-reverse}"

SUBMISSION_BATCH_SIZE="${SUBMISSION_BATCH_SIZE:-8}"

# Preserve legacy generation settings.
MIN_TOKENS="${MIN_TOKENS:-0}"
MAX_TOKENS="${MAX_TOKENS:-512}"
TOP_P="${TOP_P:-0.95}"

# Real project baseline: VPC enabled. Set ENABLE_VPC=0 only for targeted
# isolation experiments later (especially during L0 validation).
ENABLE_VPC="${ENABLE_VPC:-1}"

# Preserve legacy compiled-mode intent by default. Set to 1 if debugging.
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"

# Optional explicit question-file override. Normally blank.
QUESTIONS_JSON="${QUESTIONS_JSON:-}"

# ============================================================================
# Setup / cleanup
# ============================================================================

mkdir -p "$LOG_DIR" "$RESULT_DIR"

cd "$REPO"
source "${VENV}/bin/activate"

export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/mnt/shared/gpfs/home/sriramc2/.cache/huggingface}"

LMCACHE_PID=""
VLLM_PID=""

cleanup() {
    set +e
    echo "[$(date -Is)] cleanup starting"

    if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        kill -TERM "${VLLM_PID}" 2>/dev/null || true
        for _ in $(seq 1 30); do
            kill -0 "${VLLM_PID}" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi

    if [[ -n "${LMCACHE_PID}" ]] && kill -0 "${LMCACHE_PID}" 2>/dev/null; then
        kill -TERM "${LMCACHE_PID}" 2>/dev/null || true
        for _ in $(seq 1 30); do
            kill -0 "${LMCACHE_PID}" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "${LMCACHE_PID}" 2>/dev/null || true
        wait "${LMCACHE_PID}" 2>/dev/null || true
    fi

    echo "[$(date -Is)] cleanup finished"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

wait_for_tcp() {
    local host="$1"
    local port="$2"
    local pid="$3"
    local timeout_s="$4"
    local start
    start=$(date +%s)

    while true; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "process $pid died while waiting for ${host}:${port}"
            return 1
        fi

        if python - "$host" "$port" <<'PY' >/dev/null 2>&1
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])

sock = socket.socket()
sock.settimeout(0.25)
rc = sock.connect_ex((host, port))
sock.close()

raise SystemExit(0 if rc == 0 else 1)
PY
        then
            return 0
        fi

        if (( $(date +%s) - start >= timeout_s )); then
            echo "timeout waiting for ${host}:${port}"
            return 1
        fi

        sleep 1
    done
}

wait_for_vllm_health() {
    local pid="$1"
    local timeout_s="$2"
    local start
    start=$(date +%s)

    while true; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "vLLM died during startup"
            return 1
        fi

        if curl -sf \
            http://127.0.0.1:8000/health \
            >/dev/null 2>&1
        then
            return 0
        fi

        if (( $(date +%s) - start >= timeout_s )); then
            echo "timeout waiting for vLLM /health"
            return 1
        fi

        sleep 2
    done
}

safe_clean_lmcache_root() {
    local root="${LMCACHE_L2_ROOT%/}"

    if [[ "$CLEAN_OLD_LMCACHE" != "1" ]]; then
        echo "CLEAN_OLD_LMCACHE=${CLEAN_OLD_LMCACHE}; preserving all MP cache state under: $root"
        mkdir -p "$root"
        return 0
    fi

    # Safety guard: only permit deleting inside the dedicated MP cache roots
    # used by this project. This keeps the same function portable to a future
    # H100 /scratch2 sbatch while preventing accidental deletion elsewhere.
    case "$root" in
        /mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/lmcache_mp_l2|\
        /scratch2/sriramc2/lmcache_mp_l2)
            ;;
        *)
            echo "ERROR: refusing to clean unexpected LMCache MP root:"
            echo "  $root"
            exit 12
            ;;
    esac

    echo "Cleaning ALL previous LMCache MP cache state under: $root"
    mkdir -p "$root"

    # mindepth=1 keeps the dedicated root itself while deleting every prior
    # CACHE_TAG, including hidden files/directories.
    find "$root" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +

    echo "LMCache MP root is clean: $root"
}

echo "============================================================"
echo "KV-Aware MP no-GNN project baseline"
echo "job_id:              ${SLURM_JOB_ID}"
echo "node:                $(hostname)"
echo "run_dir:             ${RUN_DIR}"
echo "dataset:             ${DATASET_NAME}"
echo "questions:           ${NUM_QUESTIONS}"
echo "model:               ${MODEL}"
echo "TP:                  ${TENSOR_PARALLEL_SIZE}"
echo "quantization:        online FP8"
echo "VPC enabled:         ${ENABLE_VPC}"
echo "LMCache chunk:       ${LMCACHE_CHUNK_SIZE}"
echo "LMCache L1 GiB:      ${LMCACHE_L1_SIZE_GB}"
echo "LMCache L2 dir:      ${LMCACHE_L2_DIR}"
echo "clean old LMCache:   ${CLEAN_OLD_LMCACHE}"
echo "============================================================"

# ============================================================================
# Preflight sanity checks
# ============================================================================

echo
echo "===== PREFLIGHT: NODE / GPU ====="
date -Is
hostname
free -h
df -h /scratch2 2>/dev/null || true

nvidia-smi \
    --query-gpu=index,name,driver_version,memory.total,memory.free,compute_cap \
    --format=csv,noheader || nvidia-smi

echo
echo "===== PREFLIGHT: REPOSITORY ====="
git branch --show-current
git rev-parse HEAD
git status --short || true

if [[ -f third_party/LMCache.UPSTREAM_COMMIT ]]; then
    echo -n "LMCache upstream commit: "
    cat third_party/LMCache.UPSTREAM_COMMIT
fi

echo
echo "===== PREFLIGHT: PYTHON / CUDA / NATIVE IMPORTS ====="
python - <<'PY'
import inspect
import torch
import vllm
import vllm._C
import lmcache
from lmcache.integration.vllm.vllm_multi_process_adapter import (
    LMCacheMPSchedulerAdapter,
    LMCacheMPWorkerAdapter,
    LoadStoreOp,
)

print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))

print("vllm:", vllm.__file__)
print("vllm._C:", vllm._C.__file__)
print("lmcache:", lmcache.__file__)
print("LoadStoreOp:", inspect.signature(LoadStoreOp))
print("scheduler_adapter:", inspect.signature(LMCacheMPSchedulerAdapter))
print("worker_adapter:", inspect.signature(LMCacheMPWorkerAdapter))
PY

echo
echo "===== PREFLIGHT: SOURCE COMPATIBILITY ====="

python -m py_compile \
    experiments/mp_nognn_project.py \
    vllm/distributed/kv_transfer/kv_connector/v1/lmcache_mp_connector.py

CONNECTOR=vllm/distributed/kv_transfer/kv_connector/v1/lmcache_mp_connector.py

if grep -q 'LoadStoreOp(block_hashes=' "$CONNECTOR"; then
    echo "ERROR: stale LoadStoreOp(block_hashes=...) API remains"
    exit 20
fi

grep -n -E \
    'LoadStoreOp|maybe_submit_lookup_request' \
    "$CONNECTOR" | head -n 30

echo
echo "===== PREFLIGHT: LINEARRAG RUNTIME ASSETS ====="

DEFAULT_QUESTIONS="${LINEARRAG_ROOT}/dataset/${DATASET_NAME}/questions.json"
PASSAGE_EMBEDDINGS="${LINEARRAG_ROOT}/import/${DATASET_NAME}/passage_embedding.parquet"
EMBEDDING_MODEL_DIR="${LINEARRAG_ROOT}/model/all-mpnet-base-v2"
EMBEDDING_MODEL_WEIGHTS="${EMBEDDING_MODEL_DIR}/model.safetensors"

if [[ -n "$QUESTIONS_JSON" ]]; then
    QUESTION_FILE="$QUESTIONS_JSON"
else
    QUESTION_FILE="$DEFAULT_QUESTIONS"
fi

for required in \
    "$QUESTION_FILE" \
    "$PASSAGE_EMBEDDINGS" \
    "$EMBEDDING_MODEL_WEIGHTS"
do
    if [[ ! -f "$required" ]]; then
        echo "ERROR: required runtime asset is missing: $required"
        exit 21
    fi
    ls -lh "$required"
done

echo
echo "--- vendored large-file manifest ---"
cat Hierarchical_KV/LinearRAG.VENDORED_LARGE_FILES.txt || true

echo
echo "--- ignored large-file presence (informational) ---"
while IFS= read -r rel; do
    [[ -z "$rel" ]] && continue
    path="${LINEARRAG_ROOT}/${rel}"
    if [[ -f "$path" ]]; then
        printf 'PRESENT  %s\n' "$rel"
    else
        printf 'MISSING  %s\n' "$rel"
    fi
done < Hierarchical_KV/LinearRAG.VENDORED_LARGE_FILES.txt

echo
echo "===== PREFLIGHT: PORTS ====="

if ss -ltn | grep -Eq ':5556[[:space:]]|:8000[[:space:]]'; then
    echo "ERROR: port 5556 or 8000 already occupied on $(hostname)"
    ss -ltn | grep -E ':5556|:8000' || true
    exit 22
fi

echo "ports 5556 and 8000 are free"

echo
echo "===== PREFLIGHT: LMCACHE L2 / CLEAN_OLD_LMCACHE ====="

safe_clean_lmcache_root
mkdir -p "$LMCACHE_L2_DIR"
df -h "$LMCACHE_L2_ROOT" || true

cat > "${RUN_DIR}/run_config.json" <<EOF
{
  "job_id": "${SLURM_JOB_ID}",
  "node": "$(hostname)",
  "model": "${MODEL}",
  "tensor_parallel_size": ${TENSOR_PARALLEL_SIZE},
  "quantization": "fp8",
  "max_model_len": ${MAX_MODEL_LEN},
  "gpu_memory_utilization": ${GPU_MEMORY_UTILIZATION},
  "max_num_seqs": ${MAX_NUM_SEQS},
  "enable_vpc": ${ENABLE_VPC},
  "lmcache_chunk_size": ${LMCACHE_CHUNK_SIZE},
  "lmcache_l1_size_gb": ${LMCACHE_L1_SIZE_GB},
  "lmcache_l2_dir": "${LMCACHE_L2_DIR}",
  "dataset_name": "${DATASET_NAME}",
  "num_questions": ${NUM_QUESTIONS},
  "question_file": "${QUESTION_FILE}",
  "request_order": "${REQUEST_ORDER}",
  "warm_order": "${WARM_ORDER}"
}
EOF

# ============================================================================
# Standalone LMCache MP server: CPU L1 + filesystem L2
# ============================================================================

echo
echo "===== START LMCACHE MP SERVER ====="

L2_ADAPTER_JSON=$(
python - "$LMCACHE_L2_DIR" "$LMCACHE_L2_WORKERS" <<'PY'
import json
import sys

base_path = sys.argv[1]
num_workers = int(sys.argv[2])

print(
    json.dumps(
        {
            "type": "fs_native",
            "base_path": base_path,
            "num_workers": num_workers,
            "use_odirect": False,
        }
    )
)
PY
)

lmcache server \
    --host localhost \
    --port 5556 \
    --chunk-size "$LMCACHE_CHUNK_SIZE" \
    --l1-size-gb "$LMCACHE_L1_SIZE_GB" \
    --eviction-policy LRU \
    --max-workers "$LMCACHE_MAX_WORKERS" \
    --l2-adapter "$L2_ADAPTER_JSON" \
    > "${LOG_DIR}/lmcache.log" 2>&1 &

LMCACHE_PID=$!

if ! wait_for_tcp 127.0.0.1 5556 "$LMCACHE_PID" 180; then
    echo "LMCache failed to become ready"
    tail -n 240 "${LOG_DIR}/lmcache.log" || true
    exit 30
fi

echo "LMCache ready: pid=${LMCACHE_PID}"

# ============================================================================
# vLLM server
# ============================================================================

echo
echo "===== START VLLM SERVER ====="

KV_TRANSFER_CONFIG=$(
python - "$LMCACHE_MP_TIMEOUT" <<'PY'
import json
import sys

timeout = float(sys.argv[1])

print(
    json.dumps(
        {
            "kv_connector": "LMCacheMPConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "lmcache.mp.host": "tcp://localhost",
                "lmcache.mp.port": 5556,
                "lmcache.mp.mq_timeout": timeout,
            },
        }
    )
)
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

if [[ "$ENABLE_VPC" == "1" ]]; then
    VLLM_ARGS+=(--enable-prefix-caching)
else
    VLLM_ARGS+=(--no-enable-prefix-caching)
fi

if [[ "$ENFORCE_EAGER" == "1" ]]; then
    VLLM_ARGS+=(--enforce-eager)
fi

vllm "${VLLM_ARGS[@]}" \
    > "${LOG_DIR}/vllm.log" 2>&1 &

VLLM_PID=$!

# First Qwen 32B download / online quantization can take a while.
if ! wait_for_vllm_health "$VLLM_PID" 1500; then
    echo "vLLM failed to become healthy"
    tail -n 300 "${LOG_DIR}/vllm.log" || true
    exit 40
fi

echo "vLLM ready: pid=${VLLM_PID}"

# ============================================================================
# Tiny API sanity before project work
# ============================================================================

echo
echo "===== API SANITY ====="

cat > "${RUN_DIR}/sanity_request.json" <<EOF
{
  "model": "${MODEL}",
  "prompt": "Reply with exactly the word OK.",
  "temperature": 0,
  "max_tokens": 4
}
EOF

curl -fsS \
    http://127.0.0.1:8000/v1/completions \
    -H 'Content-Type: application/json' \
    -d @"${RUN_DIR}/sanity_request.json" \
    > "${RUN_DIR}/sanity_response.json"

python - "${RUN_DIR}/sanity_response.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    result = json.load(f)

print("sanity_id:", result.get("id"))
print("sanity_usage:", result.get("usage"))
print("sanity_text:", repr(result["choices"][0]["text"]))
PY

curl -sf \
    http://127.0.0.1:8000/metrics \
    > "${RUN_DIR}/metrics_before.txt" || true

# ============================================================================
# Actual project workload
#
# Hide CUDA from the retrieval driver so its sentence-transformer work cannot
# consume memory from the already-running vLLM L40 process.
# ============================================================================

echo
echo "===== PROJECT WORKLOAD: NO GNN ====="

DRIVER_ARGS=(
    --dataset_name "$DATASET_NAME"
    --max_questions "$NUM_QUESTIONS"
    --retrieval_top_k "$RETRIEVAL_TOP_K"
    --llm_model "$MODEL"
    --request_order "$REQUEST_ORDER"
    --prefix_sort_depth "$PREFIX_SORT_DEPTH"
    --request_order_seed "$REQUEST_ORDER_SEED"
    --warm_order "$WARM_ORDER"
    --submission_batch_size "$SUBMISSION_BATCH_SIZE"
    --server_url http://127.0.0.1:8000
    --temperature 0
    --top_p "$TOP_P"
    --min_tokens "$MIN_TOKENS"
    --max_tokens "$MAX_TOKENS"
    --output_dir "$RESULT_DIR"
)

if [[ -n "$QUESTIONS_JSON" ]]; then
    DRIVER_ARGS+=(--questions_json "$QUESTIONS_JSON")
fi

if ! CUDA_VISIBLE_DEVICES="" \
    python experiments/mp_nognn_project.py \
    "${DRIVER_ARGS[@]}" \
    > "${LOG_DIR}/driver.log" 2>&1
then
    echo "Project workload failed"
    tail -n 300 "${LOG_DIR}/driver.log" || true
    exit 50
fi

# Give asynchronous store activity a short drain window before inspection.
sleep 5

curl -sf \
    http://127.0.0.1:8000/metrics \
    > "${RUN_DIR}/metrics_after.txt" || true

# ============================================================================
# Post-run cache evidence
# ============================================================================

echo
echo "===== POST-RUN CACHE EVIDENCE ====="

grep -Ei \
    'Stored|Retrieved|Prefetch request completed|L2|lookup|hit' \
    "${LOG_DIR}/lmcache.log" \
    > "${RUN_DIR}/lmcache_cache_events.txt" || true

grep -Ei \
    'External prefix cache hit rate|Prefix cache hit rate|ERROR|WARNING' \
    "${LOG_DIR}/vllm.log" \
    > "${RUN_DIR}/vllm_cache_events.txt" || true

{
    echo "l2_root=${LMCACHE_L2_ROOT}"
    echo "l2_dir=${LMCACHE_L2_DIR}"

    echo -n "current_tag_file_count="
    find "$LMCACHE_L2_DIR" -type f 2>/dev/null | wc -l

    echo -n "current_tag_disk_bytes="
    du -sb "$LMCACHE_L2_DIR" 2>/dev/null | awk '{print $1}'

    echo -n "whole_root_disk_bytes="
    du -sb "$LMCACHE_L2_ROOT" 2>/dev/null | awk '{print $1}'

    echo "sample_files:"
    find "$LMCACHE_L2_DIR" -type f 2>/dev/null | head -n 30 || true
} > "${RUN_DIR}/l2_filesystem_summary.txt"

echo "--- driver summary ---"
cat "${RESULT_DIR}/summary.json"

echo
echo "--- LMCache cache events (tail) ---"
tail -n 100 "${RUN_DIR}/lmcache_cache_events.txt" || true

echo
echo "--- L2 filesystem summary ---"
cat "${RUN_DIR}/l2_filesystem_summary.txt"

echo
echo "===== COMPLETE ====="
echo "run directory:"
echo "$RUN_DIR"
