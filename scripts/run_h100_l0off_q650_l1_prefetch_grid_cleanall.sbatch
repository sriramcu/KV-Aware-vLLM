#!/usr/bin/env bash
#SBATCH --job-name=kvaware-l0off-grid
#SBATCH --partition=gpu-h100-dchen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.out
#SBATCH --error=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.err

set -Eeuo pipefail

# ============================================================================
# Paths
# ============================================================================

REPO=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
VENV=/mnt/shared/gpfs/home/sriramc2/venvs/kvaware-vllm029
CUDA_HOME=/usr/local/cuda-13.1

LINEARRAG_ROOT="${REPO}/Hierarchical_KV/LinearRAG"

RUN_ROOT=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro
GRID_LABEL="${GRID_LABEL:-l1_${LMCACHE_L1_SIZE_GB:-40}_pf_${LMCACHE_L2_PREFETCH_MAX_IN_FLIGHT:-8}}"
RUN_DIR="${RUN_ROOT}/h100_l0off_grid_${GRID_LABEL}_${SLURM_JOB_ID}"
LOG_DIR="${RUN_DIR}/logs"
RESULT_DIR="${RUN_DIR}/results"

# Each grid cell starts from a fully clean user scratch tree.  These jobs are
# intentionally run serially, so it is safe and desirable to remove every
# previous cache/artifact beneath this exact guarded root before the run.
SCRATCH_USER_ROOT="/scratch2/sriramc2"
LMCACHE_L2_ROOT="${SCRATCH_USER_ROOT}/lmcache_mp_l2"
LMCACHE_L2_DIR="${LMCACHE_L2_ROOT}/medical_q650_dup_vanilla_mp"

# ============================================================================
# Experiment defaults
# ============================================================================

# H100 baseline: Llama-3.3-70B-Instruct on TP2 with FP8 weight quantization.
MODEL="${MODEL:-meta-llama/Llama-3.3-70B-Instruct}"

TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-8000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.65}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
LMCACHE_L2_PREFETCH_MAX_IN_FLIGHT="${LMCACHE_L2_PREFETCH_MAX_IN_FLIGHT:-8}"

# Match the legacy project chunk size rather than the 256-token smoke-test size.
LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-512}"
LMCACHE_L1_SIZE_GB="${LMCACHE_L1_SIZE_GB:-40}"
LMCACHE_MAX_WORKERS="${LMCACHE_MAX_WORKERS:-8}"
LMCACHE_L2_WORKERS="${LMCACHE_L2_WORKERS:-8}"
LMCACHE_MP_TIMEOUT="${LMCACHE_MP_TIMEOUT:-30}"

# Dataset-first interface. No QUESTIONS_JSON is needed for normal use.
DATASET_NAME="${DATASET_NAME:-medical}"
NUM_QUESTIONS="${NUM_QUESTIONS:-650}"
RETRIEVAL_TOP_K="${RETRIEVAL_TOP_K:-5}"

REQUEST_ORDER="${REQUEST_ORDER:-legacy_prefix_hash}"
PREFIX_SORT_DEPTH="${PREFIX_SORT_DEPTH:-2}"
REQUEST_ORDER_SEED="${REQUEST_ORDER_SEED:-0}"
WARM_ORDER="${WARM_ORDER:-reverse}"

SUBMISSION_BATCH_SIZE="${SUBMISSION_BATCH_SIZE:-650}"

# Preserve legacy generation settings.
MIN_TOKENS="${MIN_TOKENS:-128}"
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

# This project does not intentionally use NVIDIA MPS.  Point CUDA at a unique,
# deliberately nonexistent MPS pipe directory so CUDA initialization falls
# back to ordinary direct CUDA rather than connecting to the node-wide MPS
# control daemon.  Do NOT create this directory.
export CUDA_MPS_PIPE_DIRECTORY="/tmp/kvaware-no-mps-${USER}-${SLURM_JOB_ID}-DO-NOT-CREATE"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/mnt/shared/gpfs/home/sriramc2/.cache/huggingface}"

# Post-fix recovery/accounting instrumentation. These are intentionally scoped
# to anomaly/recovery paths; L1 normal-lifetime logging only emits for reads held
# longer than the threshold. Set any of these to 0 to disable after validation.
export VLLM_KV_RECOMPUTE_DEBUG="${VLLM_KV_RECOMPUTE_DEBUG:-1}"
export LMCACHE_KV_ACCOUNTING_DEBUG="${LMCACHE_KV_ACCOUNTING_DEBUG:-1}"
export LMCACHE_PREFETCH_LIFETIME_DEBUG="${LMCACHE_PREFETCH_LIFETIME_DEBUG:-1}"
export LMCACHE_PREFETCH_LIFETIME_WARN_S="${LMCACHE_PREFETCH_LIFETIME_WARN_S:-30}"

# This grid is explicitly the L0-OFF baseline.  The L0 server feature is
# disabled by default because this script never passes --l0-enable; also
# neutralize the temporary smoke-store environment flag if inherited.
export LMCACHE_L0_SMOKE_STORE=0

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

safe_clean_scratch_user_root() {
    # Destructive by design, but only for the exact known per-user scratch root.
    # Grid jobs are submitted serially and each cell must start from an empty L2.
    if [[ "$SCRATCH_USER_ROOT" != "/scratch2/sriramc2" ]]; then
        echo "ERROR: refusing clean-all for unexpected scratch root: $SCRATCH_USER_ROOT"
        exit 12
    fi

    mkdir -p "$SCRATCH_USER_ROOT"
    echo "Cleaning all previous contents beneath: $SCRATCH_USER_ROOT"
    find "$SCRATCH_USER_ROOT" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
    mkdir -p "$LMCACHE_L2_DIR"
    echo "Recreated MP L2 directory:"
    echo "  $LMCACHE_L2_DIR"
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
echo "MP prefetch inflight: ${LMCACHE_L2_PREFETCH_MAX_IN_FLIGHT}"
echo "LMCache L2 dir:      ${LMCACHE_L2_DIR}"
echo "scratch cleanup:     FULL CLEAN of /scratch2/sriramc2 before run"
echo "grid label:          ${GRID_LABEL}"
echo "L0:                  OFF"
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
echo "===== PREFLIGHT: GPU EXCLUSIVITY ====="

# KV-Aware explicitly bypasses MPS via CUDA_MPS_PIPE_DIRECTORY above.
#
# A tiny foreign MPS server can therefore be tolerated as idle daemon residue,
# but a large MPS footprint may represent active hidden MPS clients and would
# still contaminate a performance run.  Any foreign non-MPS CUDA process is
# always fatal.
FOREIGN_MPS_MAX_MIB="${FOREIGN_MPS_MAX_MIB:-256}"

check_gpu_exclusivity() {
    local current_job="${SLURM_JOB_ID:-}"
    local current_user
    current_user="$(id -un)"

    if [[ -z "$current_job" ]]; then
        echo "ERROR: SLURM_JOB_ID is not set"
        return 1
    fi

    echo "current Slurm job: $current_job"
    echo "current user:      $current_user"
    echo "MPS bypass path:   ${CUDA_MPS_PIPE_DIRECTORY}"
    echo "idle MPS ceiling:  ${FOREIGN_MPS_MAX_MIB} MiB per GPU entry"
    echo

    if [[ -e "${CUDA_MPS_PIPE_DIRECTORY}" ]]; then
        echo "ERROR: MPS bypass path unexpectedly exists:"
        echo "       ${CUDA_MPS_PIPE_DIRECTORY}"
        return 44
    fi

    local bad=0
    local found=0
    local foreign_mps=0
    local foreign_clients=0

    while IFS=',' read -r pid process gpu_uuid used_mem; do
        pid="$(xargs <<<"$pid")"
        process="$(xargs <<<"$process")"
        gpu_uuid="$(xargs <<<"$gpu_uuid")"
        used_mem="$(xargs <<<"$used_mem")"

        [[ -z "$pid" ]] && continue
        found=1

        local owner cgroup
        owner="$(ps -o user= -p "$pid" 2>/dev/null | xargs || true)"
        cgroup="$(cat "/proc/$pid/cgroup" 2>/dev/null || true)"

        echo "GPU compute process detected:"
        echo "  pid:       $pid"
        echo "  owner:     ${owner:-UNKNOWN}"
        echo "  process:   $process"
        echo "  gpu:       $gpu_uuid"
        echo "  memory:    ${used_mem} MiB"
        echo "  cgroup:    ${cgroup:-UNREADABLE}"

        if grep -q "/job_${current_job}/" <<<"$cgroup"; then
            echo "  verdict:   OK -- belongs to current job"

        elif [[ "$process" == *"nvidia-cuda-mps-server"* ]]; then
            foreign_mps=$((foreign_mps + 1))
            if [[ "$used_mem" =~ ^[0-9]+$ ]] && \
               (( used_mem <= FOREIGN_MPS_MAX_MIB )); then
                echo "  verdict:   WARNING -- small foreign MPS server; bypassed"
            else
                echo "  verdict:   FAIL -- foreign MPS footprint exceeds idle ceiling"
                bad=1
            fi

        elif [[ "$owner" == "root" ]]; then
            echo "  verdict:   WARNING -- root/system GPU process"

        else
            echo "  verdict:   FAIL -- foreign CUDA client"
            foreign_clients=$((foreign_clients + 1))
            bad=1
        fi

        echo
    done < <(
        nvidia-smi \
            --query-compute-apps=pid,process_name,gpu_uuid,used_memory \
            --format=csv,noheader,nounits 2>/dev/null || true
    )

    if [[ "$found" -eq 0 ]]; then
        echo "No CUDA compute processes detected before experiment startup."
    fi

    echo "foreign MPS entries:     $foreign_mps"
    echo "foreign CUDA clients:    $foreign_clients"

    if [[ "$bad" -ne 0 ]]; then
        echo
        echo "============================================================"
        echo "ERROR: GPU EXCLUSIVITY CHECK FAILED"
        echo
        echo "A foreign direct CUDA client or non-idle MPS workload is"
        echo "already present on the allocated GPU node."
        echo "============================================================"
        echo
        echo "===== NVIDIA-SMI AT REJECTION ====="
        nvidia-smi || true
        echo
        echo "===== MPS PROCESS VIEW AT REJECTION ====="
        ps -eo user,pid,ppid,etimes,cmd \
            | grep -E "nvidia-cuda-mps-(control|server)" \
            | grep -v grep || true
        echo
        echo "===== CURRENT NODE SLURM VIEW ====="
        squeue -w "$(hostname -s)" \
            -o "%.18i %.12u %.9P %.10T %.10M %.6D %R" || true
        return 42
    fi

    if [[ "$foreign_mps" -gt 0 ]]; then
        echo
        echo "WARNING: small foreign MPS server residue detected."
        echo "Continuing because KV-Aware explicitly bypasses MPS and no"
        echo "foreign direct CUDA client or large MPS footprint is visible."
    else
        echo "GPU exclusivity preflight passed cleanly."
    fi

    return 0
}

check_gpu_exclusivity || exit $?

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
echo "===== PREFLIGHT: CUDA TOOLKIT / JIT ====="
echo "CUDA_HOME=${CUDA_HOME}"
echo "nvcc=$(command -v nvcc || true)"

if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
    echo "ERROR: nvcc missing at ${CUDA_HOME}/bin/nvcc"
    exit 45
fi

"${CUDA_HOME}/bin/nvcc" --version

echo
echo "===== PREFLIGHT: PYTHON / CUDA / NATIVE IMPORTS ====="
echo "CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY}"

if [[ -e "${CUDA_MPS_PIPE_DIRECTORY}" ]]; then
    echo "ERROR: MPS bypass path unexpectedly exists: ${CUDA_MPS_PIPE_DIRECTORY}"
    exit 44
fi

# Keep a 120-second backstop around the first real CUDA initialization and
# native-import preflight.  The tiny tensor allocation/synchronize on each
# visible GPU proves that direct CUDA is usable before the expensive workload
# starts.
set +e
timeout --signal=TERM --kill-after=10s 120s python -u - <<'PY'
import inspect
import os
import torch
import vllm
import vllm._C_stable_libtorch as vllm_C
import lmcache
from lmcache.integration.vllm.vllm_multi_process_adapter import (
    LMCacheMPSchedulerAdapter,
    LMCacheMPWorkerAdapter,
    LoadStoreOp,
)

print("torch:", torch.__version__, flush=True)
print("torch.version.cuda:", torch.version.cuda, flush=True)
print("CUDA_MPS_PIPE_DIRECTORY:",
      os.environ.get("CUDA_MPS_PIPE_DIRECTORY"), flush=True)
print("PYTORCH_NVML_BASED_CUDA_CHECK:",
      os.environ.get("PYTORCH_NVML_BASED_CUDA_CHECK"), flush=True)
print("cuda_initialized_before:", torch.cuda.is_initialized(), flush=True)
print("cuda_available:", torch.cuda.is_available(), flush=True)
print("cuda_initialized_after_available:",
      torch.cuda.is_initialized(), flush=True)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

device_count = torch.cuda.device_count()
print("device_count:", device_count, flush=True)
if device_count < 2:
    raise RuntimeError(f"expected at least 2 visible GPUs, got {device_count}")

for i in range(2):
    print(f"gpu[{i}]:", torch.cuda.get_device_name(i), flush=True)
    print(f"capability[{i}]:", torch.cuda.get_device_capability(i), flush=True)
    x = torch.zeros(1, device=f"cuda:{i}")
    torch.cuda.synchronize(i)
    print(f"direct_cuda_smoke[{i}]: PASS", flush=True)
    del x

print("vllm:", vllm.__file__, flush=True)
print("vllm native:", vllm_C.__file__, flush=True)
print("lmcache:", lmcache.__file__, flush=True)
print("LoadStoreOp:", inspect.signature(LoadStoreOp), flush=True)
print("scheduler_adapter:", inspect.signature(LMCacheMPSchedulerAdapter), flush=True)
print("worker_adapter:", inspect.signature(LMCacheMPWorkerAdapter), flush=True)
print("DIRECT CUDA / NATIVE PREFLIGHT PASSED", flush=True)
PY
cuda_preflight_rc=$?
set -e

if [[ "$cuda_preflight_rc" -eq 124 || "$cuda_preflight_rc" -eq 137 ]]; then
    echo
    echo "============================================================"
    echo "ERROR: CUDA/NATIVE PREFLIGHT TIMED OUT AFTER 120 SECONDS"
    echo
    echo "Direct CUDA initialization/device access did not complete."
    echo "Refusing to consume the H100 allocation."
    echo "============================================================"
    nvidia-smi || true
    echo
    ps -eo user,pid,ppid,etimes,cmd \
        | grep -E "nvidia-cuda-mps-(control|server)" \
        | grep -v grep || true
    exit 43
elif [[ "$cuda_preflight_rc" -ne 0 ]]; then
    echo "ERROR: CUDA/native import preflight failed with rc=${cuda_preflight_rc}"
    exit "$cuda_preflight_rc"
fi

echo
echo "===== PREFLIGHT: SOURCE COMPATIBILITY ====="

python -m py_compile \
    experiments/mp_nognn_project.py \
    third_party/LMCache/lmcache/integration/vllm/lmcache_mp_connector.py

CONNECTOR=third_party/LMCache/lmcache/integration/vllm/lmcache_mp_connector.py

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
echo "===== PREFLIGHT: LMCACHE L2 / FULL CLEAN ====="

safe_clean_scratch_user_root
df -h "$LMCACHE_L2_ROOT" || true

cat > "${RUN_DIR}/run_config.json" <<EOF
{
  "job_id": "${SLURM_JOB_ID}",
  "grid_label": "${GRID_LABEL}",
  "l0_enabled": false,
  "node": "$(hostname)",
  "model": "${MODEL}",
  "tensor_parallel_size": ${TENSOR_PARALLEL_SIZE},
  "quantization": "fp8",
  "max_model_len": ${MAX_MODEL_LEN},
  "gpu_memory_utilization": ${GPU_MEMORY_UTILIZATION},
  "max_num_seqs": ${MAX_NUM_SEQS},
  "lmcache_l2_prefetch_max_in_flight": ${LMCACHE_L2_PREFETCH_MAX_IN_FLIGHT},
  "enable_vpc": ${ENABLE_VPC},
  "cuda_mps_bypass": true,
  "cuda_mps_pipe_directory": "${CUDA_MPS_PIPE_DIRECTORY}",
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
    --l2-prefetch-max-in-flight "$LMCACHE_L2_PREFETCH_MAX_IN_FLIGHT" \
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
            "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector",
            "kv_role": "kv_both",
            "kv_load_failure_policy": "recompute",
            "kv_connector_extra_config": {
                "lmcache.mp.host": "tcp://localhost",
                "lmcache.mp.port": 5556,
                "lmcache.mp.mq_timeout": timeout,
                "lmcache.mp.lookup_timeout": 0.0,
                "lmcache.mp.starvation_fallback": True,
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
# consume memory from the already-running vLLM H100 process.
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
    --request_timeout_s 7200
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
