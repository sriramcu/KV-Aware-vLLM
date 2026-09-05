#!/usr/bin/env bash
set -euo pipefail

# Creates a targeted, full-content "gitingest-like" bundle for debugging
# KV-Aware-vLLM / vendored LMCache / Hierarchical_KV runs in a new chat.
# Version 7 adds adapter-dispatch, random/ReqMeta importance routing, per-chunk target-tier, and TP/disk-key coverage.
# It intentionally excludes patch/diff/pristine-comparison artifacts from the bundle.
#
# LMCache is expected to live in:
#   $REPO/third_party/LMCache
#
# and to be installed editable from that source tree. The script records the
# actual Python import location, but always bundles the vendored source rather
# than treating site-packages as the authoritative copy.
#
# Usage:
#   cd /mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
#   bash /path/to/make_kvaware_code_bundle_updated_v6.sh
#
# Optional overrides:
#   REPO=/path/to/KV-Aware-vLLM
#   LMCACHE_ROOT=/path/to/vendored/LMCache
#   OUT=/path/to/kvaware_important_code_bundle.txt
#   HOME_DIR=/mnt/shared/gpfs/home/sriramc2
#
# Output default:
#   /mnt/shared/gpfs/home/sriramc2/kvaware_important_code_bundle.txt

HOME_DIR="${HOME_DIR:-/mnt/shared/gpfs/home/sriramc2}"
REPO="${REPO:-$HOME_DIR/KV-Aware-vLLM}"
LMCACHE_ROOT="${LMCACHE_ROOT:-$REPO/third_party/LMCache}"
LMCACHE_PKG="${LMCACHE_PKG:-$LMCACHE_ROOT/lmcache}"
OUT="${OUT:-$HOME_DIR/kvaware_important_code_bundle.txt}"

cd "$REPO"

if [[ ! -d "$LMCACHE_ROOT" || ! -f "$LMCACHE_ROOT/pyproject.toml" ]]; then
  echo "ERROR: vendored LMCache source not found at: $LMCACHE_ROOT" >&2
  exit 1
fi

if [[ ! -d "$LMCACHE_PKG" ]]; then
  echo "ERROR: LMCache Python package not found at: $LMCACHE_PKG" >&2
  exit 1
fi

declare -A APPENDED=()

section() {
  local title="$1"
  {
    echo
    echo "================================================================================"
    echo "$title"
    echo "================================================================================"
  } >> "$OUT"
}

append_file() {
  local f="$1"

  # Avoid duplicate content when a file is reached through both an explicit
  # path and a find/glob.
  if [[ -n "${APPENDED["$f"]+x}" ]]; then
    return 0
  fi
  APPENDED["$f"]=1

  if [[ -f "$f" ]]; then
    {
      echo
      echo "================================================================================"
      echo "FILE: $f"
      echo "================================================================================"
      cat "$f"
      echo
    } >> "$OUT"
  else
    {
      echo
      echo "================================================================================"
      echo "MISSING FILE: $f"
      echo "================================================================================"
      echo
    } >> "$OUT"
  fi
}

append_existing_files_from_find() {
  while IFS= read -r f; do
    [[ -n "$f" ]] && append_file "$f"
  done
}

{
  echo "KV-Aware-vLLM / Vendored LMCache Important Code Bundle"
  echo "Generated: $(date --iso-8601=seconds 2>/dev/null || date)"
  echo "Host: $(hostname)"
  echo "User: $(whoami)"
  echo "Repo: $REPO"
  echo "Vendored LMCache root: $LMCACHE_ROOT"
  echo "Vendored LMCache package: $LMCACHE_PKG"
  echo "Output: $OUT"
  echo

  echo "Parent repository status:"
  git status --short || true
  echo
  echo "Parent repository branch/commit/remotes:"
  git branch --show-current || true
  git rev-parse HEAD || true
  git remote -v || true
  echo
  echo "Submodule status:"
  git submodule status --recursive || true
  echo

  echo "Vendored LMCache Git identity:"
  git -C "$LMCACHE_ROOT" rev-parse --show-toplevel 2>/dev/null || true
  git -C "$LMCACHE_ROOT" branch --show-current 2>/dev/null || true
  git -C "$LMCACHE_ROOT" rev-parse HEAD 2>/dev/null || true
  git -C "$LMCACHE_ROOT" status --short 2>/dev/null || true
  echo

  echo "Python / source import paths / native extensions / versions:"
  REPO="$REPO" LMCACHE_ROOT="$LMCACHE_ROOT" python - <<'PY' || true
import importlib
import os
import sys
from pathlib import Path

repo = Path(os.environ["REPO"]).resolve()
lmcache_root = Path(os.environ["LMCACHE_ROOT"]).resolve()

print("python_executable:", sys.executable)
print("python_version:", sys.version.replace("\n", " "))

for modname in ["vllm", "lmcache", "torch", "transformers"]:
    try:
        module = importlib.import_module(modname)
        path = getattr(module, "__file__", None)
        print(
            f"{modname}: "
            f"file={path} "
            f"version={getattr(module, '__version__', None)}"
        )
        if path:
            resolved = Path(path).resolve()
            if modname == "vllm":
                print("vllm_imports_from_repo:", repo in resolved.parents)
            elif modname == "lmcache":
                print(
                    "lmcache_imports_from_vendored_source:",
                    lmcache_root in resolved.parents,
                )
    except Exception as exc:
        print(f"{modname}: import_error={exc!r}")

for modname in ["vllm._C", "lmcache.c_ops"]:
    try:
        module = importlib.import_module(modname)
        print(f"{modname}: file={getattr(module, '__file__', None)} import=OK")
    except Exception as exc:
        print(f"{modname}: import_error={exc!r}")

try:
    import torch
    print("torch_cuda_version:", torch.version.cuda)
    print("torch_cuda_available:", torch.cuda.is_available())
    print("torch_cuda_device_count:", torch.cuda.device_count())
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        print("torch_cuda_device_0:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch_cuda_probe_error:", repr(exc))

print("selected_environment:")
for name in sorted(os.environ):
    if (
        name.startswith("LMCACHE_")
        or name.startswith("VLLM_")
        or name.startswith("SC_")
        or name.startswith("GRID_")
        or name.startswith("SRIRAM_")
        or name.startswith("DYN_KVBM_")
        or name in {
            "CUDA_VISIBLE_DEVICES",
            "CUDA_DEVICE_ORDER",
            "PYTHONPATH",
            "HF_HOME",
            "HUGGINGFACE_HUB_CACHE",
            "TRANSFORMERS_CACHE",
            "PROMETHEUS_MULTIPROC_DIR",
            "TMPDIR",
            "MODE",
            "DATASET_NAME",
            "MAX_QUESTIONS",
            "SUBMISSION_BATCH_SIZE",
        }
    ):
        print(f"{name}={os.environ.get(name)}")
PY
  echo

  echo "Relevant installed packages:"
  python -m pip show lmcache vllm torch transformers 2>/dev/null || true
  echo
  echo "Editable-install metadata:"
  python -m pip list --editable 2>/dev/null || true
  echo
  echo "Dependency consistency:"
  python -m pip check 2>/dev/null || true
} > "$OUT"

section "STORAGE MOUNT SNAPSHOT"
{
  echo "IMPORTANT: do not infer filesystem type from the /mnt/shared/gpfs path name."
  echo "Home/repo mount identity:"
  findmnt -T "$HOME_DIR" -o TARGET,SOURCE,FSTYPE,OPTIONS 2>/dev/null || true
  df -T "$HOME_DIR" 2>/dev/null || true
  echo

  if [[ -n "${SC_LMCACHE_DATA_DIR:-}" ]]; then
    echo "Active SC_LMCACHE_DATA_DIR=$SC_LMCACHE_DATA_DIR"
    if [[ -e "$SC_LMCACHE_DATA_DIR" ]]; then
      findmnt -T "$SC_LMCACHE_DATA_DIR" -o TARGET,SOURCE,FSTYPE,OPTIONS 2>/dev/null || true
      df -T "$SC_LMCACHE_DATA_DIR" 2>/dev/null || true
    else
      echo "SC_LMCACHE_DATA_DIR does not exist on this host at bundle-generation time."
    fi
  else
    echo "SC_LMCACHE_DATA_DIR is not set in the bundle-generation shell."
  fi
} >> "$OUT"

section "HIGH-LEVEL DIRECTORY STRUCTURE"
cat >> "$OUT" <<EOF
$REPO/
├── local_repro/
│   ├── cpu_offload_lmcache_sriram.py
│   ├── run_driver.sh
│   ├── grid_search/
│   │   ├── run_pressure_grid.py
│   │   ├── rerun_failed_grid_job.py
│   │   ├── run_future_combo_grid.py
│   │   └── submit_*grid.sbatch
│   └── sbatch/
├── vllm/
│   ├── entrypoints/
│   ├── v1/core/sched/
│   ├── v1/engine/
│   └── distributed/kv_transfer/kv_connector/v1/
├── third_party/LMCache/
│   ├── pyproject.toml
│   ├── requirements/
│   └── lmcache/
│       ├── integration/vllm/
│       └── v1/
│           ├── lookup_client/
│           ├── storage_backend/
│           ├── gpu_connector/
│           ├── cache_engine.py
│           └── memory_management.py
├── Hierarchical_KV/
│   ├── LinearRAG/
│   ├── data/
│   └── hierarchical-kv-gnn-3tier-compression/
├── p0_first_half_bundle/
├── lmcache_config.yaml
├── lmcache_hit_hook.py
├── kvcache_monitor.py
└── kvcache_visualize.py

Runtime/build paths:
├── venv: $HOME_DIR/venvs/kvaware
├── run root: $HOME_DIR/runs/kvaware_repro
└── LMCache disk data: SC_LMCACHE_DATA_DIR when explicitly set; otherwise a job-specific directory below the run root

This is intentionally a high-level map, not a recursive listing of every file.
EOF

section "BUILD AND PACKAGING FILES"
append_file "pyproject.toml"
append_file "CMakeLists.txt"
append_file "requirements/build.txt"
append_file "$LMCACHE_ROOT/pyproject.toml"
append_file "$LMCACHE_ROOT/CMakeLists.txt"
append_file "$LMCACHE_ROOT/setup.py"
append_file "$LMCACHE_ROOT/requirements/build.txt"

echo "Adding targeted repository files..."

# ---------------------------------------------------------------------------
# Driver, launch, configuration, monitoring, setup, and local repro variants.
# ---------------------------------------------------------------------------
append_file "local_repro/cpu_offload_lmcache_sriram.py"
append_file "local_repro/run_driver.sh"
if [[ -f "local_repro/sbatch/17_h100_pvtsc_q250.sbatch" ]]; then
  append_file "local_repro/sbatch/17_h100_pvtsc_q250.sbatch"
fi
append_file "lmcache_hit_hook.py"
append_file "kvcache_monitor.py"
append_file "kvcache_visualize.py"
append_file "lmcache_config.yaml"

append_existing_files_from_find < <(
  find local_repro -maxdepth 4 -type f \
    \( -name '*.py' -o -name '*.sh' -o -name '*.sbatch' -o -name '*.yaml' -o -name '*.yml' -o -name '*.md' \) \
    ! -path 'local_repro/lmcache_source_migration/*' \
    ! -iname '*pristine*' \
    ! -iname '*attribution*' \
    | sort
)

# Current source files are preferred over historical patch/diff bundles.

append_existing_files_from_find < <(
  find . -path './.git' -prune -o \
    -type f \( \
      -iname '*cpu*offload*lmcache*.py' -o \
      -iname '*lmcache*cpu*offload*.py' -o \
      -iname '*cpu_offload_lmcache*.py' \
    \) -print | sed 's#^\./##' | sort -u
)

# ---------------------------------------------------------------------------
# Project/GNN/LinearRAG code needed to understand prompt construction,
# GNN labels, tier mapping, and dataset access patterns.
# ---------------------------------------------------------------------------
append_file "Hierarchical_KV/linearrag_gnn_infer.py"
append_file "Hierarchical_KV/rag_query_gnn_predictor.py"
append_file "Hierarchical_KV/LinearRAG/build_import_only.py"
append_file "Hierarchical_KV/LinearRAG/make_minimal_predictions.py"
append_file "Hierarchical_KV/LinearRAG/run.py"

append_existing_files_from_find < <(
  find Hierarchical_KV -maxdepth 3 -type f \
    \( -name '*.py' -o -name '*.yaml' -o -name '*.yml' \) \
    ! -path '*/__pycache__/*' \
    ! -path '*/results/*' \
    ! -path '*/import/*' \
    | sort
)

# ---------------------------------------------------------------------------
# vLLM request submission, scheduler admission, native KV/prefix cache,
# connector lifecycle, and custom importance-tier integration.
# ---------------------------------------------------------------------------
append_file "vllm/entrypoints/llm.py"
append_file "vllm/engine/arg_utils.py"
append_file "vllm/config/scheduler.py"
append_file "vllm/v1/core/sched/scheduler.py"
append_file "vllm/v1/core/sched/request_queue.py"
append_file "vllm/v1/core/kv_cache_manager.py"
append_file "vllm/v1/core/single_type_kv_cache_manager.py"
append_file "vllm/v1/core/kv_cache_utils.py"
for _sc_gpu_residency_path in \
  "vllm/v1/core/block_pool.py" \
  "vllm/v1/core/kv_cache_coordinator.py" \
  "vllm/v1/core/kv_cache_coordinator/__init__.py"; do
  [[ -f "$_sc_gpu_residency_path" ]] && append_file "$_sc_gpu_residency_path"
done
append_file "vllm/v1/engine/core.py"
append_file "vllm/v1/engine/core_client.py"
append_file "vllm/v1/engine/async_llm.py"
append_file "vllm/distributed/kv_transfer/kv_connector/v1/base.py"
append_file "vllm/distributed/kv_transfer/kv_connector/v1/factory.py"
append_file "vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py"
append_file "vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration/vllm_v1_adapter.py"
append_file "vllm/v1/request.py"
append_file "vllm/v1/importance_registry.py"
append_file "vllm/v1/metrics/loggers.py"

if [[ -d "vllm/distributed/kv_transfer/kv_connector/v1" ]]; then
  append_existing_files_from_find < <(
    find vllm/distributed/kv_transfer/kv_connector/v1 \
      -maxdepth 4 -type f -name '*.py' | sort
  )
fi

# ---------------------------------------------------------------------------
# Vendored LMCache source. This is authoritative even when Python import
# metadata is stale or the editable installation is temporarily broken.
# ---------------------------------------------------------------------------
section "VENDORED LMCACHE SOURCE ROOT: $LMCACHE_ROOT"

append_file "$LMCACHE_PKG/__init__.py"
append_file "$LMCACHE_PKG/config.py"
append_file "$LMCACHE_PKG/utils.py"
append_file "$LMCACHE_PKG/integration/vllm/vllm_v1_adapter.py"

append_file "$LMCACHE_PKG/v1/config.py"
append_file "$LMCACHE_PKG/v1/metadata.py"
append_file "$LMCACHE_PKG/v1/cache_engine.py"
append_file "$LMCACHE_PKG/v1/event_manager.py"
append_file "$LMCACHE_PKG/v1/pin_monitor.py"
append_file "$LMCACHE_PKG/v1/memory_management.py"

append_file "$LMCACHE_PKG/v1/lookup_client/factory.py"
append_file "$LMCACHE_PKG/v1/lookup_client/async_lookup_message.py"
append_file "$LMCACHE_PKG/v1/lookup_client/lmcache_async_lookup_client.py"

append_file "$LMCACHE_PKG/v1/storage_backend/__init__.py"
append_file "$LMCACHE_PKG/v1/storage_backend/storage_manager.py"
append_file "$LMCACHE_PKG/v1/storage_backend/local_cpu_backend.py"
append_file "$LMCACHE_PKG/v1/storage_backend/local_disk_backend.py"
append_file "$LMCACHE_PKG/v1/storage_backend/abstract_backend.py"
append_file "$LMCACHE_PKG/v1/storage_backend/storage_backend.py"

append_file "$LMCACHE_PKG/v1/gpu_connector/gpu_connectors.py"

if [[ -d "$LMCACHE_PKG/v1/storage_backend" ]]; then
  append_existing_files_from_find < <(
    find "$LMCACHE_PKG/v1/storage_backend" \
      -maxdepth 3 -type f -name '*.py' \
      ! -path '*/__pycache__/*' \
      ! -path '*/tests/*' \
      | sort
  )
fi

append_existing_files_from_find < <(
  find "$LMCACHE_PKG" -maxdepth 5 -type f \
    \( \
      -iname '*lookup*.py' -o \
      -iname '*prefetch*.py' -o \
      -iname '*serializer*.py' -o \
      -iname '*event*.py' -o \
      -iname '*worker*.py' \
    \) \
    ! -path '*/__pycache__/*' \
    ! -path '*/tests/*' \
    | sort
)

if [[ -d "$LMCACHE_PKG/integration/vllm" ]]; then
  append_existing_files_from_find < <(
    find "$LMCACHE_PKG/integration/vllm" \
      -maxdepth 3 -type f -name '*.py' \
      ! -path '*/__pycache__/*' \
      ! -path '*/tests/*' \
      | sort
  )
fi

# Target tests related to async lookup, storage admission, and local disk.
if [[ -d "$LMCACHE_ROOT/tests" ]]; then
  append_existing_files_from_find < <(
    find "$LMCACHE_ROOT/tests" -maxdepth 6 -type f -name '*.py' \
      \( \
        -iname '*lookup*' -o \
        -iname '*storage*' -o \
        -iname '*disk*' -o \
        -iname '*admission*' -o \
        -iname '*p0*' -o \
        -iname '*pvtsc*' -o \
        -iname '*fair*' \
      \) | sort
  )
fi

# Site-wide debug hooks can materially change behavior/log volume.
SITEDEBUG="$HOME_DIR/venvs/kvaware/lib/python3.12/site-packages/sitecustomize.py"
if [[ -f "$SITEDEBUG" ]]; then
  append_file "$SITEDEBUG"
fi

# ---------------------------------------------------------------------------
# Metadata snapshots and current-path summaries.
# ---------------------------------------------------------------------------
section "CURATED PATH EXISTENCE SNAPSHOT"
{
  for path in \
    "local_repro" \
    "local_repro/sbatch" \
    "local_repro/grid_search" \
    "vllm/v1/core/sched" \
    "vllm/distributed/kv_transfer/kv_connector/v1" \
    "third_party/LMCache" \
    "third_party/LMCache/lmcache/v1/lookup_client" \
    "third_party/LMCache/lmcache/v1/lookup_client/async_lookup_message.py" \
    "third_party/LMCache/lmcache/v1/storage_backend" \
    "local_repro/sbatch/17_h100_pvtsc_q250.sbatch" \
    "Hierarchical_KV" \
    "Hierarchical_KV/LinearRAG" \
    "p0_first_half_bundle"
  do
    if [[ -e "$path" ]]; then
      printf "PRESENT  %s\n" "$path"
    else
      printf "MISSING  %s\n" "$path"
    fi
  done
} >> "$OUT"

section "GREP SUMMARY: lifecycle, adapter dispatch, importance routing, storage placement, and settings"
grep -R \
  "SC_LMCACHE_\|SC_IO_\|SC_LOAD_\|SC_MEMORY_\|SC_SCHEDULER_\|SC_WORKER_\|SC_DISK_PUT_\|SC_DRIVER_\|SC_LMCACHE_DATA_DIR\|PVTSC\|SERIALIZER_FAIRNESS\|CPU_BURST_RATIO\|RESIDENT_PUT_DEDUP\|COLD_WARM_PUT_BARRIER\|readinto\|AsyncPQThreadPoolExecutor\|proc_io_delta\|GRID_RUN_\|GRID_CONFIG_SHA256\|SRIRAM_REQDBG\|SRIRAM_LOOKUPDBG\|SRIRAM_MONITOR\|SRIRAM_MEMDBG\|KVDBG_\|KVIO_\|VLLM_KV_IMPORTANCE\|kv_importance_tiers\|set_block_importance\|allocate_new_computed_blocks\|num_external_computed_tokens\|to_gpu\|multi_layer_kv_transfer\|SC_EXT_WAIT_CRITICAL\|critical_wait\|max_num_seqs\|submission_batch_size\|enable_async_loading\|lookup_timeout_ms\|pin_timeout_sec\|local_disk\|max_local_disk_size\|SRIRAM_LMCACHE_DIR\|LMCACHE_P0_CLIENT_LOOKUP_MAX_INFLIGHT\|LMCACHE_P0_CLIENT_LOOKUP_ADMISSION_TIMEOUT_MS\|LMCACHE_P0_LOOKUP_MAX_INFLIGHT\|LMCACHE_P0_DISK_PUT_MAX_PENDING\|P0_CLIENT_LOOKUP_ADMISSION\|P0_LOOKUP_ADMISSION\|P0_PUT_ADMISSION\|use_native\|Initializing native LMCache connector\|Initializing latest dev LMCache connector\|target_tiers\|_get_target_tiers_for_request\|_put_with_optional_target_tiers\|importance_to_tier\|ReqMeta\|pop_importance\|random.randint\|store_location\|locations=\|location=\|_key_to_path\|CacheEngineKey\|worker_id\|chunk_size\|num_kv_head" \
  -n local_repro vllm Hierarchical_KV third_party/LMCache \
  lmcache_hit_hook.py kvcache_monitor.py kvcache_visualize.py lmcache_config.yaml \
  2>/dev/null >> "$OUT" || true

section "END OF BUNDLE"

echo "Wrote: $OUT"
wc -l "$OUT"
du -h "$OUT"
