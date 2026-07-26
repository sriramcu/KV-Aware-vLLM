#!/usr/bin/env bash
set -euo pipefail

# Creates a targeted, full-content "gitingest-like" bundle for debugging
# KV-Aware-vLLM / LMCache / Hierarchical_KV runs in a new chat.
#
# Usage:
#   cd /mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
#   bash /path/to/make_kvaware_code_bundle_updated.sh
#
# Optional overrides:
#   REPO=/path/to/KV-Aware-vLLM
#   OUT=/path/to/kvaware_important_code_bundle.txt
#   HOME_DIR=/mnt/shared/gpfs/home/sriramc2
#
# Output default:
#   /mnt/shared/gpfs/home/sriramc2/kvaware_important_code_bundle.txt

HOME_DIR="${HOME_DIR:-/mnt/shared/gpfs/home/sriramc2}"
REPO="${REPO:-$HOME_DIR/KV-Aware-vLLM}"
OUT="${OUT:-$HOME_DIR/kvaware_important_code_bundle.txt}"

cd "$REPO"

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
      # Full content is intentional. The old script stopped at line 2400,
      # which omitted relevant scheduler/adapter/backend code.
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
  echo "KV-Aware-vLLM / LMCache Important Code Bundle"
  echo "Generated: $(date --iso-8601=seconds 2>/dev/null || date)"
  echo "Host: $(hostname)"
  echo "User: $(whoami)"
  echo "Repo: $REPO"
  echo "Output: $OUT"
  echo
  echo "Git status:"
  git status --short || true
  echo
  echo "Git branch/commit:"
  git branch --show-current || true
  git rev-parse HEAD || true
  echo
  echo "Python / package import paths and versions:"
  python - <<'PY' || true
import os
import sys

print("python_executable:", sys.executable)
print("python_version:", sys.version.replace("\n", " "))
for modname in ["vllm", "lmcache", "torch", "transformers"]:
    try:
        m = __import__(modname)
        print(
            f"{modname}: "
            f"file={getattr(m, '__file__', None)} "
            f"version={getattr(m, '__version__', None)}"
        )
    except Exception as e:
        print(f"{modname}: import_error={e!r}")

print("selected_environment:")
for name in sorted(os.environ):
    if (
        name.startswith("LMCACHE_")
        or name.startswith("VLLM_")
        or name.startswith("SRIRAM_")
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
} > "$OUT"

echo "Adding targeted repo files..."

# ---------------------------------------------------------------------------
# Driver, launch, configuration, monitoring, and all local repro variants.
# ---------------------------------------------------------------------------
append_file "local_repro/cpu_offload_lmcache_sriram.py"
append_file "local_repro/run_driver.sh"
append_file "lmcache_hit_hook.py"
append_file "kvcache_monitor.py"
append_file "kvcache_visualize.py"
append_file "lmcache_config.yaml"

append_existing_files_from_find < <(
  find local_repro -maxdepth 4 -type f \
    \( -name '*.py' -o -name '*.sh' -o -name '*.sbatch' -o -name '*.yaml' -o -name '*.yml' \) \
    | sort
)

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
append_file "vllm/v1/core/kv_cache_manager.py"
append_file "vllm/v1/core/single_type_kv_cache_manager.py"
append_file "vllm/v1/core/kv_cache_utils.py"
append_file "vllm/v1/engine/core.py"
append_file "vllm/v1/engine/core_client.py"
append_file "vllm/v1/engine/async_llm.py"
append_file "vllm/distributed/kv_transfer/kv_connector/v1/base.py"
append_file "vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration/vllm_v1_adapter.py"
append_file "vllm/v1/importance_registry.py"
append_file "vllm/v1/metrics/loggers.py"

# Include the complete local vLLM KV connector v1 Python surface because
# request IDs, lookup timing, and scheduler/worker cleanup span several files.
if [[ -d "vllm/distributed/kv_transfer/kv_connector/v1" ]]; then
  append_existing_files_from_find < <(
    find vllm/distributed/kv_transfer/kv_connector/v1 \
      -maxdepth 4 -type f -name '*.py' | sort
  )
fi

# ---------------------------------------------------------------------------
# Resolve the *actually imported* LMCache installation and include the full
# code involved in config parsing, lookup/prefetch lifecycle, storage tiers,
# pin/ref ownership, disk worker queues, and GPU transfers.
# ---------------------------------------------------------------------------
LMCACHE_DIR="$(python - <<'PY' 2>/dev/null || true
import lmcache
import os
print(os.path.dirname(lmcache.__file__))
PY
)"

if [[ -n "${LMCACHE_DIR:-}" && -d "$LMCACHE_DIR" ]]; then
  section "LMCACHE_DIR: $LMCACHE_DIR"

  append_file "$LMCACHE_DIR/__init__.py"
  append_file "$LMCACHE_DIR/config.py"
  append_file "$LMCACHE_DIR/integration/vllm/vllm_v1_adapter.py"

  append_file "$LMCACHE_DIR/v1/config.py"
  append_file "$LMCACHE_DIR/v1/cache_engine.py"
  append_file "$LMCACHE_DIR/v1/event_manager.py"
  append_file "$LMCACHE_DIR/v1/pin_monitor.py"
  append_file "$LMCACHE_DIR/v1/memory_management.py"

  append_file "$LMCACHE_DIR/v1/storage_backend/__init__.py"
  append_file "$LMCACHE_DIR/v1/storage_backend/storage_manager.py"
  append_file "$LMCACHE_DIR/v1/storage_backend/local_cpu_backend.py"
  append_file "$LMCACHE_DIR/v1/storage_backend/local_disk_backend.py"
  append_file "$LMCACHE_DIR/v1/storage_backend/abstract_backend.py"
  append_file "$LMCACHE_DIR/v1/storage_backend/storage_backend.py"

  append_file "$LMCACHE_DIR/v1/gpu_connector/gpu_connectors.py"

  # Full storage backend tree: includes backend factory, cache policies,
  # local CPU/disk implementations, and any version-specific helper modules.
  if [[ -d "$LMCACHE_DIR/v1/storage_backend" ]]; then
    append_existing_files_from_find < <(
      find "$LMCACHE_DIR/v1/storage_backend" \
        -maxdepth 3 -type f -name '*.py' \
        ! -path '*/__pycache__/*' \
        ! -path '*/tests/*' \
        | sort
    )
  fi

  # Async lookup server/client, serializers, event helpers, and worker code
  # move between files across LMCache versions. Include all matching modules.
  append_existing_files_from_find < <(
    find "$LMCACHE_DIR" -maxdepth 5 -type f \
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

  # Include all files in the active vLLM integration package, because the
  # scheduler-side and worker-side APIs are split across version-specific code.
  if [[ -d "$LMCACHE_DIR/integration/vllm" ]]; then
    append_existing_files_from_find < <(
      find "$LMCACHE_DIR/integration/vllm" \
        -maxdepth 3 -type f -name '*.py' \
        ! -path '*/__pycache__/*' \
        ! -path '*/tests/*' \
        | sort
    )
  fi
else
  section "LMCACHE IMPORT FAILED OR DIRECTORY MISSING"
  echo "LMCACHE_DIR=${LMCACHE_DIR:-<empty>}" >> "$OUT"
fi

# Site-wide debug hooks can materially change behavior/log volume.
SITEDEBUG="$HOME_DIR/venvs/kvaware/lib/python3.12/site-packages/sitecustomize.py"
if [[ -f "$SITEDEBUG" ]]; then
  append_file "$SITEDEBUG"
fi

# ---------------------------------------------------------------------------
# Metadata snapshots and diffs.
# ---------------------------------------------------------------------------
section "DIRECTORY SNAPSHOT: selected repo paths"
{
  echo "local_repro:"
  find local_repro -maxdepth 4 -type f | sort || true
  echo
  echo "Selected Hierarchical_KV source files:"
  find Hierarchical_KV -maxdepth 3 -type f \
    ! -name '*.pt' \
    ! -name '*.parquet' \
    ! -name '*.graphml' \
    ! -name '*.jsonl' \
    ! -path '*/results/*' \
    ! -path '*/import/*' \
    | sort | head -600 || true
} >> "$OUT"

section "GIT DIFF: relevant tracked source"
git diff -- \
  local_repro \
  Hierarchical_KV \
  lmcache_hit_hook.py \
  kvcache_monitor.py \
  kvcache_visualize.py \
  vllm/distributed/kv_transfer/kv_connector/v1 \
  vllm/v1/core/sched/scheduler.py \
  vllm/v1/importance_registry.py \
  vllm/v1/metrics/loggers.py \
  >> "$OUT" 2>&1 || true

section "GREP SUMMARY: lifecycle, debug hooks, and key settings"
grep -R \
  "SRIRAM_REQDBG\|SRIRAM_LOOKUPDBG\|SRIRAM_MONITOR\|SRIRAM_MEMDBG\|KVDBG_\|VLLM_KV_IMPORTANCE\|max_num_seqs\|submission_batch_size\|enable_async_loading\|lookup_timeout_ms\|pin_timeout_sec\|local_disk\|max_local_disk_size\|SRIRAM_LMCACHE_DIR" \
  -n local_repro vllm Hierarchical_KV \
  lmcache_hit_hook.py kvcache_monitor.py kvcache_visualize.py lmcache_config.yaml \
  2>/dev/null >> "$OUT" || true

if [[ -n "${LMCACHE_DIR:-}" && -d "$LMCACHE_DIR" ]]; then
  grep -R \
    "KVDBG_\|SRIRAM_REQDBG\|SRIRAM_LOOKUPDBG\|prefetch_tasks\|AsyncSingleSerializer\|AsyncMultiSerializer\|WeightedSemaphore\|lookup_timeout\|pin_timeout\|submit_put_task\|ref_count_up\|ref_count_down\|cannot schedule new futures" \
    -n "$LMCACHE_DIR" 2>/dev/null >> "$OUT" || true
fi

section "END OF BUNDLE"

echo "Wrote: $OUT"
wc -l "$OUT"
du -h "$OUT"
