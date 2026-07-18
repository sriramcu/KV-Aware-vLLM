#!/usr/bin/env bash
set -euo pipefail

# Creates a targeted "gitingest-like" bundle of important KV-Aware-vLLM / LMCache repro files.
# It prints each file path followed by the file's contents into one text file.
#
# Usage:
#   cd /mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
#   bash /path/to/make_kvaware_code_bundle.sh
#
# Output:
#   /mnt/shared/gpfs/home/sriramc2/kvaware_important_code_bundle.txt

HOME_DIR="/mnt/shared/gpfs/home/sriramc2"
REPO="${REPO:-$HOME_DIR/KV-Aware-vLLM}"
OUT="${OUT:-$HOME_DIR/kvaware_important_code_bundle.txt}"

cd "$REPO"

{
  echo "KV-Aware-vLLM / LMCache Important Code Bundle"
  echo "Generated: $(date)"
  echo "Host: $(hostname)"
  echo "User: $(whoami)"
  echo "Repo: $REPO"
  echo
  echo "Git status:"
  git status --short || true
  echo
  echo "Git branch/commit:"
  git branch --show-current || true
  git rev-parse HEAD || true
  echo
  echo "Python / package import paths:"
  python - <<'PY' || true
import sys
print("python:", sys.executable)
for modname in ["vllm", "lmcache", "torch"]:
    try:
        m = __import__(modname)
        print(f"{modname}: file={getattr(m, '__file__', None)} version={getattr(m, '__version__', None)}")
    except Exception as e:
        print(f"{modname}: import_error={e!r}")
PY
  echo
} > "$OUT"

append_file() {
  local f="$1"
  if [[ -f "$f" ]]; then
    {
      echo
      echo "================================================================================"
      echo "FILE: $f"
      echo "================================================================================"
      sed -n '1,2400p' "$f"
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

echo "Adding targeted repo files..."

append_file "local_repro/cpu_offload_lmcache_sriram.py"
append_file "local_repro/run_driver.sh"
append_file "lmcache_hit_hook.py"
append_file "lmcache_config.yaml"

while IFS= read -r f; do
  append_file "$f"
done < <(
  find . -path './.git' -prune -o \
    -type f \( \
      -iname '*cpu*offload*lmcache*.py' -o \
      -iname '*lmcache*cpu*offload*.py' -o \
      -iname '*cpu_offload_lmcache*.py' \
    \) -print | sed 's#^\./##' | sort -u
)

if [[ -d "local_repro/sbatch" ]]; then
  while IFS= read -r f; do
    append_file "$f"
  done < <(find local_repro/sbatch -maxdepth 1 -type f \( -name '*.sbatch' -o -name '*.sh' \) | sort)
fi

append_file "Hierarchical_KV/LinearRAG/build_import_only.py"
append_file "Hierarchical_KV/LinearRAG/make_minimal_predictions.py"
append_file "Hierarchical_KV/LinearRAG/run.py"

append_file "vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration/vllm_v1_adapter.py"
append_file "vllm/v1/importance_registry.py"
append_file "vllm/v1/metrics/loggers.py"

{
  echo
  echo "================================================================================"
  echo "DIRECTORY SNAPSHOT: selected repo paths"
  echo "================================================================================"
  echo
  echo "local_repro:"
  find local_repro -maxdepth 3 -type f | sort || true
  echo
  echo "Hierarchical_KV top-level selected:"
  find Hierarchical_KV -maxdepth 3 -type f \
    ! -name '*.pt' \
    ! -name '*.parquet' \
    ! -name '*.graphml' \
    ! -name '*.jsonl' \
    ! -path '*/results/*' \
    ! -path '*/import/*' \
    | sort | head -400 || true
  echo
} >> "$OUT"

LMCACHE_DIR="$(python - <<'PY' 2>/dev/null || true
import lmcache, os
print(os.path.dirname(lmcache.__file__))
PY
)"

if [[ -n "${LMCACHE_DIR:-}" && -d "$LMCACHE_DIR" ]]; then
  {
    echo
    echo "================================================================================"
    echo "LMCACHE_DIR: $LMCACHE_DIR"
    echo "================================================================================"
    echo
  } >> "$OUT"

  append_file "$LMCACHE_DIR/__init__.py"
  append_file "$LMCACHE_DIR/integration/vllm/vllm_v1_adapter.py"
  append_file "$LMCACHE_DIR/v1/storage_backend/storage_manager.py"
  append_file "$LMCACHE_DIR/v1/memory_management.py"
  append_file "$LMCACHE_DIR/v1/gpu_connector/gpu_connectors.py"
  append_file "$LMCACHE_DIR/v1/cache_engine.py"
fi

SITEDEBUG="$HOME_DIR/venvs/kvaware/lib/python3.12/site-packages/sitecustomize.py"
if [[ -f "$SITEDEBUG" ]]; then
  append_file "$SITEDEBUG"
fi

{
  echo
  echo "================================================================================"
  echo "GREP SUMMARY: SRIRAM debug hooks and key settings"
  echo "================================================================================"
  echo
  grep -R "SRIRAM_REQDBG\|SRIRAM_LOOKUPDBG\|SRIRAM_MONITOR\|SRIRAM_MEMDBG\|VLLM_KV_IMPORTANCE\|max_num_seqs\|gpu_memory_utilization\|max_model_len\|LMCACHE_LOCAL_DISK\|max_local_disk_size\|SRIRAM_LMCACHE_DIR" \
    -n local_repro vllm lmcache_hit_hook.py lmcache_config.yaml 2>/dev/null || true
  if [[ -n "${LMCACHE_DIR:-}" && -d "$LMCACHE_DIR" ]]; then
    grep -R "SRIRAM_REQDBG\|SRIRAM_LOOKUPDBG\|SRIRAM_MONITOR\|SRIRAM_MEMDBG\|assert memory_obj.tensor is not None" \
      -n "$LMCACHE_DIR" 2>/dev/null || true
  fi
  echo
  echo "================================================================================"
  echo "END OF BUNDLE"
  echo "================================================================================"
} >> "$OUT"

echo "Wrote: $OUT"
wc -l "$OUT"
du -h "$OUT"
