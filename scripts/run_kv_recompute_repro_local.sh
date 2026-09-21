#!/usr/bin/env bash
set -Eeuo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL="${MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
WORK="${WORK:-${TMPDIR:-/tmp}/kv-recompute-repro-${USER}-$$}"
EXAMPLE="$REPO/examples/disaggregated/kv_load_failure_recovery_offline"
mkdir -p "$WORK"
cd "$WORK"
rm -rf local_storage

export PYTHONPATH="$EXAMPLE:${PYTHONPATH:-}"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_KV_RECOMPUTE_DEBUG=1

# Respect Slurm's CUDA_VISIBLE_DEVICES when running under Slurm.
# For manual/non-Slurm use, GPU may optionally select a device.
if [[ -n "${GPU:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

python - <<'PY'
import os
import torch

print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.cuda.device_count() =", torch.cuda.device_count())

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1

print("allocated GPU:", torch.cuda.get_device_name(0))

x = torch.ones(1, device="cuda:0")
torch.cuda.synchronize()
print("single-GPU Slurm allocation smoke: PASS")
PY

python "$EXAMPLE/prefill_example.py" --model "$MODEL" --storage local_storage
python "$EXAMPLE/decode_example.py" --model "$MODEL" --storage local_storage \
  --no-connector --no-async-scheduling --output oracle_full_recompute.json

# Rejected synchronous load, synchronous scheduler.
python "$EXAMPLE/decode_example.py" --model "$MODEL" --storage local_storage \
  --simulate-failure --no-async-scheduling --output recovered_sync_sched.json

# Rejected synchronous load while scheduler overlaps batches. This is the key
# regression cell for the 20541-style V2 optimistic-state rewind.
python "$EXAMPLE/decode_example.py" --model "$MODEL" --storage local_storage \
  --simulate-failure --async-scheduling --output recovered_async_sched.json

# Cover asynchronous KV loading independently of scheduler overlap.
python "$EXAMPLE/decode_example.py" --model "$MODEL" --storage local_storage \
  --simulate-failure --async-load --no-async-scheduling \
  --output recovered_async_load_sync_sched.json

# Cover the full async-load + async-scheduling overlap.
python "$EXAMPLE/decode_example.py" --model "$MODEL" --storage local_storage \
  --simulate-failure --async-load --async-scheduling \
  --output recovered_async_load_async_sched.json

python "$EXAMPLE/compare_recovery.py" oracle_full_recompute.json \
  recovered_sync_sched.json recovered_async_sched.json \
  recovered_async_load_sync_sched.json \
  recovered_async_load_async_sched.json

echo "PASS: all forced KV-load-recovery cells matched full recomputation"
echo "Artifacts: $WORK"
