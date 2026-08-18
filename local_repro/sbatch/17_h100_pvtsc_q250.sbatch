#!/bin/bash
#SBATCH --job-name=kvaware_pvtsc_q250
#SBATCH --partition=gpu-h100-dchen
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --output=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.out
#SBATCH --error=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.err

set -euxo pipefail

REPO=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONFAULTHANDLER=1

# Frozen Q250 / shared-NFS configuration used for the Round-6 R16 boundary run.
# PVTSC is ON by default here, but the same script is also the OFF control:
#   sbatch --export=ALL,SC_LMCACHE_PVTSC_ENABLE=0 \
#     "$REPO/local_repro/sbatch/17_h100_pvtsc_q250.sbatch"
export SC_LMCACHE_PROFILE=custom
export SC_LMCACHE_PVTSC_ENABLE="${SC_LMCACHE_PVTSC_ENABLE:-1}"

export SC_LMCACHE_SCHEDULER_LOOKUP_ADMISSION_ENABLE=1
export SC_LMCACHE_SCHEDULER_LOOKUP_MAX_INFLIGHT=12
export SC_LMCACHE_SCHEDULER_LOOKUP_ADMISSION_WAIT_MS=0
export SC_LMCACHE_WORKER_LOOKUP_ADMISSION_ENABLE=0
export SC_LMCACHE_WORKER_LOOKUP_MAX_INFLIGHT=1
export SC_LMCACHE_DISK_PUT_ADMISSION_ENABLE=0
export SC_LMCACHE_DISK_PUT_MAX_PENDING=8

export SC_LMCACHE_SERIALIZER_FAIRNESS_ENABLE=1
export SC_LMCACHE_SERIALIZER_CPU_BURST_RATIO=16

export SC_LMCACHE_DISK_RESIDENT_PUT_DEDUP_ENABLE=1
export SC_LMCACHE_COLD_WARM_PUT_BARRIER_ENABLE=1
export SC_LMCACHE_DISK_PUT_RESIDENCY_TRACE_ENABLE=1

export SC_LMCACHE_IO_TRACE_ENABLE=1
export SC_LMCACHE_LOAD_TRACE_ENABLE=1
export SC_LMCACHE_MEMORY_TRACE_ENABLE=1
export SC_LMCACHE_MEMORY_SNAPSHOT_ENABLE=0
export SC_LMCACHE_LOOKUP_TRACE_ENABLE=0
export SC_LMCACHE_REQUEST_TRACE_ENABLE=0
export SC_LMCACHE_TIER_TRACE_ENABLE=0
export SC_LMCACHE_GPU_ASSERT_SNAPSHOT_ENABLE=0

export SC_LMCACHE_MAX_LOCAL_CPU_SIZE=20
export MAX_QUESTIONS=250
export SUBMISSION_BATCH_SIZE=250
export VLLM_MAX_NUM_SEQS=16
export DATASET_NAME=medical

# Empty/unset means the driver's normal shared-NFS cache path; never inherit a
# scratch override from an interactive shell.
unset SC_LMCACHE_DATA_DIR

# Source after the explicit exports so this file validates them and fills only
# unrelated defaults. Existing environment values win.
source "$REPO/local_repro/sc_lmcache_knobs.sh"

printf '%s\n' \
  "=== PVTSC Q250 RUN ===" \
  "SC_LMCACHE_PVTSC_ENABLE=$SC_LMCACHE_PVTSC_ENABLE" \
  "MAX_QUESTIONS=$MAX_QUESTIONS" \
  "SUBMISSION_BATCH_SIZE=$SUBMISSION_BATCH_SIZE" \
  "VLLM_MAX_NUM_SEQS=$VLLM_MAX_NUM_SEQS" \
  "SC_LMCACHE_MAX_LOCAL_CPU_SIZE=$SC_LMCACHE_MAX_LOCAL_CPU_SIZE" \
  "SC_LMCACHE_SCHEDULER_LOOKUP_MAX_INFLIGHT=$SC_LMCACHE_SCHEDULER_LOOKUP_MAX_INFLIGHT" \
  "SC_LMCACHE_SERIALIZER_FAIRNESS_ENABLE=$SC_LMCACHE_SERIALIZER_FAIRNESS_ENABLE" \
  "SC_LMCACHE_SERIALIZER_CPU_BURST_RATIO=$SC_LMCACHE_SERIALIZER_CPU_BURST_RATIO" \
  "SC_LMCACHE_COLD_WARM_PUT_BARRIER_ENABLE=$SC_LMCACHE_COLD_WARM_PUT_BARRIER_ENABLE" \
  "SC_LMCACHE_DISK_RESIDENT_PUT_DEDUP_ENABLE=$SC_LMCACHE_DISK_RESIDENT_PUT_DEDUP_ENABLE" \
  "SC_LMCACHE_DATA_DIR=driver-default-shared-NFS"

mkdir -p /mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs
"$REPO/local_repro/run_driver.sh" wo_gnn 2>&1
