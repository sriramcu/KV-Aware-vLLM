#!/usr/bin/env bash
# Thin launcher for the matched H100 Q650 write-admission-cap experiment.
# Run inside an existing 2xH100 srun allocation, or via the companion sbatch.
set -Eeuo pipefail

REPO="${REPO:-/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM}"
ARM="${ARM:-}"
L1_GB="${L1_GB:-300}"
STORE_CAP_GB="${STORE_CAP_GB:-100}"
PF="${PF:-2}"
LOOKUP_PF="${LOOKUP_PF:-16}"
MQ_TIMEOUT="${MQ_TIMEOUT:-120}"
HEARTBEAT_INTERVAL="${HEARTBEAT_INTERVAL:-40}"

case "$ARM" in
  gnn|nognn) ;;
  *)
    echo "ERROR: set ARM=gnn or ARM=nognn" >&2
    exit 2
    ;;
esac

# Matched serving/storage settings selected by the 20923--20928 grid.
export LMCACHE_L1_SIZE_GB="$L1_GB"
export LMCACHE_MP_MAX_INFLIGHT_STORE_GB="$STORE_CAP_GB"
export LMCACHE_L2_PREFETCH_MAX_IN_FLIGHT="$PF"
export LMCACHE_L2_LOOKUP_MAX_IN_FLIGHT="$LOOKUP_PF"
export LMCACHE_MP_TIMEOUT="$MQ_TIMEOUT"
export LMCACHE_MP_HEARTBEAT_INTERVAL="$HEARTBEAT_INTERVAL"
export KV_FS_PER_OP_WORKERS=1
export KV_FS_SHARED_WORKERS=5
export KV_FS_LOOKUP_WORKERS=1
export KV_FS_RETRIEVE_WORKERS=2
export KV_FS_STORE_WORKERS=0
export KV_FS_DELETE_WORKERS=0
export KV_VPC_SUFFICIENT_BYPASS=1
export KV_MUTUAL_PREFIX=1
export KV_STAGE1_STARVATION_FALLBACK=1
export KV_STAGE1_OCCUPANCY_FALLBACK=0
export LMCACHE_MP_PREFIX_DIAGNOSTICS=1
export LMCACHE_MP_CONGESTION_DEBUG=1
export LMCACHE_MP_CHTHM_DEBUG=1
export LMCACHE_MP_STAGE1_FRESHNESS_GUARD_S=270
export KV_COLD_WARM_STORE_BARRIER=1

# Make the chosen cap obvious in stdout even if an older run_config schema is used.
echo "===== Q650 STORE-CAP LAUNCH ====="
echo "arm:                 $ARM"
echo "L1 GiB:              $L1_GB"
echo "max in-flight PUT:   $STORE_CAP_GB GiB"
echo "GET/load PF:        $PF"
echo "lookup PF:           $LOOKUP_PF"
echo "MQ timeout:          $MQ_TIMEOUT s"
echo "heartbeat interval:  $HEARTBEAT_INTERVAL s"
echo "workers:             5 shared + 2 retrieve + 1 lookup"
echo "barrier:             ON"
echo "================================="

case "$ARM" in
  gnn)
    # Experiment requested after 20934: preserve GNN-selected fast-tier chunks
    # in L1, do NOT create full L2 safety backing, and bound only true L2 PUTs.
    export KV_GNN_AWARE_VPC=1
    export KV_GNN_L1_BACKING=1
    export KV_GNN_L2_BACKING=0
    export EXPERIMENT_LABEL="${EXPERIMENT_LABEL:-gnn_mutual_storecap${STORE_CAP_GB}g_l1${L1_GB}_l1back_l2backoff_lpf${LOOKUP_PF}_gpf${PF}_s5_r2}"
    exec bash "$REPO/scripts/run_h100_shortq_gnn_dynamic_vpc_q650.sbatch"
    ;;
  nognn)
    export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.65}"
    export MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
    export ENABLE_VPC=1
    export GRID_LABEL="${GRID_LABEL:-nognn_mutual_storecap${STORE_CAP_GB}g_l1${L1_GB}_lpf${LOOKUP_PF}_gpf${PF}_s5_r2}"
    exec bash "$REPO/scripts/run_h100_l0off_q650_l1_prefetch_grid_cleanall.sbatch"
    ;;
esac
