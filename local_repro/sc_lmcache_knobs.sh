#!/usr/bin/env bash
# Source this file from an sbatch script. Values already present in the
# environment win, so `sbatch --export=ALL,SC_...=...` remains suitable for
# grid searches.

_sc_default() {
  local name="$1"
  local value="$2"
  if [[ ! -v "$name" ]]; then
    printf -v "$name" '%s' "$value"
    export "$name"
  fi
}

_sc_bool() {
  case "${1,,}" in
    0|1|false|true|no|yes|off|on) return 0 ;;
    *) return 1 ;;
  esac
}

SC_LMCACHE_PROFILE="${SC_LMCACHE_PROFILE:-vanilla}"
export SC_LMCACHE_PROFILE

case "$SC_LMCACHE_PROFILE" in
  vanilla)
    _sc_profile_gates=0
    _sc_profile_logs=0
    ;;
  bounded)
    _sc_profile_gates=1
    _sc_profile_logs=0
    ;;
  diagnostic)
    _sc_profile_gates=0
    _sc_profile_logs=1
    ;;
  bounded_diagnostic)
    _sc_profile_gates=1
    _sc_profile_logs=1
    ;;
  custom)
    _sc_profile_gates=0
    _sc_profile_logs=0
    ;;
  *)
    echo "Unknown SC_LMCACHE_PROFILE=$SC_LMCACHE_PROFILE" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac

# Functional controls. Enable flags and capacities are intentionally separate.
_sc_default SC_LMCACHE_SCHEDULER_LOOKUP_ADMISSION_ENABLE "$_sc_profile_gates"
_sc_default SC_LMCACHE_SCHEDULER_LOOKUP_MAX_INFLIGHT 4
_sc_default SC_LMCACHE_SCHEDULER_LOOKUP_ADMISSION_WAIT_MS 0

_sc_default SC_LMCACHE_WORKER_LOOKUP_ADMISSION_ENABLE "$_sc_profile_gates"
_sc_default SC_LMCACHE_WORKER_LOOKUP_MAX_INFLIGHT 1

_sc_default SC_LMCACHE_DISK_PUT_ADMISSION_ENABLE "$_sc_profile_gates"
_sc_default SC_LMCACHE_DISK_PUT_MAX_PENDING 8

# Round-6 single-serializer CPU/disk fairness controls.
# Ratio=N means: while CPU and disk are both continuously waiting, select at
# most N CPU backend operations before forcing one disk backend operation.
_sc_default SC_LMCACHE_SERIALIZER_FAIRNESS_ENABLE 0
_sc_default SC_LMCACHE_SERIALIZER_CPU_BURST_RATIO 4

# Round-5 experimental controls and their independent residency trace.
# Functional controls remain disabled unless a grid cell explicitly enables them.
_sc_default SC_LMCACHE_DISK_RESIDENT_PUT_DEDUP_ENABLE 0
_sc_default SC_LMCACHE_COLD_WARM_PUT_BARRIER_ENABLE 0
_sc_default SC_LMCACHE_DISK_PUT_RESIDENCY_TRACE_ENABLE 0
_sc_default SC_LMCACHE_PUT_BARRIER_TIMEOUT_S 1200
_sc_default SC_LMCACHE_PUT_BARRIER_POLL_S 0.25
_sc_default SC_LMCACHE_PUT_BARRIER_STABLE_S 1.0
_sc_default SC_LMCACHE_PUT_BARRIER_EXPECTED_WORKERS 2
_sc_default SC_LMCACHE_PUT_BARRIER_STATUS_DIR ""

# Independent observability controls.
_sc_default SC_LMCACHE_IO_TRACE_ENABLE "$_sc_profile_logs"
_sc_default SC_LMCACHE_LOAD_TRACE_ENABLE "$_sc_profile_logs"
_sc_default SC_LMCACHE_MEMORY_TRACE_ENABLE "$_sc_profile_logs"
_sc_default SC_LMCACHE_MEMORY_SNAPSHOT_ENABLE 0
_sc_default SC_LMCACHE_LOOKUP_TRACE_ENABLE 0
_sc_default SC_LMCACHE_REQUEST_TRACE_ENABLE 0
_sc_default SC_LMCACHE_TIER_TRACE_ENABLE 0
_sc_default SC_LMCACHE_GPU_ASSERT_SNAPSHOT_ENABLE 0
# Extremely verbose per-MemoryObj ownership trace; always opt-in.
_sc_default SC_LMCACHE_LIFECYCLE_TRACE_ENABLE 0

_sc_default SC_LMCACHE_MEMORY_TRACE_INTERVAL_S 5
_sc_default SC_LMCACHE_LONG_PIN_THRESHOLD_S 10
_sc_default SC_LMCACHE_LONG_REF_THRESHOLD_S 10

# Driver-side resource monitor, separate from LMCache logging.
_sc_default SC_DRIVER_RESOURCE_MONITOR_ENABLE "$_sc_profile_logs"
_sc_default SC_DRIVER_RESOURCE_MONITOR_INTERVAL_S 30

# Optional override for the LMCache local-disk path. Empty means the driver's
# existing default path is used.
_sc_default SC_LMCACHE_DATA_DIR ""

# Validate booleans early; numeric validation is repeated in Python where used.
for _sc_name in \
  SC_LMCACHE_SCHEDULER_LOOKUP_ADMISSION_ENABLE \
  SC_LMCACHE_WORKER_LOOKUP_ADMISSION_ENABLE \
  SC_LMCACHE_DISK_PUT_ADMISSION_ENABLE \
  SC_LMCACHE_SERIALIZER_FAIRNESS_ENABLE \
  SC_LMCACHE_DISK_RESIDENT_PUT_DEDUP_ENABLE \
  SC_LMCACHE_COLD_WARM_PUT_BARRIER_ENABLE \
  SC_LMCACHE_DISK_PUT_RESIDENCY_TRACE_ENABLE \
  SC_LMCACHE_IO_TRACE_ENABLE \
  SC_LMCACHE_LOAD_TRACE_ENABLE \
  SC_LMCACHE_MEMORY_TRACE_ENABLE \
  SC_LMCACHE_MEMORY_SNAPSHOT_ENABLE \
  SC_LMCACHE_LOOKUP_TRACE_ENABLE \
  SC_LMCACHE_REQUEST_TRACE_ENABLE \
  SC_LMCACHE_TIER_TRACE_ENABLE \
  SC_LMCACHE_GPU_ASSERT_SNAPSHOT_ENABLE \
  SC_LMCACHE_LIFECYCLE_TRACE_ENABLE \
  SC_DRIVER_RESOURCE_MONITOR_ENABLE; do
  if ! _sc_bool "${!_sc_name}"; then
    echo "Invalid boolean ${_sc_name}=${!_sc_name}" >&2
    return 2 2>/dev/null || exit 2
  fi
done

if ! [[ "$SC_LMCACHE_SERIALIZER_CPU_BURST_RATIO" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid SC_LMCACHE_SERIALIZER_CPU_BURST_RATIO=$SC_LMCACHE_SERIALIZER_CPU_BURST_RATIO; expected integer >= 1" >&2
  return 2 2>/dev/null || exit 2
fi

sc_lmcache_print_knobs() {
  local name
  echo "SC_LMCACHE_PROFILE=$SC_LMCACHE_PROFILE"
  for name in \
    SC_LMCACHE_SCHEDULER_LOOKUP_ADMISSION_ENABLE \
    SC_LMCACHE_SCHEDULER_LOOKUP_MAX_INFLIGHT \
    SC_LMCACHE_SCHEDULER_LOOKUP_ADMISSION_WAIT_MS \
    SC_LMCACHE_WORKER_LOOKUP_ADMISSION_ENABLE \
    SC_LMCACHE_WORKER_LOOKUP_MAX_INFLIGHT \
    SC_LMCACHE_DISK_PUT_ADMISSION_ENABLE \
    SC_LMCACHE_DISK_PUT_MAX_PENDING \
    SC_LMCACHE_SERIALIZER_FAIRNESS_ENABLE \
    SC_LMCACHE_SERIALIZER_CPU_BURST_RATIO \
    SC_LMCACHE_DISK_RESIDENT_PUT_DEDUP_ENABLE \
    SC_LMCACHE_DISK_PUT_RESIDENCY_TRACE_ENABLE \
    SC_LMCACHE_COLD_WARM_PUT_BARRIER_ENABLE \
    SC_LMCACHE_PUT_BARRIER_TIMEOUT_S \
    SC_LMCACHE_PUT_BARRIER_POLL_S \
    SC_LMCACHE_PUT_BARRIER_STABLE_S \
    SC_LMCACHE_PUT_BARRIER_EXPECTED_WORKERS \
    SC_LMCACHE_PUT_BARRIER_STATUS_DIR \
    SC_LMCACHE_IO_TRACE_ENABLE \
    SC_LMCACHE_LOAD_TRACE_ENABLE \
    SC_LMCACHE_MEMORY_TRACE_ENABLE \
    SC_LMCACHE_MEMORY_SNAPSHOT_ENABLE \
    SC_LMCACHE_LOOKUP_TRACE_ENABLE \
    SC_LMCACHE_REQUEST_TRACE_ENABLE \
    SC_LMCACHE_TIER_TRACE_ENABLE \
    SC_LMCACHE_GPU_ASSERT_SNAPSHOT_ENABLE \
    SC_LMCACHE_LIFECYCLE_TRACE_ENABLE \
    SC_LMCACHE_MEMORY_TRACE_INTERVAL_S \
    SC_LMCACHE_LONG_PIN_THRESHOLD_S \
    SC_LMCACHE_LONG_REF_THRESHOLD_S \
    SC_DRIVER_RESOURCE_MONITOR_ENABLE \
    SC_DRIVER_RESOURCE_MONITOR_INTERVAL_S \
    SC_LMCACHE_DATA_DIR; do
    printf '%s=%s\n' "$name" "${!name}"
  done
}

unset _sc_profile_gates _sc_profile_logs _sc_name
