# SC LMCache experiment controls

Source `local_repro/sc_lmcache_knobs.sh` from an sbatch file. Values already
present in the environment are preserved, so Slurm `--export` overrides work
for grid searches.

```bash
export SC_LMCACHE_PROFILE=bounded
export SC_LMCACHE_WORKER_LOOKUP_MAX_INFLIGHT=2
sbatch --export=ALL local_repro/sbatch/02_h100_wo_gnn.sbatch
```

Set `SC_LMCACHE_SOURCE_DEFAULT_KNOBS=0` to stop the supplied sbatch files from
sourcing the defaults file. In that mode, export only the controls needed for
the experiment; unset controls use disabled/default behavior in Python.

## Profiles

| Profile | Custom gates | Verbose IO/load/memory logs | Driver monitor |
|---|---:|---:|---:|
| `vanilla` | off | off | off |
| `bounded` | on | off | off |
| `diagnostic` | off | on | on |
| `bounded_diagnostic` | on | on | on |
| `custom` | off by default | off by default | off by default |

Every individual variable overrides the profile default.

## Functional controls

| Variable | Default | Meaning |
|---|---:|---|
| `SC_LMCACHE_SCHEDULER_LOOKUP_ADMISSION_ENABLE` | `0` | Enable scheduler-side logical lookup admission before worker dispatch. |
| `SC_LMCACHE_SCHEDULER_LOOKUP_MAX_INFLIGHT` | `4` | Maximum admitted distributed lookups. |
| `SC_LMCACHE_SCHEDULER_LOOKUP_ADMISSION_WAIT_MS` | `0` | Time to wait for scheduler admission; zero means immediate rejection. |
| `SC_LMCACHE_WORKER_LOOKUP_ADMISSION_ENABLE` | `0` | Enable per-worker admission before contains/pin/staging work. |
| `SC_LMCACHE_WORKER_LOOKUP_MAX_INFLIGHT` | `1` | Maximum physical lookups per LMCache worker/rank. |
| `SC_LMCACHE_DISK_PUT_ADMISSION_ENABLE` | `0` | Enable bounded disk-write admission. |
| `SC_LMCACHE_DISK_PUT_MAX_PENDING` | `8` | Maximum admitted disk puts per worker. |

The three gates are independent. For example, worker admission can be enabled
while scheduler admission and disk-put admission remain disabled.

## Observability controls

| Variable | Default | Meaning |
|---|---:|---|
| `SC_LMCACHE_IO_TRACE_ENABLE` | `0` | Serializer, disk queue, file IO, scheduler lookup, and cleanup timing. |
| `SC_LMCACHE_LOAD_TRACE_ENABLE` | `0` | Backend load lifecycle and lookup-unpin traces. |
| `SC_LMCACHE_MEMORY_TRACE_ENABLE` | `0` | CPU-pool pressure plus long pin/reference lifetime warnings. |
| `SC_LMCACHE_MEMORY_SNAPSHOT_ENABLE` | `0` | Heavy process/cgroup/GPU/disk snapshot on explicit diagnostic calls. |
| `SC_LMCACHE_LOOKUP_TRACE_ENABLE` | `0` | Repeated worker lookup response trace and selected stacks. |
| `SC_LMCACHE_REQUEST_TRACE_ENABLE` | `0` | Repeated vLLM request matching trace and selected stacks. |
| `SC_LMCACHE_GPU_ASSERT_SNAPSHOT_ENABLE` | `0` | Heavy snapshot immediately before the GPU connector tensor assertion. |
| `SC_DRIVER_RESOURCE_MONITOR_ENABLE` | `0` | Periodic driver-level process/GPU/disk monitor. |

Related thresholds:

| Variable | Default |
|---|---:|
| `SC_LMCACHE_MEMORY_TRACE_INTERVAL_S` | `5` |
| `SC_LMCACHE_LONG_PIN_THRESHOLD_S` | `10` |
| `SC_LMCACHE_LONG_REF_THRESHOLD_S` | `10` |
| `SC_DRIVER_RESOURCE_MONITOR_INTERVAL_S` | `30` |

`SC_LMCACHE_DATA_DIR` optionally overrides the driver's local-disk path.

## Legacy-name migration

| Old | New |
|---|---|
| `LMCACHE_P0_CLIENT_LOOKUP_MAX_INFLIGHT` | `SC_LMCACHE_SCHEDULER_LOOKUP_MAX_INFLIGHT` plus the scheduler enable flag |
| `LMCACHE_P0_CLIENT_LOOKUP_ADMISSION_TIMEOUT_MS` | `SC_LMCACHE_SCHEDULER_LOOKUP_ADMISSION_WAIT_MS` |
| `LMCACHE_P0_LOOKUP_MAX_INFLIGHT` | `SC_LMCACHE_WORKER_LOOKUP_MAX_INFLIGHT` plus the worker enable flag |
| `LMCACHE_P0_DISK_PUT_MAX_PENDING` | `SC_LMCACHE_DISK_PUT_MAX_PENDING` plus the disk-put enable flag |
| `SRIRAM_KV_IO_TRACE` | `SC_LMCACHE_IO_TRACE_ENABLE` |
| `SRIRAM_KV_MEM_DEBUG` | `SC_LMCACHE_MEMORY_TRACE_ENABLE` |
| `SRIRAM_KV_MEM_DEBUG_INTERVAL_S` | `SC_LMCACHE_MEMORY_TRACE_INTERVAL_S` |
| `SRIRAM_LONG_PIN_SECONDS` | `SC_LMCACHE_LONG_PIN_THRESHOLD_S` |
| `SRIRAM_LONG_REF_SECONDS` | `SC_LMCACHE_LONG_REF_THRESHOLD_S` |
| `SRIRAM_LMCACHE_DIR` | `SC_LMCACHE_DATA_DIR` |
| `[SRIRAM_MONITOR]` | `[SC_DRIVER_MONITOR]` controlled by `SC_DRIVER_RESOURCE_MONITOR_ENABLE` |

The refactor intentionally does not read legacy names. This prevents an old
batch environment from silently enabling an experiment under a new naming
scheme.
