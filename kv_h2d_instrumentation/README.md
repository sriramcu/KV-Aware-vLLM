# KV-Aware / LMCache CPU H2D timing diagnostic (instrumentation ONLY)

Based on the user's uploaded KV-Aware-vLLM `feat-bayesian-kv-placement` snapshot (Oct 2026), including the Next9 cost-adaptation guard.

## Scope and guarantees

- **No changes to `bayesian_mapper.py`, `LmcacheCompletionCostReader`, `observe_cost`, placement, cache admission, CUDA transfer kernels, or the pre-existing `BAYES_H2D_COMPLETE` cost signal.**
- Only 3 files touched: `lmcache_driven_transfer.py`, `mp_observability/config.py`, and new `subscribers/bayes_h2d_diagnostic.py`.
- Disabled by default. Enable with `LMCACHE_BAYES_H2D_DIAG=1` in the LMCache server's environment, plus the already-used `LMCACHE_BAYES_TIMING=1` through the Bayesian launcher.
- CUDA event measurements are asynchronous and read on the **Python DeviceHostFuncDispatcher worker** after stream completion, NOT from inside the CUDA host function (where CUDA runtime calls would be unsafe).
- Existing C++ `PhaseTimer` instrumentation is reused; **no C++ rebuild is required if the installed `lmcache.device_ops` exposes `pop_completed_phase_timings` and native execution is enabled**. If absent, rebuild/install the repository's vendored LMCache native extension, or provide the original run's installed-library status.
- Diagnostic GPU timing adds events/callbacks and logging; it can perturb latency. Use it for **measurement diagnosis**, not speedup claims. Never replace the existing cost estimate with these numbers until independently validated.

## Apply (repo root, no old patches/installer reruns)

```bash
cd /mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
# Transfer the ZIP to this machine, unzip it somewhere outside the repo.
# Adjust the path below to where you unpacked the bundle:
PATCH=/path/to/kv_h2d_instrumentation/bayes_h2d_diagnostic.patch

git apply --check "$PATCH" || { echo 'STOP: installed tree differs; do not force/overwrite'; exit 1; }
git apply "$PATCH"
python /path/to/kv_h2d_instrumentation/tests/test_static_invariants.py
```

If `git apply --check` fails, supply the current three target files or a narrow diff; do **not** restore `bayesian_mapper.py`, overwrite it with an earlier snapshot, or use an old Next9 installer.

If the vendored LMCache is not installed in editable mode, check that `python -c 'import lmcache.v1.multiprocess.modules.lmcache_driven_transfer as m; print(m.__file__)'` points to the patched repo, **not** an older wheel in site-packages. Reinstall the vendored package in the active venv only if needed, following the installation method already used in the project.

## Preflight

In the *active* `kvaware-vllm029` environment, check:

```bash
python - <<'PY'
from lmcache import device_ops
from lmcache.v1.multiprocess.modules import lmcache_driven_transfer as m
print('LMCache module:', m.__file__)
print('Native phase timing:', hasattr(device_ops, 'pop_completed_phase_timings'))
print('Native plan executor:', hasattr(device_ops, 'execute_object_group_transfer'))
PY
```

With `LMCACHE_BAYES_H2D_DIAG=1`, `logs/lmcache.log` should include `[BAYES_H2D_DIAG_INIT] native_phase_timing=1`; if it says `0`, H2D wait/whole-region events may still be present, but **native per-step DMA/kernel samples will not be**. Check source/build installation rather than interpreting missing samples as 0 ms.

## Recommended run: matched calibrated-prior, cost ON diagnostic

Existing `scripts/h100_next8/run_h100_bayes_sweep_variable.sbatch` accepts a `c_*` cell ID, so no new launcher is needed. This chooses the configuration of 29954, except for instrumentation:

```bash
cd /mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
sbatch --dependency=singleton --job-name=kvq650-10batch --time=06:00:00 \
  --export=ALL,SWEEP_CELL=c_h2ddiag_cal_w05_bayes,SWEEP_POLICY=bayes,SWEEP_ORDER=original,SWEEP_GPU_GB=25,SWEEP_CPU_GB=225,SWEEP_DISK_GB=500,SWEEP_STORE_CAP_GB=0.25,SWEEP_PRIOR_WEIGHT=0.5,SWEEP_POSTERIOR_MODE=fixed,SWEEP_PRIOR_FILE=/mnt/shared/gpfs/home/sriramc2/runs/kvaware/h100_expected_ordinal_exploratory.json,KV_BAYES_ADAPT_COST=1,LMCACHE_BAYES_H2D_DIAG=1,NEXT8_DATASET=medical,NEXT8_Q=650,NEXT8_B=100,CLEAN_OLD_LMCACHE=1 \
  scripts/h100_next8/run_h100_bayes_sweep_variable.sbatch
```

**Operational safety:** singleton serializes jobs sharing this name. The existing launcher may delete **all contents** of `/scratch2/sriramc2` when `CLEAN_OLD_LMCACHE=1`; preserve the root and ensure no unrelated user or job is using that path. The command does not add `--mem`. Logs and summary stay on GPFS.

For a cheap hook-validation smoke, change only `NEXT8_Q=80,NEXT8_B=20` and `SWEEP_CELL=c_h2ddiag_smoke_bayes`. A shorter/smaller run does **not** reproduce Q650/B100 contention; the full run is needed for attribution.

## Data produced in `logs/lmcache.log`

- `[BAYES_H2D_COMPLETE]`: **unchanged** original posterior-input sample, request, chunks, and wall elapsed ms.
- `[BAYES_H2D_DIAG]`: keyed by request and `transfer_key`, device, chunks and bytes, `host_enqueue_ms`, `gpu_producer_wait_ms`, `gpu_postwait_gap_ms`, `gpu_transfer_region_ms`, `gpu_stream_region_ms`, `wall_to_dispatch_ms`.
- `[BAYES_H2D_PHASE]`: native C++ CUDA-event samples keyed by `transfer_key` and device; `staging_dma` is the host-to-device staging DMA section; `scatter_kernel` is the GPU staging-to-paged-KV kernel section. Multiple rows per logical request. Some operations can use the fallback path, emit no native samples, or be incompletely captured under a queue cap.
- `[BAYES_H2D_DIAG_INIT]`: installed native phase-timing availability.
- `[BAYES_H2D_DIAG_ERROR]`: instrumentation errors, to be investigated before interpreting the corresponding sample.

Use the analysis script on a downloaded LMCache log:

```bash
python /path/to/kv_h2d_instrumentation/scripts/analyze_h2d_diagnostic.py \
  /path/to/RUN/logs/lmcache.log --csv /path/to/h2d_diag_parsed.csv
```

**Interpretation:** CUDA event intervals are *stream GPU timelines*, which include work/queue contention between their endpoints. `gpu_producer_wait_ms` isolates the specific stream wait on the engine producer event (not every dependency). `gpu_transfer_region_ms` can include host-issued gaps and kernel contention; it is **not pure memcpy latency**. The native `staging_dma` samples are closest to actual device DMA, but may still reflect contention and instrumentation overhead. `wall_to_dispatch_ms` includes server work, GPU waits and callback dispatch, and is not simply the sum of non-overlapping GPU intervals. There is no valid generic subtraction that yields an exact marginal serving latency. Comparability with the recomputation estimator still needs validation.

**Diagnostic questions:** (1) Are the 500–1000 ms `BAYES_H2D_COMPLETE` measurements largely accounted for by `gpu_producer_wait_ms`? (2) Are native `staging_dma` durations closer to a few ms and appropriately dependent on bytes? (3) Is `scatter_kernel` delayed by GPU compute contention? (4) Is most remaining delay in server submission/callback scheduling? (5) Do both TP ranks show similar producer waits but independent native staging durations?

## Rollback

```bash
cd /mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
git apply --check -R /path/to/kv_h2d_instrumentation/bayes_h2d_diagnostic.patch
git apply -R /path/to/kv_h2d_instrumentation/bayes_h2d_diagnostic.patch
```

Only reverse if the target files have not subsequently been edited. `LMCACHE_BAYES_H2D_DIAG=0` disables the diagnostic without rollback.

## Verification performed here

The patch was generated against the uploaded repository ZIP, applied successfully to a clean extraction using `git apply --check` and `git apply`, and all three Python modules were syntax-checked. The static invariants test passed. **H100 CUDA runtime behavior was not testable in this container**; the smoke run is required before relying on GPU timing data.
