# Canonical Q650 H100 — Python-only patch and phase-trace guide

This patch targets the uploaded **KV-Aware-vLLM-lmcache-mp-l0-vllm029**
repository snapshot. It changes no C/C++/CUDA files and requires no
vLLM/CUDA rebuild **when Python imports resolve to the modified checkout**.
It has not been validated on H100 and does not automatically request a GPU.

## 1. Apply

On the login node, copy the downloaded `canonical_q650_phase_trace.patch` into
`/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM/` (for example with `scp`).

```bash
export REPO=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
cd "$REPO"
git apply --check canonical_q650_phase_trace.patch
git apply canonical_q650_phase_trace.patch
```

If `git apply --check` fails, **do not force the patch**. It was generated
against the uploaded ZIP, which may differ from your current working tree.
Back up any local modifications and inspect the reject/diff before proceeding.

Verify source import location without using a GPU:

```bash
source /mnt/shared/gpfs/home/sriramc2/venvs/kvaware-vllm029/bin/activate
python -c 'import vllm; print(vllm.__file__)'
python -m unittest discover -s tests -p test_canonical_phase_cpu.py -v
bash -n scripts/run_shortq_gnn_dynamic_vpc_common.sh \
        scripts/run_h100_canonical_q650_phase_trace.sbatch
python -m compileall -q experiments/passage_order_policies \
  vllm/v1/metrics/phase_trace.py vllm/v1/metrics/phase_device_trace.py \
  scripts/analyze_phase_trace.py scripts/plot_phase_gantt.py
```

The Python `vllm.__file__` should point to the checkout or editable install
that will be imported by the launcher. Python source changes take effect after
server restart; no extension compilation is needed.

## 2. Submit the canonical experiment (when H100 queue clears)

The new sbatch wrapper matches the **main true-DROP GNN job 29265** settings:
Q650 / TP2 / H100 / FP8 / BF16 KV / MNS16 / B200 / fixed output 128 /
C225 / U0.25 / L2 LOAD2 / LOOKUP16 / VPC W256 / L1 backing on /
L2 backing off / store barrier on / GNN 8-2 true-DROP vote.

The **cold request order** is seeded-shuffle 0, and the **independent warm
request order** is random seed 1, matching the headline methodology. Both
original and canonical reruns must use these same *respective phase seeds*.

```bash
export REPO=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
cd "$REPO"
# Do NOT run until ready to submit a real H100 job:
sbatch scripts/run_h100_canonical_q650_phase_trace.sbatch
```

The existing common launcher has `CLEAN_OLD_LMCACHE=1` **by default** and
its cleanup removes **everything immediately under `/scratch2/sriramc2`**.
Read that function and ensure this is acceptable before submitting. Set
`CLEAN_OLD_LMCACHE=0` to skip deleting scratch, but only if you can ensure an
otherwise clean and isolated run. The new wrapper has not modified that
existing behavior.

To run canonical ordering without phase traces later:

```bash
sbatch --export=ALL,VLLM_PHASE_TRACE=0 \
  scripts/run_h100_canonical_q650_phase_trace.sbatch
```

To enable **experimental** worker CUDA-event timing later (not validated on
H100; off by default):

```bash
sbatch --export=ALL,VLLM_PHASE_TRACE_DEVICE_TIMING=1 \
  scripts/run_h100_canonical_q650_phase_trace.sbatch
```

## 3. Results

The Slurm job writes the run directory to its output as `run_dir=...`.
Inside it, inspect:

- `run_config.json`: canonical policy, phase seeds, tier config, trace flags.
- `results/manifest.json`: per-run driver metadata.
- `results/prompt_records.jsonl`: final canonical order, raw passage ranks,
  exact constructed prompt, and scores carried along with their passages.
- `results/summary.json`: cold/warm E2E, TTFT and existing driver measurements.
- `vllm_phase_metrics.json`: **unchanged** historical Prometheus definitions.
- `logs/phase_trace/phase_engine.pid*.jsonl`: scheduled prompt and decode
  positions (snapshotted before the scheduler advances computed-token counts),
  per-batch scheduling/submission/result/update CPU timestamps and remote KV
  waiting-state transitions.
- `phase_trace_summary.json`: per-phase scheduled token totals, batch type
  counters, descriptive decode interstep gaps and KV wait intervals.
- `logs/phase_trace/phase_device.pid*.jsonl`: optional per-worker CUDA-forward
  timings only when `VLLM_PHASE_TRACE_DEVICE_TIMING=1`.

Analyze a trace again after the job exits (CPU-only):

```bash
export RUN_DIR=/path/from/slurm/run_dir
python scripts/analyze_phase_trace.py \
  --trace-dir "$RUN_DIR/logs/phase_trace" \
  --output "$RUN_DIR/phase_trace_summary.json"
```

Plot a representative request's **participating batch wall intervals**:

```bash
python scripts/plot_phase_gantt.py \
  --trace-dir "$RUN_DIR/logs/phase_trace" \
  --request-id kvaware-warm-000123 \
  --output "$RUN_DIR/analysis/warm_000123.png"
```

## 4. Measurement boundaries / limitations

- **Prompt compute tokens** = actual prompt positions scheduled for model
  execution, not all prompt tokens or all cache lookups. Cached-prefix tokens
  are omitted. For speculative/multimodal execution or aborted in-flight
  steps, treat counts as *issued* work, not strictly completed kernel work.
- **Decode-position tokens** = scheduled generated-token positions (the first
  sampled token is produced by final prefill, so a 128-token completion
  typically needs approximately 127 decode-position computations).
- **Engine submit-to-result wall** is **not** exclusive GPU model-forward time:
  it can contain queueing, worker communication, KV handling, and other work.
- **Decode interstep gaps** are descriptive; they do not prove that another
  request's prefill *caused* the gap. Mixed batches also cannot be assigned
  exclusive per-request GPU time.
- `kv_wait_enter/exit` times the Stage-2 `WAITING_FOR_REMOTE_KVS` scheduler
  state, not the entire asynchronous Stage-1 L2 GET or overlapping L1 staging.
- Optional PyTorch CUDA events cover the forward region on a worker/rank and
  **do not** distinguish prefill kernels from decode kernels inside mixed
  batches. CUDA-event behavior under graph capture needs GPU validation first.
- High-granularity CPU tracing is intentionally opt-in and introduces overhead.
  Rerun an `original` policy run with the **same tracing flags** before making
  rigorous E2E comparisons. Historical headline numbers remain references only.
- Passage ordering uses a stable SHA-256 digest of passage **text** and does
  **not pad passages**. The initial ordering policy registry offers `original`
  and `canonical`; cache-aware policies can be added later without changing
  the driver and precomputer separately.
- When using `KV_REQUEST_ORDER=legacy_prefix_hash`, changing passage order
  can also change *question* order. The new canonical wrapper explicitly
  chooses `seeded_shuffle` so request order stays paired across strategies.

## 5. Run a matched original-order trace later

For an instrumented original-order counterpart, change only the new wrapper's
`KV_PASSAGE_ORDER_POLICY=canonical` line to `original`, or submit the original
Q650 sbatch with the headline flag set and tracing enabled, while retaining
*identical configuration values and cold/warm seeds*. The canonical wrapper defaults to `canonical` but permits an override:

```bash
sbatch --export=ALL,KV_PASSAGE_ORDER_POLICY=original,EXPERIMENT_LABEL=original_trace_q650_b200_seed0_seed1 \
  scripts/run_h100_canonical_q650_phase_trace.sbatch
```

For a no-GNN canonical control, the existing
`scripts/run_h100_mp_nognn_q650_current_cleanall.sbatch` now also accepts
`KV_PASSAGE_ORDER_POLICY=canonical`. Its other configuration flags must be
matched separately to the no-GNN headline run (job 29219); **do not assume
its script defaults match that historical run**. It uses its own destructive
scratch cleanup and needs the same cold/warm shuffle seeds.
