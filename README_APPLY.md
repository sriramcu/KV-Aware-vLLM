# KV-Aware refactor patch bundle

Basis: the latest uploaded `KV-Aware-vLLM-lmcache-mp-l0-vllm029(1).zip`.

Reference upstreams used for the audit:

- vLLM v0.29.0: `98dff2a81d747d1dba01a47f939f48c3526d4206`
- LMCache: `60480f0e3defbce935dcf2bc3356b85decfb31c6`

## Recommended application: compact form

From the repository root:

```bash
git switch -c sc-refactor

git apply --check /path/to/kvaware_refactor_core.patch
git apply /path/to/kvaware_refactor_core.patch
bash /path/to/cleanup_obsolete_files.sh

git status --short
```

`kvaware_refactor_core.patch` contains every non-deletion change directly against the uploaded tree. `cleanup_obsolete_files.sh` removes the 42 files whose final state is deletion (historical patch artifacts, superseded L0/experiment scripts, project L0 implementation/tests, and the old tier-mapping module).

## Alternative: one complete patch

```bash
git switch -c sc-refactor
git apply --check /path/to/kvaware_refactor_full.patch
git apply /path/to/kvaware_refactor_full.patch
```

The full patch is large because it records deletion of historical text patch artifacts.

## Alternative: preserve the two logical commits

```bash
git switch -c sc-refactor
git am --3way /path/to/series/0001-*.patch
git am --3way /path/to/series/0002-*.patch
```

The commits are:

1. remove historical patches and superseded L0 experiments;
2. consolidate the surviving KV-aware extensions.

## Post-apply validation

```bash
git diff --check

PYTHONPATH=. pytest -q --confcutdir=tests/kvaware tests/kvaware

bash -n scripts/run_shortq_gnn_dynamic_vpc_common.sh
bash -n scripts/run_h100_mp_nognn_q650_current_cleanall.sbatch
bash -n scripts/run_h100_q650_storecap.sh
bash -n scripts/run_h100_shortq_gnn_batch_sweep.sbatch
bash -n scripts/run_l40_shortq_gnn_batch_sweep.sbatch
bash -n scripts/run_l40_mp_nognn_q10.sbatch
bash -n scripts/submit_h100_mutual_prefix_grid.sh
```

The source refactor was also Python-compiled before packaging. Full LMCache runtime tests cannot run in the packaging container because the LMCache native extension/device ops are not built there; run the relevant LMCache tests in the existing CodeNimbus environment after application.

## Main retained behavior

- two-pass progress-based Stage-1 fallback remains; occupancy/GPU-util fallback is removed;
- persistent project L0 is removed;
- mutual-prefix and VPC-sufficient bypass remain;
- bounded GNN-aware VPC bias remains; the older request-local importance experiment is removed;
- lookup PF and load/GET PF remain independently controlled;
- selected fs-native worker layout remains expressible directly through worker counts;
- rolling L2 store-byte admission remains, now explicit StorageManager/controller config;
- validated vLLM KV-load rewind repair and late-completion/accounting fixes remain;
- Short-Q placement vocabulary is `gpu/cpu/disk` and chunk-voting policy names no longer encode thresholds;
- source-defined voting remains the default selection mechanism, with optional environment overrides;
- all surviving modifications to existing upstream vLLM/LMCache files are marked with nearby `[SC]` comments; provenance is indexed in `docs/SC_MODIFICATIONS.md`.
