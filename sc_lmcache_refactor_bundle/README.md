# SC LMCache refactor bundle

This bundle refactors the custom LMCache additions captured in
`kvaware_important_code_bundle_3.0.txt`. It does **not** implement the proposed
CPU-priority serializer, disk-read gate, or cancellation policy. It only
renames, separates, centralizes, and makes the existing additions independently
configurable.

## Apply safely

From the KV-Aware-vLLM repository:

```bash
bash apply_sc_lmcache_refactor.sh --check
bash apply_sc_lmcache_refactor.sh
```

The default mode uses `git apply --check` first and creates a backup tarball
beside the repository before changing anything. When the captured baseline has
drifted, it exits without modifying files.

The deliberate full-file fallback is:

```bash
bash apply_sc_lmcache_refactor.sh --replace
```

That mode also creates a backup, but overwrites all 17 target files. Restore a
reported backup with:

```bash
bash apply_sc_lmcache_refactor.sh --restore /path/to/sc_lmcache_refactor_backup_TIMESTAMP.tar.gz
```

## Main changes

- Replaces `P0-A`, `P0-B`, and `Gate C` terminology with descriptive names:
  scheduler lookup admission, worker lookup admission, and disk-put admission.
- Adds separate enable flags and separate capacity knobs for all three gates.
- Removes coupling where scheduler admission inherited the worker limit.
- Replaces active `SRIRAM_*`, `LMCACHE_P0_*`, `KVIO_*`, and `KVDBG_*` names with
  `SC_*` names.
- Separates IO, load, memory, heavy snapshot, request, lookup, GPU-assert, and
  driver-monitor logging controls.
- Adds `local_repro/sc_lmcache_knobs.sh`, with caller environment values taking
  precedence for Slurm grid searches.
- Updates the active H100 with-GNN, H100 without-GNN, all-datasets, and L40
  sbatch files to source the shared knob file. Set
  `SC_LMCACHE_SOURCE_DEFAULT_KNOBS=0` to bypass it.
- Adds `third_party/LMCache/lmcache/v1/sc_config.py` for consistent boolean and
  numeric environment parsing.
- Keeps custom gates and verbose custom tracing disabled when their variables
  are unset, which is the closest behavior to vanilla LMCache.

## Profiles

The shared shell file supports:

- `vanilla`: gates off, verbose logs off
- `bounded`: gates on, verbose logs off
- `diagnostic`: gates off, verbose logs on
- `bounded_diagnostic`: gates and verbose logs on
- `custom`: neutral defaults, intended for explicit grid values

Every individual `SC_*` variable overrides the selected profile.

Example:

```bash
sbatch --export=ALL,SC_LMCACHE_PROFILE=custom,\
SC_LMCACHE_SCHEDULER_LOOKUP_ADMISSION_ENABLE=0,\
SC_LMCACHE_WORKER_LOOKUP_ADMISSION_ENABLE=1,\
SC_LMCACHE_WORKER_LOOKUP_MAX_INFLIGHT=2,\
SC_LMCACHE_DISK_PUT_ADMISSION_ENABLE=0,\
SC_LMCACHE_IO_TRACE_ENABLE=1 \
local_repro/sbatch/02_h100_wo_gnn.sbatch
```

See `replacement_tree/local_repro/SC_LMCACHE_KNOBS.md` for the complete table
and legacy-name mapping.

## Review files

- `sc_lmcache_refactor.patch`: commit-style unified patch.
- `sc_lmcache_refactor_diff.html`: colored, browser-friendly rendering.
- `sc_lmcache_refactor_commit_summary.txt`: changed-file summary.
- `replacement_tree/`: complete final versions of all affected files.

## Validation performed

- Python source compilation for every modified Python file.
- `bash -n` for the shared knobs file, driver, and modified sbatch files.
- Profile/override behavior checks for the shared knobs file.
- Safe patch apply test against the captured baseline.
- Full replacement and backup-restore tests.
- Legacy active-name scan across modified source and shell files.

No H100/vLLM runtime test was possible in this environment. The application
script therefore validates syntax and patch integrity but cannot establish
runtime performance or concurrency correctness on the cluster.
