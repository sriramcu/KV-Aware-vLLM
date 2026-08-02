# Install the switch-enabled SC pressure-grid smoke runner

From the `KV-Aware-vLLM` repository root:

```bash
bash /path/to/sc_pressure_grid_smoke_bundle_v2/apply_sc_pressure_grid.sh
```

The apply script accepts either:

- the post-SC-refactor tree before the pressure-grid patch; or
- the earlier pressure-grid implementation without the lightweight-metric switch.

It performs patch checks, creates a direct tar backup in `$HOME`, applies the
appropriate patch, restores executable modes, compiles the Python sources
without importing vLLM, validates shell syntax, verifies all 16 configurations,
and verifies that the lightweight metric records are gated by
`SC_LMCACHE_IO_TRACE_ENABLE`.

Start the foreground one-question smoke grid with:

```bash
./local_repro/grid_search/run_smoke_grid.sh
```

The pressure grid explicitly sets `SC_LMCACHE_IO_TRACE_ENABLE=1`, so its
lightweight metrics remain enabled. Setting that variable to `0` disables the
four `SC_DRIVER_*` metric records together with the detailed IO trace. The
vanilla profile uses `0`.

No H100 job was run while building this revision.
