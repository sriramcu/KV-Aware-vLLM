# Gate C only

This patch changes only:

`third_party/LMCache/lmcache/v1/lookup_client/lmcache_async_lookup_client.py`

It adds scheduler/client-side logical lookup admission before pending lookup creation, before `lookup_timeout_ms` starts, and before worker messages are sent.

No P0-C cleanup changes and no P0-D worker-read cancellation changes are included.

## Knobs

```bash
# Gate C logical lookup limit. If unset, inherits the existing worker-side knob.
export LMCACHE_P0_CLIENT_LOOKUP_MAX_INFLIGHT=4

# Maximum time to wait for Gate C. 0 means fail-fast and recompute.
export LMCACHE_P0_CLIENT_LOOKUP_ADMISSION_TIMEOUT_MS=0
```

`LMCACHE_P0_CLIENT_LOOKUP_MAX_INFLIGHT=0` disables Gate C while leaving worker-side P0-A unchanged.

An admitted Gate C permit is held until every configured lookup worker responds. A scheduler-side 3-second timeout does not release it, because without P0-D the underlying worker read is still unfinished.

## Apply

```bash
bash gate_c_only_bundle/apply_gate_c_only.sh
```

The editable install needs no reinstall or native rebuild.
