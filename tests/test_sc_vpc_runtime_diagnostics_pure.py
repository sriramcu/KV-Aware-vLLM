# SPDX-License-Identifier: Apache-2.0
"""Standalone lifecycle regression without importing GPU-compiled vLLM."""
import importlib.util
import sys
import time
from collections import Counter
from pathlib import Path
from types import ModuleType, SimpleNamespace


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_diagnostics_sample_first_victim_and_hole(monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    diag_dir = repo / "vllm/v1/core/vpc_policies"
    pure = _module(diag_dir / "diagnostics.py", "sc_test_pure_helpers")
    # Stub only imports used in the diagnostic module; actual production
    # vLLM core policy integration is covered by test_sc_vpc_diagnostics.py.
    for package in ("vllm", "vllm.v1", "vllm.v1.core",
                    "vllm.v1.core.vpc_policies"):
        monkeypatch.setitem(sys.modules, package, ModuleType(package))
    util = ModuleType("vllm.v1.core.kv_cache_utils")
    util.resolve_block_hashes = lambda hashes, *unused: hashes
    util.make_block_hash_with_group_id = lambda key, group: key
    monkeypatch.setitem(sys.modules, util.__name__, util)
    monkeypatch.setitem(sys.modules,
        "vllm.v1.core.vpc_policies.diagnostics", pure)
    runtime = _module(diag_dir / "runtime_diagnostics.py", "sc_runtime_pure")
    b0 = SimpleNamespace(block_id=0, ref_cnt=0, is_null=True, block_hash=None)
    b1 = SimpleNamespace(block_id=1, ref_cnt=0, is_null=False, block_hash="k0")
    b2 = SimpleNamespace(block_id=2, ref_cnt=0, is_null=False, block_hash=None)
    keys = tuple(f"k{i}" for i in range(32))
    cache = SimpleNamespace(_cache={key: b1 for key in keys})
    stats = SimpleNamespace(
        start_ns=time.monotonic_ns(), diagnostic_sample=None, first_gpu_reclaim=None,
        gpu_chunk_reclaims=Counter(), gpu_chunk_reclaim_physical_events=0,
        gpu_chunk_complete_idle_breaks=0, gpu_chunk_complete_any_breaks=0,
    )
    pool = SimpleNamespace(
        blocks=[b0, b1, b2], kv_importance_by_block_id={1: "gpu"},
        cached_block_hashes_by_block={1: set(keys)},
        cached_block_hash_to_block=cache,
        hash_block_size=16, num_gpu_blocks=3, vpc_stats=stats,
        get_num_free_blocks=lambda: 2,
    )
    d = runtime.VPCDiagnostics(pool, interval_s=5)
    d.observe_request_prefix(SimpleNamespace(
        request_id="kvaware-cold-0", block_hashes=list(keys),
        num_tokens=512, kv_importance_placements={0: "gpu"}), 16)
    assert d.request_count == 1
    assert len(d.observed_chunks) == 1
    d.maybe_sample(force=True)
    assert stats.diagnostic_sample["accounting_ok"]
    assert stats.diagnostic_sample["physical_blocks"]["idle_cached"]["gpu"] == 1
    assert stats.diagnostic_sample["observed_chunk_universe"]["gpu"]["complete_idle"] == 1
    d.before_reclaim(b1, "gpu")
    assert stats.first_gpu_reclaim["requests_first_seen"] == 1
    cache._cache.clear()  # mimic the allocator's hash invalidation
    d.after_reclaim()
    assert stats.gpu_chunk_reclaim_physical_events == 1
    assert stats.gpu_chunk_complete_idle_breaks == 1
    assert stats.gpu_chunk_complete_any_breaks == 1
    d.reset()
    assert d.request_count == 0 and not d.observed_chunks
