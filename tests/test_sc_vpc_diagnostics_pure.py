# SPDX-License-Identifier: Apache-2.0
"""Standalone geometry/counting regression checks without vLLM compiled deps."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "vpc_diag_pure", HERE / "vllm/v1/core/vpc_policies/diagnostics.py")
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_physical_accounting_and_classification():
    b = [SimpleNamespace(is_null=i == 0, block_id=i, ref_cnt=0,
                         block_hash=None) for i in range(5)]
    b[1].ref_cnt = 2
    b[2].block_hash = b"k2"
    b[3].block_hash = b"k3"
    o = module.occupancy(b, {1: "cpu", 2: "gpu", 3: "disk"}, {})
    assert o["total"] == 5
    assert o["reserved_special"] == 1
    assert o["active"] == 1 and o["active_by_label"]["cpu"] == 1
    assert o["idle_cached"]["gpu"] == 1
    assert o["idle_cached"]["disk"] == 1
    assert o["uncached_available"] == 1
    assert o["accounted_total"] == 5


def test_prefix_gap_and_chunk_universe_are_read_only():
    paths = [tuple(f"c{j}-block-{i}" for i in range(32)) for j in range(2)]
    cache = SimpleNamespace(_cache={})
    for index, k in enumerate(paths[0]):
        if index not in (3, 4):
            cache._cache[k] = SimpleNamespace(ref_cnt=0)
    for k in paths[1]:
        cache._cache[k] = SimpleNamespace(ref_cnt=1)
    state_before = dict(cache._cache)
    result = module.chunk_summary(cache, {paths[0]: "gpu", paths[1]: "cpu"})
    assert result["gpu"]["partial_any"] == 1
    assert result["gpu"]["resident_any_blocks"] == 30
    assert result["gpu"]["stranded_any_blocks"] == 27
    assert result["gpu"]["complete_idle"] == 0
    assert result["cpu"]["complete_any"] == 1
    assert result["cpu"]["complete_idle"] == 0
    assert cache._cache == state_before
