from __future__ import annotations

from Hierarchical_KV.shortq_placement.chunk_voting import (
    ChunkVoteConfig,
    make_block_placement_vote,
    resolve_vote_config,
    selected_vote_policy_name,
    vote_chunk_placement,
)


def _votes(gpu: int = 0, cpu: int = 0, *, cpu_top2_extra: int = 0):
    rows = [make_block_placement_vote("gpu") for _ in range(gpu)]
    rows += [make_block_placement_vote("cpu") for _ in range(cpu)]
    rows += [
        make_block_placement_vote("disk", cpu_top2=i < cpu_top2_extra)
        for i in range(32 - gpu - cpu)
    ]
    return rows


def test_source_default_preserves_latest_effective_thresholds(monkeypatch):
    monkeypatch.delenv("GNN_CHUNK_VOTE_POLICY", raising=False)
    for name in (
        "GNN_CHUNK_EXPECTED_BLOCKS",
        "GNN_CHUNK_GPU_MIN",
        "GNN_CHUNK_CPU_MIN",
        "GNN_CHUNK_CPU_TOP2_MIN",
    ):
        monkeypatch.delenv(name, raising=False)

    assert selected_vote_policy_name() == "threshold_with_cpu_top2"
    assert vote_chunk_placement(_votes(gpu=8)) == "gpu"
    assert vote_chunk_placement(_votes(gpu=7, cpu=2)) == "cpu"
    assert vote_chunk_placement(_votes(gpu=7, cpu=1, cpu_top2_extra=6)) == "cpu"
    assert vote_chunk_placement(_votes(gpu=7, cpu=1, cpu_top2_extra=5)) == "disk"


def test_environment_can_select_policy_and_knobs(monkeypatch):
    monkeypatch.setenv("GNN_CHUNK_VOTE_POLICY", "threshold")
    monkeypatch.setenv("GNN_CHUNK_GPU_MIN", "4")
    monkeypatch.setenv("GNN_CHUNK_CPU_MIN", "3")

    config = resolve_vote_config()
    assert config.gpu_min == 4
    assert config.cpu_min == 3
    assert vote_chunk_placement(_votes(gpu=4)) == "gpu"
    assert vote_chunk_placement(_votes(cpu=3)) == "cpu"


def test_explicit_config_overrides_environment(monkeypatch):
    monkeypatch.setenv("GNN_CHUNK_GPU_MIN", "1")
    config = ChunkVoteConfig(gpu_min=9, cpu_min=9, cpu_top2_min=9)
    assert vote_chunk_placement(_votes(gpu=8), config=config) == "disk"
