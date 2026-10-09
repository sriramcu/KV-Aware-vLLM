# SPDX-License-Identifier: Apache-2.0
"""Static hindsight ranking controls, no Torch / vLLM dependency."""
import json
import pytest
from Hierarchical_KV.shortq_placement.hindsight_oracle import (
    rank_from_occurrences, load_ranking, assign_oracle,
    oracle_mode_from_environment, apply_oracle_placements,
)


def test_oracle_rank_budget_and_drop(tmp_path):
    rows = [{"chunk_hash": x} for x in ("b", "a", "a", "c", "a", "b", "d")]
    ranking = rank_from_occurrences(rows)
    assert [x["chunk_hash"] for x in ranking["ranking"]] == ["a", "b", "c", "d"]
    p = tmp_path / "rank.json"
    p.write_text(json.dumps(ranking))
    hashes = load_ranking(p)
    assigned = assign_oracle({"a", "b", "c", "d"}, hashes, gpu=1, cpu=1, disk=1)
    assert assigned == {"a": "gpu", "b": "cpu", "c": "disk", "d": "drop"}
    assert assign_oracle(set(assigned), hashes, gpu=2, cpu=0, disk=0)["b"] == "gpu"


def test_oracle_mismatch_and_overallocation_fail(tmp_path):
    path = tmp_path / "rank.json"
    path.write_text('["a", "b", "c"]')
    with pytest.raises(ValueError, match="mismatch"):
        assign_oracle({"a", "b", "unexpected"}, load_ranking(path), 1, 1, 1)
    with pytest.raises(ValueError, match="exceed"):
        assign_oracle({"a", "b", "c"}, load_ranking(path), 2, 2, 0)
    path.write_text('["a", "a"]')
    with pytest.raises(ValueError, match="duplicate"):
        load_ranking(path)


def test_oracle_metadata_rewrite_and_opt_in(tmp_path):
    path = tmp_path / "rank.json"
    path.write_text('["a", "b", "c", "d"]')
    runtime = {h: "cpu" for h in "abcd"}
    occurrences = [{"chunk_hash": h, "candidate_placement": "cpu"}
                   for h in "aabc"]
    predictions = [{"chunk_hash": h, "prediction": "cpu"}
                   for h in "abcd"]
    # GNN default must have no side effects.
    before = (dict(runtime), [dict(row) for row in occurrences])
    assert apply_oracle_placements(runtime, occurrences, predictions,
                                   environ={}) == (None, None, None)
    assert (runtime, occurrences) == before
    env = {
        "KV_CHUNK_PLACEMENT_MODE": "oracle",
        "VLLM_VPC_POLICY": "selective",
        "KV_ORACLE_RANKING_FILE": str(path),
        "KV_ORACLE_GPU_CHUNKS": "1", "KV_ORACLE_CPU_CHUNKS": "1",
        "KV_ORACLE_DISK_CHUNKS": "1",
    }
    assert oracle_mode_from_environment(env) == "oracle"
    meta, labels, by_hash = apply_oracle_placements(
        runtime, occurrences, predictions, environ=env)
    assert meta["budgets"] == {"gpu": 1, "cpu": 1, "disk": 1}
    assert runtime == {"a": "gpu", "b": "cpu", "c": "disk", "d": "drop"}
    assert labels == ["gpu", "gpu", "cpu", "disk"]
    assert by_hash["a"]["gpu"] == 2
    assert occurrences[0]["runtime_placement"] == "gpu"
    assert predictions[-1]["prediction"] == "drop"
    with pytest.raises(ValueError, match="smoke forcing"):
        apply_oracle_placements(runtime, occurrences, predictions,
                                smoke_force_min_unique_per_placement=1,
                                environ=env)
    with pytest.raises(ValueError, match="selective"):
        oracle_mode_from_environment({**env, "VLLM_VPC_POLICY": "window"})
