#!/usr/bin/env python3
"""Precompute request-conditioned Short-Q chunk placement for MP experiments.

Runtime metadata is deliberately minimal: a JSON object mapping rolling LMCache
chunk-hash hex strings to final L0/L1/L2 decisions. Rich diagnostics (including
per-occurrence request-conditioned predictions and timing) are saved separately
and are never consumed by the runtime placement path.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable, TypeVar

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "LMCache"))

from experiments.mp_nognn_project import (  # noqa: E402
    build_vllm_prompts_from_retrieval_results,
    dense_retrieve_from_import,
    load_embedding_model,
    load_questions,
    reorder_requests,
)
from Hierarchical_KV.shortq_placement.block_prediction import get_block_predictions  # noqa: E402
from Hierarchical_KV.shortq_placement.chunk_voting import (
    make_block_tier_vote,
    selected_vote_policy_name,
    vote_chunk_placement,
)
from Hierarchical_KV.shortq_placement.duplicate_resolution import resolve_duplicate  # noqa: E402
from Hierarchical_KV.shortq_placement.model import (  # noqa: E402
    load_ranker_from_checkpoint,
    pool_hidden_to_blocks,
    recency_and_passage_ids,
)
from Hierarchical_KV.shortq_placement.prediction_cache import (  # noqa: E402
    lookup_prediction,
    store_prediction,
)
from Hierarchical_KV.shortq_placement.runtime_metadata import write_runtime_metadata  # noqa: E402
from Hierarchical_KV.shortq_placement.tier_mapping import (  # noqa: E402
    CLASS_NAMES,
    map_block_predictions_to_tiers,
)
from lmcache.v1.multiprocess.token_hasher import TokenHasher  # noqa: E402

T = TypeVar("T")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", default="medical")
    p.add_argument("--dataset_root", default=str(REPO_ROOT / "Hierarchical_KV/LinearRAG/dataset"))
    p.add_argument("--linearrag_import_dir", default=str(REPO_ROOT / "Hierarchical_KV/LinearRAG/import"))
    p.add_argument("--embedding_model", default=str(REPO_ROOT / "Hierarchical_KV/LinearRAG/model/all-mpnet-base-v2"))
    p.add_argument("--questions_json", default="")
    p.add_argument("--max_questions", type=int, default=650)
    p.add_argument("--retrieval_top_k", type=int, default=5)
    p.add_argument("--request_order", default="legacy_prefix_hash")
    p.add_argument("--prefix_sort_depth", type=int, default=2)
    p.add_argument("--request_order_seed", type=int, default=0)
    p.add_argument("--serving_model", required=True)
    p.add_argument("--feature_model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--checkpoint", default=str(REPO_ROOT / "Hierarchical_KV/shortq_placement/model.pt"))
    p.add_argument("--chunk_size", type=int, default=512)
    p.add_argument("--max_seq_len", type=int, default=8000)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--inference_batch_size", type=int, default=1)
    p.add_argument("--runtime_metadata", required=True)
    p.add_argument(
        "--vpc_importance_sidecar",
        default="",
        help=(
            "Optional JSON output mapping reordered request index to native "
            "16-token block importance tiers for GNN-aware vLLM VPC."
        ),
    )
    p.add_argument("--placement_trace", required=True)
    p.add_argument("--hash_prediction_trace", required=True)
    p.add_argument("--timing_trace", required=True)
    p.add_argument("--summary", required=True)
    p.add_argument(
        "--smoke_force_min_unique_per_tier",
        type=int,
        default=0,
        help=(
            "Smoke-only validation aid. If >0, deterministically override the minimal "
            "number of runtime hash entries needed to exercise every L0/L1/L2 path. "
            "Raw GNN candidate predictions remain unchanged in the diagnostic traces."
        ),
    )
    return p.parse_args()


def _load_graph(import_root: Path, dataset: str, device: torch.device):
    try:
        import igraph as ig
    except Exception as exc:  # pragma: no cover - cluster dependency gate
        raise RuntimeError("python-igraph is required for Short-Q graph conditioning") from exc

    ds = import_root / dataset
    graph_path = ds / "LinearRAG.graphml"
    passage_path = ds / "passage_embedding.parquet"
    entity_path = ds / "entity_embedding.parquet"
    for path in (graph_path, passage_path, entity_path):
        if not path.exists():
            raise FileNotFoundError(f"required Short-Q graph asset missing: {path}")

    passage_df = pd.read_parquet(passage_path)
    entity_df = pd.read_parquet(entity_path)
    passage_emb = dict(zip(passage_df["hash_id"].astype(str), passage_df["embedding"]))
    entity_emb = dict(zip(entity_df["hash_id"].astype(str), entity_df["embedding"]))
    passage_text_to_hash = dict(zip(passage_df["text"].astype(str), passage_df["hash_id"].astype(str)))

    graph = ig.Graph.Read_GraphML(str(graph_path))
    node_names = [str(x) for x in graph.vs["name"]]
    node_to_idx = {name: i for i, name in enumerate(node_names)}
    any_emb = next(iter(passage_emb.values()), None)
    if any_emb is None:
        any_emb = next(iter(entity_emb.values()), None)
    if any_emb is None:
        raise ValueError("graph embedding parquet files contain no embeddings")
    node_features = torch.zeros((len(node_names), len(any_emb)), dtype=torch.float32)
    for i, name in enumerate(node_names):
        emb = passage_emb.get(name)
        if emb is None:
            emb = entity_emb.get(name)
        if emb is not None:
            node_features[i] = torch.from_numpy(np.array(emb, dtype=np.float32, copy=True))

    src: list[int] = []
    dst: list[int] = []
    weights: list[float] = []
    edge_weights = graph.es["weight"] if "weight" in graph.es.attributes() else [1.0] * graph.ecount()
    for (u, v), w in zip(graph.get_edgelist(), edge_weights, strict=True):
        src.extend((u, v))
        dst.extend((v, u))
        weights.extend((float(w), float(w)))
    edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
    edge_weight = torch.tensor(weights, dtype=torch.float32, device=device)
    return node_features.to(device), edge_index, edge_weight, node_to_idx, passage_text_to_hash


def _retrieved_indices(
    record: dict[str, Any],
    passage_text_to_hash: dict[str, str],
    node_to_idx: dict[str, int],
    max_retrieved: int,
) -> list[int]:
    out: list[int] = []
    for text in record.get("sorted_passage", [])[:max_retrieved]:
        hash_id = passage_text_to_hash.get(str(text))
        out.append(node_to_idx.get(hash_id, -1) if hash_id is not None else -1)
    out.extend([-1] * (max_retrieved - len(out)))
    return out[:max_retrieved]


def _load_feature_model(name: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, low_cpu_mem_usage=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tokenizer, model


@torch.inference_mode()
def _block_features_batch(
    token_id_rows: list[list[int]],
    model,
    device: torch.device,
    block_size: int,
    pad_token_id: int,
):
    if not token_id_rows:
        raise ValueError("empty inference batch")
    max_len = max(len(row) for row in token_id_rows)
    ids = torch.full(
        (len(token_id_rows), max_len),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for i, row in enumerate(token_id_rows):
        n = len(row)
        ids[i, :n] = torch.tensor(row, dtype=torch.long, device=device)
        mask[i, :n] = True
    backbone = getattr(model, "model", None)
    if backbone is None:
        out = model(
            input_ids=ids,
            attention_mask=mask.long(),
            output_hidden_states=True,
            use_cache=False,
        )
        hidden = out.hidden_states[-1].float()
    else:
        out = backbone(input_ids=ids, attention_mask=mask.long(), use_cache=False)
        hidden = out.last_hidden_state.float()
    return pool_hidden_to_blocks(hidden, mask, block_size)


def _counter_dict(values: list[str]) -> dict[str, int]:
    c = Counter(values)
    return {k: int(c.get(k, 0)) for k in ("L0", "L1", "L2")}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(device: torch.device, fn: Callable[[], T]) -> tuple[T, float]:
    _sync(device)
    t0 = time.perf_counter()
    value = fn()
    _sync(device)
    return value, time.perf_counter() - t0


def _dist(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "sum": 0.0, "mean": None, "p50": None, "p95": None, "max": None}
    vals = sorted(values)

    def pct(p: float) -> float:
        x = (len(vals) - 1) * p
        lo = math.floor(x)
        hi = math.ceil(x)
        if lo == hi:
            return vals[lo]
        return vals[lo] * (hi - x) + vals[hi] * (x - lo)

    return {
        "n": len(vals),
        "sum": float(sum(vals)),
        "mean": float(sum(vals) / len(vals)),
        "p50": float(pct(0.50)),
        "p95": float(pct(0.95)),
        "max": float(vals[-1]),
    }




def _write_vpc_importance_sidecar(
    path: str | Path,
    occurrences: list[dict[str, Any]],
    blocks_per_chunk: int,
) -> None:
    """Write stable per-request native-block labels for vLLM prefix caching.

    Short-Q votes at 512-token LMCache chunk granularity. vLLM's physical APC
    blocks remain 16 tokens, so every full logical chunk contributes the same
    label to ``blocks_per_chunk`` consecutive native blocks. The final resolved
    runtime tier is used so shared hashes have stable semantics across requests.
    """
    tier_to_importance = {"L0": "gpu", "L1": "cpu", "L2": "disk"}
    per_request: dict[str, dict[str, str]] = {}
    for row in occurrences:
        request_idx = str(int(row["request_order_index"]))
        chunk_idx = int(row["chunk_index"])
        tier = str(row["runtime_tier"])
        importance = tier_to_importance[tier]
        request_map = per_request.setdefault(request_idx, {})
        start = chunk_idx * blocks_per_chunk
        for block_idx in range(start, start + blocks_per_chunk):
            request_map[str(block_idx)] = importance

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(per_request, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _force_smoke_tier_coverage(
    runtime: dict[str, str], min_unique_per_tier: int
) -> list[dict[str, str]]:
    """Deterministically force minimal path coverage for smoke tests only."""
    if min_unique_per_tier <= 0:
        return []
    required = min_unique_per_tier * 3
    if len(runtime) < required:
        raise RuntimeError(
            f"smoke tier coverage requires at least {required} unique hashes, got {len(runtime)}"
        )
    counts = Counter(runtime.values())
    overrides: list[dict[str, str]] = []
    ordered_hashes = list(runtime.keys())  # insertion order = first-seen request order
    for target in ("L0", "L1", "L2"):
        while counts[target] < min_unique_per_tier:
            donor_tiers = sorted(
                (tier for tier in ("L0", "L1", "L2") if counts[tier] > min_unique_per_tier),
                key=lambda tier: (-counts[tier], tier),
            )
            if not donor_tiers:
                raise RuntimeError(f"cannot force smoke coverage for {target}; counts={dict(counts)}")
            donor = donor_tiers[0]
            chosen = next(h for h in ordered_hashes if runtime[h] == donor)
            runtime[chosen] = target
            counts[donor] -= 1
            counts[target] += 1
            overrides.append({"chunk_hash": chosen, "from": donor, "to": target})
    return overrides


def main() -> None:
    args = parse_args()
    if args.chunk_size % 16 != 0:
        raise ValueError("LMCache chunk size must be a multiple of the 16-token Short-Q block size")
    blocks_per_chunk = args.chunk_size // 16
    device = torch.device(args.device)
    started = time.monotonic()

    serving_tok = AutoTokenizer.from_pretrained(args.serving_model, use_fast=True)
    retrieval_prompt_started = time.monotonic()

    # Build the exact serving prompts first. The retrieval embedding model may
    # itself use CUDA, so release it before loading the frozen 8B feature model.
    qargs = argparse.Namespace(
        question=None,
        questions_json=args.questions_json or None,
        dataset_root=args.dataset_root,
        dataset_name=args.dataset_name,
    )
    questions = load_questions(qargs)[: args.max_questions]
    embedding_model = load_embedding_model(Path(args.embedding_model))
    retrieval = dense_retrieve_from_import(
        import_root=Path(args.linearrag_import_dir),
        dataset_name=args.dataset_name,
        questions=questions,
        embedding_model=embedding_model,
        retrieval_top_k=args.retrieval_top_k,
    )
    prompts, records = build_vllm_prompts_from_retrieval_results(retrieval, serving_tok)
    prompts, records = reorder_requests(
        prompts,
        records,
        order_mode=args.request_order,
        prefix_sort_depth=args.prefix_sort_depth,
        seed=args.request_order_seed,
    )
    del embedding_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    retrieval_prompt_seconds = time.monotonic() - retrieval_prompt_started

    model_setup_started = time.monotonic()
    feature_tok, feature_model = _load_feature_model(args.feature_model, device)
    ranker, cfg = load_ranker_from_checkpoint(args.checkpoint, device)
    ranker.eval()
    ranker.requires_grad_(False)
    block_size = int(cfg.get("block_size", 16))
    if block_size != 16:
        raise ValueError(f"first integration expects 16-token Short-Q blocks, checkpoint has {block_size}")
    capacity = int(cfg["max_blocks"]) * block_size
    if args.max_seq_len > capacity:
        raise ValueError(f"max_seq_len {args.max_seq_len} exceeds Short-Q capacity {capacity}")

    graph_load_started = time.monotonic()
    node_features, edge_index, edge_weight, node_to_idx, passage_text_to_hash = _load_graph(
        Path(args.linearrag_import_dir), args.dataset_name, device
    )
    graph_load_seconds = time.monotonic() - graph_load_started
    if node_features.shape[1] != int(cfg["node_dim"]):
        raise ValueError(f"graph node dim {node_features.shape[1]} != checkpoint node dim {cfg['node_dim']}")
    with torch.inference_mode():
        node_repr, graph_encode_seconds = _timed(
            device, lambda: ranker.encode_graph(node_features, edge_index, edge_weight)
        )
    model_setup_seconds = time.monotonic() - model_setup_started

    hasher = TokenHasher(chunk_size=args.chunk_size, hash_algorithm="blake3")
    runtime: dict[str, str] = {}
    occurrences: list[dict[str, Any]] = []
    hash_prediction_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    occurrence_tiers: list[str] = []
    candidate_by_hash: dict[str, Counter[str]] = defaultdict(Counter)
    conflict_occurrences = 0
    cache_hits = 0
    exact_alignment_requests = 0
    class_occurrences = Counter()

    max_retrieved = int(cfg["max_retrieved"])
    if args.inference_batch_size < 1:
        raise ValueError("--inference_batch_size must be >= 1")
    pad_token_id = feature_tok.pad_token_id
    if pad_token_id is None:
        pad_token_id = feature_tok.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0

    batch_timing_rows: list[dict[str, Any]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    inference_started = time.monotonic()
    for batch_start in range(0, len(prompts), args.inference_batch_size):
        batch_end = min(batch_start + args.inference_batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]
        batch_records = records[batch_start:batch_end]
        serving_id_rows: list[list[int]] = []
        tokenization_seconds_rows: list[float] = []
        source_indices: list[int] = []

        for local_index, (prompt, record) in enumerate(
            zip(batch_prompts, batch_records, strict=True)
        ):
            request_order_index = batch_start + local_index
            tok_t0 = time.perf_counter()
            serving_ids = serving_tok.encode(prompt, add_special_tokens=True)
            feature_ids = feature_tok.encode(prompt, add_special_tokens=True)
            tokenization_seconds = time.perf_counter() - tok_t0
            source_index = int(record.get("source_index", request_order_index))
            if serving_ids != feature_ids:
                first = next(
                    (i for i, (a, b) in enumerate(zip(serving_ids, feature_ids)) if a != b),
                    None,
                )
                raise RuntimeError(
                    f"serving/feature tokenizer mismatch at source_index={source_index}, "
                    f"request_order_index={request_order_index}, first_diff={first}, "
                    f"serving_len={len(serving_ids)}, feature_len={len(feature_ids)}"
                )
            exact_alignment_requests += 1
            if len(serving_ids) > args.max_seq_len:
                raise RuntimeError(
                    f"prompt {source_index} has {len(serving_ids)} tokens > max_seq_len={args.max_seq_len}; "
                    "do not silently truncate placement relative to serving"
                )
            serving_id_rows.append(serving_ids)
            tokenization_seconds_rows.append(tokenization_seconds)
            source_indices.append(source_index)

        (batch_block_features, batch_block_mask), batch_feature_seconds = _timed(
            device,
            lambda: _block_features_batch(
                serving_id_rows,
                feature_model,
                device,
                block_size,
                pad_token_id,
            ),
        )
        if batch_block_features.shape[-1] != int(cfg["llm_feat_dim"]):
            raise ValueError(
                f"feature dim {batch_block_features.shape[-1]} != checkpoint {cfg['llm_feat_dim']}"
            )
        recency, passage_ids = recency_and_passage_ids(batch_block_mask, max_retrieved)
        retrieved = torch.tensor(
            [
                _retrieved_indices(record, passage_text_to_hash, node_to_idx, max_retrieved)
                for record in batch_records
            ],
            dtype=torch.long,
            device=device,
        )
        with torch.inference_mode():
            (batch_class_logits, batch_rank_scores), batch_gnn_forward_seconds = _timed(
                device,
                lambda: ranker.forward_blocks(
                    batch_block_features,
                    batch_block_mask,
                    recency,
                    passage_ids,
                    node_repr,
                    retrieved,
                ),
            )

        batch_size_actual = len(batch_prompts)
        amortized_feature_seconds = batch_feature_seconds / batch_size_actual
        amortized_gnn_forward_seconds = batch_gnn_forward_seconds / batch_size_actual
        batch_timing_rows.append(
            {
                "batch_start": batch_start,
                "batch_size": batch_size_actual,
                "max_prompt_tokens": max(len(x) for x in serving_id_rows),
                "prompt_tokens": sum(len(x) for x in serving_id_rows),
                "feature_extraction_seconds": batch_feature_seconds,
                "gnn_forward_seconds": batch_gnn_forward_seconds,
            }
        )
        print(
            f"[SHORTQ_GNN_BATCH_TIMING] batch_start={batch_start} batch_size={batch_size_actual} "
            f"max_tokens={max(len(x) for x in serving_id_rows)} "
            f"feature_s={batch_feature_seconds:.6f} gnn_forward_s={batch_gnn_forward_seconds:.6f}",
            flush=True,
        )

        for local_index, record in enumerate(batch_records):
            request_order_index = batch_start + local_index
            source_index = source_indices[local_index]
            serving_ids = serving_id_rows[local_index]
            tokenization_seconds = tokenization_seconds_rows[local_index]
            block_mask = batch_block_mask[local_index : local_index + 1]
            class_logits = batch_class_logits[local_index]
            rank_scores = batch_rank_scores[local_index]

            post_t0 = time.perf_counter()
            valid_blocks = int(block_mask[0].sum().item())
            class_ids = get_block_predictions(
                class_logits[:valid_blocks], rank_scores[:valid_blocks]
            ).tolist()
            class_occurrences.update(int(x) for x in class_ids)
            
            # Preserve the normal argmax-derived L0/L1/L2 tier for every block,
            # but attach whether CPU was among that block's top-2 class logits.
            base_mapped = map_block_predictions_to_tiers(class_ids)

            cpu_class_id = CLASS_NAMES.index("cpu")

            top2_class_ids = torch.topk(
                        class_logits[:valid_blocks],
                        k=2,
                        dim=-1,
                    ).indices
            cpu_top2_flags = (
                (top2_class_ids == cpu_class_id)
                .any(dim=-1)
                .tolist()
            )

            mapped = [
                make_block_tier_vote(
                    tier,
                    cpu_top2=cpu_top2_flags[i],
                )
                for i, tier in enumerate(base_mapped)
            ]

            hashes = hasher.compute_chunk_hashes(serving_ids)
            full_chunks = len(serving_ids) // args.chunk_size
            if len(hashes) != full_chunks:
                raise AssertionError("TokenHasher full-chunk count mismatch")

            for chunk_index, chunk_hash in enumerate(hashes):
                h = chunk_hash.hex()
                cached = lookup_prediction(h)
                if cached is not None:
                    cache_hits += 1
                    candidate = cached
                else:
                    lo = chunk_index * blocks_per_chunk
                    hi = lo + blocks_per_chunk
                    block_tiers = mapped[lo:hi]
                    if len(block_tiers) != blocks_per_chunk:
                        raise RuntimeError(
                            "full LMCache chunk did not have 32 aligned block predictions"
                        )
                    candidate = vote_chunk_placement(block_tiers)
                    store_prediction(h, candidate)
                previous = runtime.get(h)
                resolved = resolve_duplicate(previous, candidate)
                conflict = previous is not None and previous != candidate
                if conflict:
                    conflict_occurrences += 1
                runtime[h] = resolved
                candidate_by_hash[h][candidate] += 1
                occurrence_tiers.append(candidate)
                class_slice = class_ids[
                    chunk_index * blocks_per_chunk : (chunk_index + 1) * blocks_per_chunk
                ]
                tier_slice = mapped[
                    chunk_index * blocks_per_chunk : (chunk_index + 1) * blocks_per_chunk
                ]
                occurrences.append(
                    {
                        "source_index": source_index,
                        "request_order_index": request_order_index,
                        "chunk_index": chunk_index,
                        "chunk_hash": h,
                        "candidate_tier": candidate,
                        "runtime_tier": resolved,
                        "duplicate_conflict": conflict,
                        "block_class_counts": {
                            CLASS_NAMES[i]: sum(1 for x in class_slice if int(x) == i)
                            for i in range(4)
                        },
                        "block_tier_counts": _counter_dict(tier_slice),
                    }
                )
                hash_prediction_rows.append(
                    {
                        "chunk_hash": h,
                        "source_index": source_index,
                        "request_order_index": request_order_index,
                        "chunk_index": chunk_index,
                        "prediction": candidate,
                        "duplicate_conflict": conflict,
                    }
                )

            postprocess_seconds = time.perf_counter() - post_t0
            total_prediction_seconds = (
                tokenization_seconds
                + amortized_feature_seconds
                + amortized_gnn_forward_seconds
                + postprocess_seconds
            )
            timing_row = {
                "source_index": source_index,
                "request_order_index": request_order_index,
                "prompt_tokens": len(serving_ids),
                "full_chunks": full_chunks,
                "tokenization_alignment_seconds": tokenization_seconds,
                "feature_extraction_seconds": amortized_feature_seconds,
                "gnn_forward_seconds": amortized_gnn_forward_seconds,
                "postprocess_seconds": postprocess_seconds,
                "total_prediction_pipeline_seconds": total_prediction_seconds,
                "inference_batch_size": batch_size_actual,
            }
            timing_rows.append(timing_row)
            print(
                f"[SHORTQ_GNN_TIMING] source_index={source_index} request_order_index={request_order_index} "
                f"tokens={len(serving_ids)} feature_s_amortized={amortized_feature_seconds:.6f} "
                f"gnn_forward_s_amortized={amortized_gnn_forward_seconds:.6f} "
                f"postprocess_s={postprocess_seconds:.6f} total_prediction_s_amortized={total_prediction_seconds:.6f}",
                flush=True,
            )
            print(
                f"[SHORTQ_PLACEMENT] {request_order_index + 1}/{len(prompts)} tokens={len(serving_ids)} "
                f"full_chunks={full_chunks} unique_hashes={len(runtime)} conflicts={conflict_occurrences}",
                flush=True,
            )

    inference_seconds = time.monotonic() - inference_started
    peak_cuda_memory_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )

    smoke_overrides = _force_smoke_tier_coverage(
        runtime, args.smoke_force_min_unique_per_tier
    )
    override_hashes = {row["chunk_hash"] for row in smoke_overrides}
    for row in occurrences:
        row["runtime_tier"] = runtime[row["chunk_hash"]]
        row["smoke_runtime_override"] = row["chunk_hash"] in override_hashes
    for row in hash_prediction_rows:
        row["runtime_tier"] = runtime[row["chunk_hash"]]
        row["smoke_runtime_override"] = row["chunk_hash"] in override_hashes
    if smoke_overrides:
        print(
            f"[SHORTQ_SMOKE_TIER_OVERRIDE] count={len(smoke_overrides)} "
            f"details={json.dumps(smoke_overrides, sort_keys=True)}",
            flush=True,
        )

    metadata_started = time.monotonic()
    runtime_path = Path(args.runtime_metadata)
    trace_path = Path(args.placement_trace)
    hash_trace_path = Path(args.hash_prediction_trace)
    timing_path = Path(args.timing_trace)
    summary_path = Path(args.summary)
    write_runtime_metadata(runtime_path, runtime)
    if args.vpc_importance_sidecar:
        _write_vpc_importance_sidecar(
            args.vpc_importance_sidecar, occurrences, blocks_per_chunk
        )

    for path, rows in (
        (trace_path, occurrences),
        (hash_trace_path, hash_prediction_rows),
        (timing_path, timing_rows),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")

    conflicting_hashes = sum(1 for c in candidate_by_hash.values() if len(c) > 1)
    repeated_hashes = sum(1 for c in candidate_by_hash.values() if sum(c.values()) > 1)
    metadata_write_seconds = time.monotonic() - metadata_started
    elapsed = time.monotonic() - started

    timing_summary = {
        "feature_extraction_seconds": _dist([r["feature_extraction_seconds"] for r in timing_rows]),
        "gnn_forward_seconds": _dist([r["gnn_forward_seconds"] for r in timing_rows]),
        "postprocess_seconds": _dist([r["postprocess_seconds"] for r in timing_rows]),
        "total_prediction_pipeline_seconds": _dist(
            [r["total_prediction_pipeline_seconds"] for r in timing_rows]
        ),
    }
    summary = {
        "elapsed_seconds": elapsed,
        "retrieval_and_prompt_build_seconds": retrieval_prompt_seconds,
        "model_and_graph_setup_seconds": model_setup_seconds,
        "graph_load_seconds": graph_load_seconds,
        "graph_encode_seconds": graph_encode_seconds,
        "request_prediction_loop_seconds": inference_seconds,
        "inference_batch_size": args.inference_batch_size,
        "request_prediction_throughput_rps": (len(prompts) / inference_seconds if inference_seconds else None),
        "peak_cuda_memory_bytes": peak_cuda_memory_bytes,
        "batch_timing": batch_timing_rows,
        "metadata_write_seconds": metadata_write_seconds,
        "prediction_timing": timing_summary,
        "requests": len(prompts),
        "token_alignment_exact_requests": exact_alignment_requests,
        "prediction_cache_mode": "dormant",
        "prediction_cache_hits": cache_hits,
        "block_policy": "argmax_class_logits",
        "class_to_tier_mapping": {"drop": "L2", "disk": "L2", "cpu": "L1", "gpu": "L0"},
        "chunk_vote": selected_vote_policy_name(),
        "duplicate_resolution": "first_seen_wins",
        "block_class_counts": {CLASS_NAMES[i]: int(class_occurrences[i]) for i in range(4)},
        "chunk_occurrences": len(occurrences),
        "unique_chunk_hashes": len(runtime),
        "occurrence_candidate_tier_counts": _counter_dict(occurrence_tiers),
        "unique_runtime_tier_counts": _counter_dict(list(runtime.values())),
        "repeated_hashes": repeated_hashes,
        "conflicting_hashes": conflicting_hashes,
        "conflicting_hash_fraction_among_repeated": (
            conflicting_hashes / repeated_hashes if repeated_hashes else 0.0
        ),
        "conflict_occurrences": conflict_occurrences,
        "smoke_force_min_unique_per_tier": args.smoke_force_min_unique_per_tier,
        "smoke_runtime_overrides": smoke_overrides,
        "runtime_metadata": str(runtime_path),
        "vpc_importance_sidecar": args.vpc_importance_sidecar or None,
        "placement_trace": str(trace_path),
        "hash_prediction_trace": str(hash_trace_path),
        "timing_trace": str(timing_path),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "[SHORTQ_GNN_TIMING_SUMMARY] "
        + json.dumps(timing_summary, sort_keys=True),
        flush=True,
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
