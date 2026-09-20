#!/usr/bin/env python3
"""Profile block-score patterns for KV-cache compression decisions.

This script does not compress or drop anything. It profiles the shape of
attention-derived block importance on a workload and summarizes whether the
workload looks more suitable for:

- block dropping / pruning
- recency-aware pruning
- uniform quantization
- hybrid keep/quantize/drop

It uses real LinearRAG-style prompts when requested and produces:
- per-sample block score statistics
- aggregate score concentration / recency bias metrics
- rule-based recommendations
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Hierarchical_KV.experiments.kv_cache_compression_study import (
    blockify_token_scores,
    build_dense_linearrag_retrieval,
    build_prompt_text,
    load_retrieval_json,
    token_positions_from_offsets,
)


def gini(x: torch.Tensor) -> float:
    if x.numel() == 0:
        return 0.0
    x = x.float().clamp_min(0)
    total = float(x.sum().item())
    if total <= 0:
        return 0.0
    xs = torch.sort(x).values
    n = xs.numel()
    idx = torch.arange(1, n + 1, dtype=torch.float32)
    return float(((2 * idx - n - 1) * xs).sum().item() / (n * total))


def normalized_entropy(x: torch.Tensor) -> float:
    if x.numel() <= 1:
        return 0.0
    p = x.float() / x.sum().clamp_min(1e-8)
    p = p.clamp_min(1e-8)
    ent = -(p * p.log()).sum().item()
    return float(ent / math.log(x.numel()))


def effective_block_fraction(x: torch.Tensor) -> float:
    if x.numel() == 0:
        return 0.0
    p = x.float() / x.sum().clamp_min(1e-8)
    return float(1.0 / (p.square().sum().item() * x.numel()))


def top_mass(x: torch.Tensor, frac: float) -> float:
    if x.numel() == 0:
        return 0.0
    k = max(1, int(math.ceil(x.numel() * frac)))
    vals = torch.topk(x, k=k).values
    return float(vals.sum().item() / x.sum().clamp_min(1e-8).item())


def recent_mass(x: torch.Tensor, recent_blocks: int) -> float:
    if x.numel() == 0:
        return 0.0
    recent_blocks = max(0, min(recent_blocks, x.numel()))
    if recent_blocks == 0:
        return 0.0
    return float(x[-recent_blocks:].sum().item() / x.sum().clamp_min(1e-8).item())


def oldest_mass(x: torch.Tensor, frac: float) -> float:
    if x.numel() == 0:
        return 0.0
    k = max(1, int(math.ceil(x.numel() * frac)))
    return float(x[:k].sum().item() / x.sum().clamp_min(1e-8).item())


def newest_mass(x: torch.Tensor, frac: float) -> float:
    if x.numel() == 0:
        return 0.0
    k = max(1, int(math.ceil(x.numel() * frac)))
    return float(x[-k:].sum().item() / x.sum().clamp_min(1e-8).item())


def minimum_blocks_for_mass(x: torch.Tensor, target_mass: float) -> int:
    if x.numel() == 0:
        return 0
    vals = torch.sort(x, descending=True).values
    csum = torch.cumsum(vals, dim=0)
    target = x.sum() * target_mass
    idx = int(torch.searchsorted(csum, target).item())
    return idx + 1


def recency_correlation(x: torch.Tensor) -> float:
    if x.numel() <= 1:
        return 0.0
    pos = torch.arange(x.numel(), dtype=torch.float32)
    x = x.float()
    pos = (pos - pos.mean()) / pos.std().clamp_min(1e-8)
    x = (x - x.mean()) / x.std().clamp_min(1e-8)
    return float((pos * x).mean().item())


def load_real_block_scores(
    model_name: str,
    retrieval_items: list[dict[str, Any]],
    block_size: int,
    max_seq_len: int,
    max_samples: int | None,
    layer_index: int,
    device: str,
    dtype: str,
) -> list[dict[str, Any]]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[dtype]

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()

    items = retrieval_items[:max_samples] if max_samples is not None else retrieval_items
    out: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        question = str(item["question"])
        passages = list(item.get("sorted_passage", []))
        prompt, q_start, q_end = build_prompt_text(passages, question)
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_seq_len,
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        question_positions = token_positions_from_offsets(offsets, q_start, q_end)
        if not question_positions:
            continue
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            outputs = model(
                **enc,
                output_attentions=True,
                use_cache=False,
            )
        attn = outputs.attentions[layer_index][0].float()
        seq_len = int(attn.shape[-1])
        prompt_positions = list(range(seq_len))
        attn_slice = attn[:, question_positions][:, :, prompt_positions]
        token_weights = attn_slice.mean(dim=(0, 1)).cpu()
        token_weights = token_weights / token_weights.sum().clamp_min(1e-8)
        block_scores = blockify_token_scores(token_weights, block_size)
        out.append({
            "sample_id": f"linearrag_{idx}",
            "question": question,
            "num_tokens": seq_len,
            "num_blocks": int(block_scores.numel()),
            "block_scores": block_scores,
        })
    return out


def build_synthetic_block_scores(
    seeds: list[int],
    block_size: int,
    num_blocks: int,
) -> list[dict[str, Any]]:
    out = []
    for seed in seeds:
        g = torch.Generator(device="cpu").manual_seed(seed)
        ranks = torch.arange(1, num_blocks + 1, dtype=torch.float32)
        probs = 1.0 / torch.pow(ranks, 1.15)
        probs = probs / probs.sum()
        token_count = num_blocks * block_size
        sampled = torch.multinomial(probs, token_count, replacement=True, generator=g)
        token_weights = torch.zeros(token_count, dtype=torch.float32)
        for idx in sampled.tolist():
            start = idx * block_size
            pos = start + int(torch.randint(0, block_size, (1,), generator=g).item())
            token_weights[pos] += 1
        token_weights = token_weights / token_weights.sum().clamp_min(1e-8)
        out.append({
            "sample_id": f"synthetic_{seed}",
            "question": f"synthetic_{seed}",
            "num_tokens": token_count,
            "num_blocks": num_blocks,
            "block_scores": blockify_token_scores(token_weights, block_size),
        })
    return out


def sample_profile(rec: dict[str, Any]) -> dict[str, Any]:
    scores = rec["block_scores"].float()
    top10 = top_mass(scores, 0.10)
    top25 = top_mass(scores, 0.25)
    top50 = top_mass(scores, 0.50)
    recent4 = recent_mass(scores, 4)
    recent8 = recent_mass(scores, 8)
    keep90 = minimum_blocks_for_mass(scores, 0.90)
    keep95 = minimum_blocks_for_mass(scores, 0.95)
    keep99 = minimum_blocks_for_mass(scores, 0.99)
    n = max(1, scores.numel())
    return {
        "sample_id": rec["sample_id"],
        "question": rec["question"],
        "num_tokens": rec["num_tokens"],
        "num_blocks": rec["num_blocks"],
        "gini": gini(scores),
        "normalized_entropy": normalized_entropy(scores),
        "effective_block_fraction": effective_block_fraction(scores),
        "top10_mass": top10,
        "top25_mass": top25,
        "top50_mass": top50,
        "recent4_mass": recent4,
        "recent8_mass": recent8,
        "newest25_mass": newest_mass(scores, 0.25),
        "oldest25_mass": oldest_mass(scores, 0.25),
        "recency_correlation": recency_correlation(scores),
        "blocks_for_90_mass": keep90,
        "blocks_for_95_mass": keep95,
        "blocks_for_99_mass": keep99,
        "fraction_blocks_for_90_mass": keep90 / n,
        "fraction_blocks_for_95_mass": keep95 / n,
        "fraction_blocks_for_99_mass": keep99 / n,
    }


def mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(r[key]) for r in rows) / max(1, len(rows))


def recommend(summary: dict[str, float]) -> list[str]:
    recs: list[str] = []

    if summary["fraction_blocks_for_95_mass_mean"] <= 0.30 and summary["top25_mass_mean"] >= 0.90:
        recs.append("Strong pruning signal: top-scoring blocks carry most of the mass.")
    elif summary["fraction_blocks_for_95_mass_mean"] <= 0.50:
        recs.append("Moderate pruning signal: try drop-bottom-25% and hybrid drop+int8.")
    else:
        recs.append("Weak pruning signal: prefer quantization-first over aggressive dropping.")

    if summary["recent4_mass_mean"] >= 0.50 or summary["recency_correlation_mean"] >= 0.20:
        recs.append("Strong recency bias: always protect the newest few blocks.")
    elif summary["recent8_mass_mean"] >= 0.50:
        recs.append("Moderate recency bias: recent-window protection is likely useful.")
    else:
        recs.append("Limited recency bias: score-based pruning may beat recency-only pruning.")

    if summary["normalized_entropy_mean"] >= 0.85 or summary["effective_block_fraction_mean"] >= 0.45:
        recs.append("Scores are diffuse: uniform int8 is a safer default than dropping.")
    elif summary["normalized_entropy_mean"] <= 0.65:
        recs.append("Scores are concentrated: hybrid keep/quantize/drop is promising.")

    if summary["top10_mass_mean"] >= 0.60:
        recs.append("A tiny hot set dominates attention: test hot-block fp16 plus cold-block int8.")

    return recs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--workload-mode", choices=["synthetic", "linearrag"], default="linearrag")
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--output", type=Path,
                   default=Path("Hierarchical_KV/outputs/kv_cache_compression_profile/profile.json"))

    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--synthetic-num-blocks", type=int, default=256)

    p.add_argument("--retrieval-json", type=Path, default=None)
    p.add_argument("--dataset-name", type=str, default="hotpotqa")
    p.add_argument("--dataset-root", type=Path, default=Path("Hierarchical_KV/dataset_linear-rag"))
    p.add_argument("--linearrag-import-dir", type=Path, default=Path("Hierarchical_KV/LinearRAG/import"))
    p.add_argument("--embedding-model", type=str,
                   default="sentence-transformers/all-mpnet-base-v2")
    p.add_argument("--retrieval-top-k", type=int, default=5)
    p.add_argument("--max-samples", type=int, default=32)

    p.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B")
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--layer-index", type=int, default=-1)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.workload_mode == "synthetic":
        records = build_synthetic_block_scores(
            seeds=args.seeds,
            block_size=args.block_size,
            num_blocks=args.synthetic_num_blocks,
        )
    else:
        if args.retrieval_json is not None:
            retrieval_items = load_retrieval_json(args.retrieval_json)
        else:
            retrieval_items = build_dense_linearrag_retrieval(
                dataset_name=args.dataset_name,
                dataset_root=args.dataset_root,
                linearrag_import_dir=args.linearrag_import_dir,
                embedding_model_name=args.embedding_model,
                retrieval_top_k=args.retrieval_top_k,
                max_samples=args.max_samples,
                device=args.device,
            )
        records = load_real_block_scores(
            model_name=args.model,
            retrieval_items=retrieval_items,
            block_size=args.block_size,
            max_seq_len=args.max_seq_len,
            max_samples=args.max_samples,
            layer_index=args.layer_index,
            device=args.device,
            dtype=args.dtype,
        )

    samples = [sample_profile(r) for r in records]
    summary = {
        "num_samples": len(samples),
        "block_size": args.block_size,
        "gini_mean": mean_metric(samples, "gini"),
        "normalized_entropy_mean": mean_metric(samples, "normalized_entropy"),
        "effective_block_fraction_mean": mean_metric(samples, "effective_block_fraction"),
        "top10_mass_mean": mean_metric(samples, "top10_mass"),
        "top25_mass_mean": mean_metric(samples, "top25_mass"),
        "top50_mass_mean": mean_metric(samples, "top50_mass"),
        "recent4_mass_mean": mean_metric(samples, "recent4_mass"),
        "recent8_mass_mean": mean_metric(samples, "recent8_mass"),
        "newest25_mass_mean": mean_metric(samples, "newest25_mass"),
        "oldest25_mass_mean": mean_metric(samples, "oldest25_mass"),
        "recency_correlation_mean": mean_metric(samples, "recency_correlation"),
        "fraction_blocks_for_90_mass_mean": mean_metric(samples, "fraction_blocks_for_90_mass"),
        "fraction_blocks_for_95_mass_mean": mean_metric(samples, "fraction_blocks_for_95_mass"),
        "fraction_blocks_for_99_mass_mean": mean_metric(samples, "fraction_blocks_for_99_mass"),
    }
    recommendations = recommend(summary)

    payload = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "summary": summary,
        "recommendations": recommendations,
        "samples": samples,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print("\nRecommendations:")
    for rec in recommendations:
        print(f"- {rec}")
    print(f"\nsaved -> {args.output}")


if __name__ == "__main__":
    main()
