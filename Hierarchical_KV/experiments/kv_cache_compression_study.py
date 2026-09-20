#!/usr/bin/env python3
"""Study block-level keep/quantize/drop policies for vLLM-style KV blocks.

This script has two goals:
1. Use real LinearRAG prompts when available.
2. Report signals that help decide whether a vLLM block should be kept,
   quantized, or dropped.

There are two workload modes:
- synthetic: a fast sanity check with synthetic Q/K/V-like tensors
- linearrag: real prompts built from LinearRAG retrieval outputs or dataset files

The real-workload path uses last-layer attention over prompt tokens and the
last-layer hidden states as a value-like proxy. That is not an exact KV-cache
kernel simulation, but it gives a grounded signal for block decisions:
- how much attention mass a block carries
- how much context-vector quality changes if a block is dropped
- how much value-like quantization hurts on real prompts
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


torch.set_grad_enabled(False)


@dataclass(frozen=True)
class QuantSpec:
    bits: int


@dataclass(frozen=True)
class BlockPolicy:
    name: str
    quant_bits: int | None = None
    drop_fraction: float = 0.0
    keep_recent_blocks: int = 0
    hot_fp16_ratio: float = 0.0


def bytes_for_dtype(dtype: str) -> int:
    if dtype in ("fp16", "bf16"):
        return 2
    if dtype == "fp32":
        return 4
    if dtype == "int8":
        return 1
    raise ValueError(f"Unsupported dtype label: {dtype}")


def quantized_code_bytes(numel: int, bits: int) -> int:
    return math.ceil(numel * bits / 8.0)


def make_blockwise_scales(x: torch.Tensor, qmax: int) -> torch.Tensor:
    # x shape: [num_blocks, block_size, dim]
    absmax = x.abs().amax(dim=(1, 2), keepdim=True)
    return (absmax / max(qmax, 1)).clamp_min(1e-8)


def quantize_blockwise(x: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    if bits == 8:
        qmax = 127
    elif bits == 4:
        qmax = 7
    else:
        raise ValueError(f"Unsupported quant bits: {bits}")
    scale = make_blockwise_scales(x, qmax)
    q = torch.round(x / scale).clamp(-qmax, qmax).to(torch.int8)
    return q, scale.squeeze(-1).squeeze(-1).to(torch.float16)


def dequantize_blockwise(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.to(torch.float32) * scale[:, None, None].to(torch.float32)


def build_block_ranges(num_tokens: int, block_size: int) -> list[tuple[int, int]]:
    return [
        (start, min(start + block_size, num_tokens))
        for start in range(0, num_tokens, block_size)
    ]


def blockify_token_scores(token_scores: torch.Tensor, block_size: int) -> torch.Tensor:
    blocks = []
    for start, end in build_block_ranges(token_scores.numel(), block_size):
        blocks.append(token_scores[start:end].sum())
    return torch.stack(blocks) if blocks else torch.empty(0, dtype=token_scores.dtype)


def pad_to_blocks(x: torch.Tensor, block_size: int) -> torch.Tensor:
    num_tokens, hidden_dim = x.shape
    num_blocks = math.ceil(num_tokens / block_size)
    padded = torch.zeros(num_blocks * block_size, hidden_dim, dtype=x.dtype, device=x.device)
    padded[:num_tokens] = x
    return padded.view(num_blocks, block_size, hidden_dim)


def flatten_blocks(x: torch.Tensor, valid_tokens: int) -> torch.Tensor:
    return x.view(-1, x.shape[-1])[:valid_tokens]


def cosine_scalar(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.view(1, -1), b.view(1, -1)).item())


def mean_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().mean().item())


def normalize_project_path(path: Path) -> Path:
    """Allow both repo-root-relative paths and paths prefixed with project name."""
    if path.exists() or path.is_absolute():
        return path

    parts = path.parts
    if parts and parts[0] == "Hierarchical_KV":
        candidate = Path(*parts[1:])
        if candidate.exists():
            return candidate

    return path


def kl_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    eps = 1e-8
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return float((p * (p.log() - q.log())).sum().item())


def retained_attention_mass(mask: torch.Tensor, block_scores: torch.Tensor) -> float:
    total = float(block_scores.sum().item())
    if total <= 0:
        return 0.0
    return float(block_scores[mask].sum().item() / total)


def choose_kept_blocks(
    block_scores: torch.Tensor,
    policy: BlockPolicy,
) -> torch.Tensor:
    num_blocks = block_scores.numel()
    if num_blocks == 0:
        return torch.zeros(0, dtype=torch.bool)

    keep_mask = torch.ones(num_blocks, dtype=torch.bool)
    protected = torch.zeros(num_blocks, dtype=torch.bool)
    if policy.keep_recent_blocks > 0:
        protected[max(0, num_blocks - policy.keep_recent_blocks):] = True

    drop_candidates = (~protected).nonzero(as_tuple=False).flatten()
    if policy.drop_fraction > 0 and drop_candidates.numel() > 0:
        num_drop = int(round(drop_candidates.numel() * policy.drop_fraction))
        if num_drop > 0:
            scores = block_scores[drop_candidates]
            drop_idx = torch.topk(scores, k=num_drop, largest=False).indices
            keep_mask[drop_candidates[drop_idx]] = False

    keep_mask |= protected
    return keep_mask


def choose_hot_blocks(
    block_scores: torch.Tensor,
    policy: BlockPolicy,
) -> torch.Tensor:
    num_blocks = block_scores.numel()
    hot_mask = torch.zeros(num_blocks, dtype=torch.bool)
    if policy.hot_fp16_ratio <= 0 or num_blocks == 0:
        return hot_mask
    num_hot = max(1, int(round(num_blocks * policy.hot_fp16_ratio)))
    hot_idx = torch.topk(block_scores, k=num_hot, largest=True).indices
    hot_mask[hot_idx] = True
    return hot_mask


def kv_storage_bytes_for_policy(
    num_blocks: int,
    block_size: int,
    hidden_dim: int,
    policy: BlockPolicy,
    kept_mask: torch.Tensor,
    hot_mask: torch.Tensor,
) -> int:
    elems_per_block = block_size * hidden_dim
    fp16_block_bytes = elems_per_block * 2 * bytes_for_dtype("fp16")

    total = 0
    for b in range(num_blocks):
        if not bool(kept_mask[b]):
            continue
        if bool(hot_mask[b]) or policy.quant_bits is None:
            total += fp16_block_bytes
        elif policy.quant_bits in (8, 4):
            code_bytes = quantized_code_bytes(elems_per_block, policy.quant_bits)
            scale_bytes = 2  # one fp16 scale per tensor
            total += 2 * (code_bytes + scale_bytes)
        else:
            raise ValueError(f"Unsupported quant bits: {policy.quant_bits}")
    return total


def apply_proxy_policy(
    token_weights: torch.Tensor,
    token_values: torch.Tensor,
    block_size: int,
    policy: BlockPolicy,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    num_tokens, hidden_dim = token_values.shape
    block_scores = blockify_token_scores(token_weights, block_size)
    value_blocks = pad_to_blocks(token_values, block_size)
    num_blocks = value_blocks.shape[0]

    kept_mask = choose_kept_blocks(block_scores, policy)
    hot_mask = choose_hot_blocks(block_scores, policy) & kept_mask
    cold_kept_mask = kept_mask & ~hot_mask

    quant_start = time.perf_counter()
    value_out = torch.zeros_like(value_blocks)
    if hot_mask.any():
        value_out[hot_mask] = value_blocks[hot_mask]
    if cold_kept_mask.any():
        cold_values = value_blocks[cold_kept_mask]
        if policy.quant_bits is None:
            value_out[cold_kept_mask] = cold_values
        else:
            q, scale = quantize_blockwise(cold_values, policy.quant_bits)
            value_out[cold_kept_mask] = dequantize_blockwise(q, scale)
    quant_ms = (time.perf_counter() - quant_start) * 1000.0

    flat_values = flatten_blocks(value_out, num_tokens)
    masked_weights = token_weights.clone()
    for b, (start, end) in enumerate(build_block_ranges(num_tokens, block_size)):
        if not bool(kept_mask[b]):
            masked_weights[start:end] = 0

    retained_mass = float(masked_weights.sum().item())
    renorm_weights = masked_weights / masked_weights.sum().clamp_min(1e-8)
    baseline_context = token_weights @ token_values
    test_context = renorm_weights @ flat_values
    block_test_scores = blockify_token_scores(renorm_weights, block_size)

    metrics = {
        "retained_attention_mass": retained_mass,
        "dropped_block_fraction": 1.0 - float(kept_mask.float().mean().item()),
        "storage_bytes": float(kv_storage_bytes_for_policy(
            num_blocks=num_blocks,
            block_size=block_size,
            hidden_dim=hidden_dim,
            policy=policy,
            kept_mask=kept_mask,
            hot_mask=hot_mask,
        )),
        "quantize_ms": quant_ms,
        "output_rel_l2": float((test_context - baseline_context).norm()
                                / baseline_context.norm().clamp_min(1e-8)),
        "output_cosine": cosine_scalar(baseline_context, test_context),
        "block_attn_l1": mean_abs_diff(block_scores, block_test_scores),
        "block_attn_kl": kl_divergence(block_scores / block_scores.sum().clamp_min(1e-8),
                                        block_test_scores / block_test_scores.sum().clamp_min(1e-8)),
        "top1_block_match": float(block_scores.argmax().item() == block_test_scores.argmax().item())
        if block_scores.numel() > 0 else 1.0,
        "hot_blocks": float(hot_mask.sum().item()),
    }
    return baseline_context, test_context, metrics


def build_synthetic_records(
    seeds: list[int],
    block_size: int,
    num_blocks: int,
    hidden_dim: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for seed in seeds:
        g = torch.Generator(device="cpu").manual_seed(seed)
        num_tokens = num_blocks * block_size
        token_values = torch.randn(num_tokens, hidden_dim, generator=g, dtype=torch.float32)
        ranks = torch.arange(1, num_blocks + 1, dtype=torch.float32)
        probs = 1.0 / torch.pow(ranks, 1.15)
        probs = probs / probs.sum()
        block_scores = torch.multinomial(probs, num_tokens, replacement=True, generator=g)
        token_weights = torch.zeros(num_tokens, dtype=torch.float32)
        for idx in block_scores.tolist():
            start = idx * block_size
            pos = start + int(torch.randint(0, block_size, (1,), generator=g).item())
            token_weights[pos] += 1.0
        token_weights = token_weights / token_weights.sum().clamp_min(1e-8)
        records.append({
            "sample_id": f"synthetic_seed_{seed}",
            "question": f"synthetic_seed_{seed}",
            "token_weights": token_weights,
            "token_values": token_values,
            "num_tokens": num_tokens,
        })
    return records


def build_prompt_text(passages: list[str], question: str) -> tuple[str, int, int]:
    parts = [p for p in passages if p]
    prompt = "\n".join(parts + ([question] if question else []))
    if question:
        question_start = len("\n".join(parts)) + (1 if parts else 0)
        return prompt, question_start, question_start + len(question)
    return prompt, 0, 0


def token_positions_from_offsets(offsets: list[list[int]], start_char: int, end_char: int) -> list[int]:
    positions = []
    for i, (s, e) in enumerate(offsets):
        if e <= start_char or s >= end_char:
            continue
        if e > s:
            positions.append(i)
    return positions


def load_retrieval_json(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("retrieval JSON must contain a list")
    return data


def build_dense_linearrag_retrieval(
    dataset_name: str,
    dataset_root: Path,
    linearrag_import_dir: Path,
    embedding_model_name: str,
    retrieval_top_k: int,
    max_samples: int | None,
    device: str,
) -> list[dict[str, Any]]:
    sys.path.insert(0, str((Path(__file__).resolve().parents[1]).resolve()))
    # Import the LinearRAG helper first so its huggingface_hub compatibility
    # shim is applied before sentence-transformers is loaded.
    from linearrag_gnn_infer import dense_retrieve_from_import
    from sentence_transformers import SentenceTransformer

    questions_path = dataset_root / dataset_name / "questions.json"
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    if max_samples is not None:
        questions = questions[:max_samples]
    questions = [{"question": x["question"], "answer": x.get("answer", "")} for x in questions]

    emb_model = SentenceTransformer(embedding_model_name, device=device)
    return dense_retrieve_from_import(
        import_root=linearrag_import_dir,
        dataset_name=dataset_name,
        questions=questions,
        embedding_model=emb_model,
        retrieval_top_k=retrieval_top_k,
    )


def build_real_records(
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
    records: list[dict[str, Any]] = []
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
                output_hidden_states=True,
                use_cache=False,
            )
        attn = outputs.attentions[layer_index][0].float()  # [heads, seq, seq]
        hidden = outputs.hidden_states[-1][0].float()      # [seq, hidden]

        seq_len = hidden.shape[0]
        prompt_positions = list(range(seq_len))
        attn_slice = attn[:, question_positions][:, :, prompt_positions]
        token_weights = attn_slice.mean(dim=(0, 1))
        token_weights = token_weights / token_weights.sum().clamp_min(1e-8)

        records.append({
            "sample_id": f"linearrag_{idx}",
            "question": question,
            "prompt": prompt,
            "num_tokens": seq_len,
            "token_weights": token_weights.cpu(),
            "token_values": hidden.cpu(),
            "block_scores": blockify_token_scores(token_weights.cpu(), block_size),
        })
    return records


def default_policies() -> list[BlockPolicy]:
    return [
        BlockPolicy("keep_fp16"),
        BlockPolicy("uniform_int8", quant_bits=8),
        BlockPolicy("uniform_int4", quant_bits=4),
        BlockPolicy("drop_bottom25", drop_fraction=0.25),
        BlockPolicy("drop_bottom50", drop_fraction=0.50),
        BlockPolicy("recent4_drop_bottom25", drop_fraction=0.25, keep_recent_blocks=4),
        BlockPolicy("recent4_drop_bottom50", drop_fraction=0.50, keep_recent_blocks=4),
        BlockPolicy("drop25_then_int8", quant_bits=8, drop_fraction=0.25),
        BlockPolicy("recent4_drop25_then_int8", quant_bits=8, drop_fraction=0.25, keep_recent_blocks=4),
        BlockPolicy("hot20_fp16_else_int8", quant_bits=8, hot_fp16_ratio=0.20),
    ]


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["policy"]), []).append(row)

    summary = []
    for policy, items in grouped.items():
        summary.append({
            "policy": policy,
            "compression_ratio_mean": sum(float(x["compression_ratio"]) for x in items) / len(items),
            "retained_attention_mass_mean": sum(float(x["retained_attention_mass"]) for x in items) / len(items),
            "dropped_block_fraction_mean": sum(float(x["dropped_block_fraction"]) for x in items) / len(items),
            "output_rel_l2_mean": sum(float(x["output_rel_l2"]) for x in items) / len(items),
            "output_cosine_mean": sum(float(x["output_cosine"]) for x in items) / len(items),
            "block_attn_l1_mean": sum(float(x["block_attn_l1"]) for x in items) / len(items),
            "block_attn_kl_mean": sum(float(x["block_attn_kl"]) for x in items) / len(items),
            "top1_block_match_mean": sum(float(x["top1_block_match"]) for x in items) / len(items),
            "quantize_ms_mean": sum(float(x["quantize_ms"]) for x in items) / len(items),
        })
    return sorted(
        summary,
        key=lambda x: (float(x["output_rel_l2_mean"]), -float(x["compression_ratio_mean"])),
    )


def print_table(summary_rows: list[dict[str, Any]]) -> None:
    headers = [
        "policy",
        "compression_ratio_mean",
        "retained_attention_mass_mean",
        "dropped_block_fraction_mean",
        "output_rel_l2_mean",
        "output_cosine_mean",
        "block_attn_l1_mean",
        "top1_block_match_mean",
        "quantize_ms_mean",
    ]
    print("\t".join(headers))
    for row in summary_rows:
        print("\t".join([
            str(row["policy"]),
            f"{float(row['compression_ratio_mean']):.3f}",
            f"{float(row['retained_attention_mass_mean']):.5f}",
            f"{float(row['dropped_block_fraction_mean']):.5f}",
            f"{float(row['output_rel_l2_mean']):.5f}",
            f"{float(row['output_cosine_mean']):.5f}",
            f"{float(row['block_attn_l1_mean']):.8f}",
            f"{float(row['top1_block_match_mean']):.5f}",
            f"{float(row['quantize_ms_mean']):.3f}",
        ]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--workload-mode", choices=["synthetic", "linearrag"], default="linearrag")
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--output-dir", type=Path,
                   default=Path("Hierarchical_KV/outputs/kv_cache_compression_study"))

    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--synthetic-num-blocks", type=int, default=256)
    p.add_argument("--synthetic-hidden-dim", type=int, default=4096)

    p.add_argument("--retrieval-json", type=Path, default=None,
                   help="JSON list with at least question + sorted_passage")
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
    args.output_dir = normalize_project_path(args.output_dir)
    args.dataset_root = normalize_project_path(args.dataset_root)
    args.linearrag_import_dir = normalize_project_path(args.linearrag_import_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.workload_mode == "synthetic":
        records = build_synthetic_records(
            seeds=args.seeds,
            block_size=args.block_size,
            num_blocks=args.synthetic_num_blocks,
            hidden_dim=args.synthetic_hidden_dim,
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
        records = build_real_records(
            model_name=args.model,
            retrieval_items=retrieval_items,
            block_size=args.block_size,
            max_seq_len=args.max_seq_len,
            max_samples=args.max_samples,
            layer_index=args.layer_index,
            device=args.device,
            dtype=args.dtype,
        )

    policies = default_policies()
    rows: list[dict[str, Any]] = []
    for rec in records:
        token_weights = rec["token_weights"].float()
        token_values = rec["token_values"].float()
        num_blocks = math.ceil(token_values.shape[0] / args.block_size)
        hidden_dim = token_values.shape[1]
        baseline_storage = num_blocks * args.block_size * hidden_dim * 2 * bytes_for_dtype("fp16")

        for policy in policies:
            _, _, metrics = apply_proxy_policy(
                token_weights=token_weights,
                token_values=token_values,
                block_size=args.block_size,
                policy=policy,
            )
            rows.append({
                "sample_id": rec["sample_id"],
                "question": rec["question"],
                "workload_mode": args.workload_mode,
                "block_size": args.block_size,
                "num_tokens": int(token_values.shape[0]),
                "policy": policy.name,
                "compression_ratio": float(baseline_storage / max(metrics["storage_bytes"], 1.0)),
                **metrics,
            })

    summary = summarize(rows)
    out_path = args.output_dir / "results.json"
    serializable_args = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()
    }
    out_path.write_text(json.dumps({
        "args": serializable_args,
        "num_records": len(records),
        "rows": rows,
        "summary": summary,
    }, indent=2), encoding="utf-8")
    print_table(summary)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
