#!/usr/bin/env python3
"""Build a unified multi-graph training file from multiple LinearRAG datasets."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import igraph as ig
import pandas as pd
import torch
from transformers import AutoTokenizer
from llm_attention_supervision import (load_attention_supervision,
                                       maybe_find_attention_file)


def build_prompt_text(passages: Sequence[str], question: str) -> str:
    parts = [p for p in passages if p]
    if question:
        parts.append(question)
    return "\n".join(parts)


def encode_prompt_text(text: str, tokenizer, max_seq_len: int) -> Tuple[List[int], List[int], List[str]]:
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_seq_len,
        padding="max_length",
        return_attention_mask=True,
    )
    input_ids = enc["input_ids"]
    attention = enc["attention_mask"]
    valid_len = int(sum(attention))
    tokens = tokenizer.convert_ids_to_tokens(input_ids[:valid_len])
    return input_ids, attention, tokens


def build_block_labels(token_labels: List[float], attention: List[int],
                       max_seq_len: int, block_size: int) -> Tuple[List[float], List[int]]:
    max_blocks = (max_seq_len + block_size - 1) // block_size
    block_labels: List[float] = []
    block_mask: List[int] = []
    for b in range(max_blocks):
        start = b * block_size
        end = min(start + block_size, max_seq_len)
        valid_idx = [i for i in range(start, end) if attention[i] == 1]
        if not valid_idx:
            block_labels.append(0.0)
            block_mask.append(0)
            continue
        scores = sorted((token_labels[i] for i in valid_idx), reverse=True)
        keep_n = max(1, (len(scores) + 9) // 10)
        block_labels.append(sum(scores[:keep_n]) / keep_n)
        block_mask.append(1)
    return block_labels, block_mask


def load_embeddings(parquet_path: Path) -> Dict[str, List[float]]:
    if not parquet_path.exists():
        return {}
    df = pd.read_parquet(parquet_path)
    return dict(zip(df["hash_id"].tolist(), df["embedding"].tolist()))


def latest_predictions(results_root: Path, dataset: str) -> Path:
    cands = sorted((results_root / dataset).glob("*/predictions.json"),
                   key=lambda p: p.stat().st_mtime,
                   reverse=True)
    if not cands:
        raise FileNotFoundError(
            f"No predictions found under {results_root / dataset}/*/predictions.json")
    return cands[0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets",
                   default="hotpotqa,2wikimultihop,musique,medical",
                   help="Comma-separated dataset names")
    p.add_argument("--results-root",
                   default="LinearRAG/results",
                   help="LinearRAG results root")
    p.add_argument("--linearrag-import-dir",
                   default="LinearRAG/import",
                   help="LinearRAG import root")
    p.add_argument("--tokenizer",
                   required=True,
                   help="HF tokenizer name/path used for prompt token IDs")
    p.add_argument("--max-seq-len", type=int, default=64)
    p.add_argument("--block-size", type=int, default=16,
                   help="Logical vLLM block size in tokens")
    p.add_argument("--max-retrieved", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--attention-filename",
                   default="prompt_attention_scores.pt",
                   help="Attention supervision file stored next to predictions.json")
    p.add_argument("--require-attention-supervision",
                   action="store_true",
                   help="Fail unless a last-layer attention supervision file is found")
    p.add_argument("--skip-missing",
                   action="store_true",
                   help="Skip datasets without required artifacts instead of failing")
    p.add_argument("--output",
                   default="data/all_multigraph_gnn_train.pt",
                   help="Output .pt path")
    args = p.parse_args()

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    if not datasets:
        raise ValueError("No datasets provided")

    import_root = Path(args.linearrag_import_dir)
    results_root = Path(args.results_root)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError("Tokenizer must define a pad_token_id or eos_token_id")

    graph_names: List[str] = []
    graph_node_features: List[torch.Tensor] = []
    graph_edge_index: List[torch.Tensor] = []
    graph_edge_weight: List[torch.Tensor] = []
    graph_node_name_to_idx: List[Dict[str, int]] = []
    graph_passage_text_to_hash: List[Dict[str, str]] = []

    all_predictions: List[Tuple[int, Dict]] = []
    all_prompt_texts: List[str] = []
    all_label_scores: List[List[float]] = []

    for ds in datasets:
        ds_dir = import_root / ds
        graph_path = ds_dir / "LinearRAG.graphml"
        passage_parquet = ds_dir / "passage_embedding.parquet"
        entity_parquet = ds_dir / "entity_embedding.parquet"
        try:
            pred_path = latest_predictions(results_root, ds)
            if not graph_path.exists():
                raise FileNotFoundError(f"Missing graph: {graph_path}")
        except FileNotFoundError as exc:
            if args.skip_missing:
                print(f"[skip] {ds}: {exc}")
                continue
            raise

        with open(pred_path, "r", encoding="utf-8") as f:
            preds = json.load(f)
        if not isinstance(preds, list) or not preds:
            raise ValueError(f"Invalid predictions file: {pred_path}")
        attn_path = maybe_find_attention_file(pred_path.parent, args.attention_filename)
        if attn_path is None and args.require_attention_supervision:
            raise FileNotFoundError(
                f"Missing attention supervision in {pred_path.parent}")
        attn_records = load_attention_supervision(attn_path) if attn_path else None
        if attn_records is not None and len(attn_records) != len(preds):
            raise ValueError(
                f"Attention supervision length mismatch for {ds}: "
                f"{len(attn_records)} vs predictions {len(preds)}")

        gid = len(graph_names)

        g = ig.Graph.Read_GraphML(str(graph_path))
        node_names = g.vs["name"]
        node_name_to_idx = {n: i for i, n in enumerate(node_names)}

        passage_emb = load_embeddings(passage_parquet)
        entity_emb = load_embeddings(entity_parquet)
        any_emb = next(iter(passage_emb.values()), None)
        if any_emb is None:
            any_emb = next(iter(entity_emb.values()), None)
        if any_emb is None:
            raise ValueError(f"No embeddings found for dataset {ds}")
        node_dim = len(any_emb)

        node_features = torch.zeros((len(node_names), node_dim), dtype=torch.float32)
        for i, node_name in enumerate(node_names):
            emb = passage_emb.get(node_name)
            if emb is None:
                emb = entity_emb.get(node_name)
            if emb is not None:
                node_features[i] = torch.tensor(emb, dtype=torch.float32)

        edges = g.get_edgelist()
        weights = g.es["weight"] if "weight" in g.es.attributes() else [1.0] * len(edges)
        src: List[int] = []
        dst: List[int] = []
        ew: List[float] = []
        for (u, v), w in zip(edges, weights):
            src.extend([u, v])
            dst.extend([v, u])
            ew.extend([float(w), float(w)])

        graph_names.append(ds)
        graph_node_features.append(node_features)
        graph_edge_index.append(torch.tensor([src, dst], dtype=torch.long))
        graph_edge_weight.append(torch.tensor(ew, dtype=torch.float32))
        graph_node_name_to_idx.append(node_name_to_idx)

        passage_text_to_hash: Dict[str, str] = {}
        if passage_parquet.exists():
            dfp = pd.read_parquet(passage_parquet)
            passage_text_to_hash = dict(
                zip(dfp["text"].tolist(), dfp["hash_id"].tolist()))
        graph_passage_text_to_hash.append(passage_text_to_hash)

        for idx, item in enumerate(preds):
            all_predictions.append((gid, item))
            if attn_records is not None:
                rec = attn_records[idx]
                prompt_text = build_prompt_text(item.get("sorted_passage", []),
                                                item.get("question", ""))
                prompt_tokens = rec.get("prompt_tokens")
                if prompt_tokens is None:
                    _, _, prompt_tokens = encode_prompt_text(prompt_text, tokenizer, args.max_seq_len)
                label_scores = rec["prompt_token_scores"]
            else:
                prompt_text = build_prompt_text(item.get("sorted_passage", []),
                                                item.get("question", ""))
                _, _, prompt_tokens = encode_prompt_text(prompt_text, tokenizer, args.max_seq_len)
                passage_blob = " ".join(item.get("sorted_passage", [])).lower()
                question_blob = item.get("question", "").lower()
                label_scores = [
                    1.0 if tok in f"{passage_blob} {question_blob}" else 0.0
                    for tok in prompt_tokens
                ]
            all_prompt_texts.append(prompt_text)
            all_label_scores.append(label_scores)

    if not graph_names:
        raise ValueError(
            "No usable datasets found. Run LinearRAG first or remove missing datasets.")

    token_id_rows: List[List[int]] = []
    attention_rows: List[List[int]] = []
    label_rows: List[List[float]] = []
    block_label_rows: List[List[float]] = []
    block_mask_rows: List[List[int]] = []
    retrieved_rows: List[List[int]] = []
    sample_graph_id: List[int] = []

    for (gid, item), prompt_text, raw_scores in zip(all_predictions,
                                               all_prompt_texts,
                                               all_label_scores):
        token_ids, attention, prompt_toks = encode_prompt_text(prompt_text, tokenizer, args.max_seq_len)
        if len(raw_scores) < len(prompt_toks):
            raise ValueError(
                f"Prompt token length mismatch for dataset sample graph={gid}: "
                f"labels={len(raw_scores)} tokenizer={len(prompt_toks)}"
            )
        raw_scores = raw_scores[:len(prompt_toks)]
        token_labels = [
            float(raw_scores[i]) if attention[i] == 1 and i < len(raw_scores) else 0.0
            for i in range(min(len(prompt_toks), args.max_seq_len))
        ]
        if len(token_labels) < args.max_seq_len:
            token_labels.extend([0.0] * (args.max_seq_len - len(token_labels)))
        block_labels, block_mask = build_block_labels(token_labels, attention,
                                                      args.max_seq_len,
                                                      args.block_size)

        node_name_to_idx = graph_node_name_to_idx[gid]
        passage_text_to_hash = graph_passage_text_to_hash[gid]

        retrieved: List[int] = []
        if "final_passage_node_ids" in item:
            for nid in item["final_passage_node_ids"][:args.max_retrieved]:
                nid = int(nid)
                if nid >= 0:
                    retrieved.append(nid)
        elif "final_passage_hash_ids" in item:
            for hid in item["final_passage_hash_ids"][:args.max_retrieved]:
                vidx = node_name_to_idx.get(hid)
                if vidx is not None:
                    retrieved.append(vidx)
        else:
            for ptxt in item.get("sorted_passage", [])[:args.max_retrieved]:
                hid = passage_text_to_hash.get(ptxt)
                if hid is None:
                    continue
                vidx = node_name_to_idx.get(hid)
                if vidx is not None:
                    retrieved.append(vidx)
        if len(retrieved) < args.max_retrieved:
            retrieved.extend([-1] * (args.max_retrieved - len(retrieved)))

        token_id_rows.append(token_ids)
        attention_rows.append(attention)
        label_rows.append(token_labels)
        block_label_rows.append(block_labels)
        block_mask_rows.append(block_mask)
        retrieved_rows.append(retrieved)
        sample_graph_id.append(gid)

    n = len(token_id_rows)
    rng = random.Random(args.seed)
    perm = list(range(n))
    rng.shuffle(perm)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    out = {
        "token_ids": torch.tensor(token_id_rows, dtype=torch.long),
        "attention_mask": torch.tensor(attention_rows, dtype=torch.bool),
        "token_labels": torch.tensor(label_rows, dtype=torch.float32),
        "block_labels": torch.tensor(block_label_rows, dtype=torch.float32),
        "block_attention_mask": torch.tensor(block_mask_rows, dtype=torch.bool),
        "retrieved_node_ids": torch.tensor(retrieved_rows, dtype=torch.long),
        "sample_graph_id": torch.tensor(sample_graph_id, dtype=torch.long),
        "graph_names": graph_names,
        "graph_node_features": graph_node_features,
        "graph_edge_index": graph_edge_index,
        "graph_edge_weight": graph_edge_weight,
        "tokenizer_name_or_path": args.tokenizer,
        "tokenizer_vocab_size": int(len(tokenizer)),
        "pad_token_id": int(tokenizer.pad_token_id),
        "vllm_block_size": int(args.block_size),
        "label_source": ("last_layer_attention"
                          if args.require_attention_supervision else
                          "last_layer_attention_or_prompt_overlap_fallback"),
        "train_idx": torch.tensor(perm[:n_train], dtype=torch.long),
        "val_idx": torch.tensor(perm[n_train:n_train + n_val], dtype=torch.long),
        "test_idx": torch.tensor(perm[n_train + n_val:], dtype=torch.long),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"saved -> {out_path}")
    print(f"samples={n} graphs={len(graph_names)} tokenizer_vocab={int(len(tokenizer))}")
    for i, name in enumerate(graph_names):
        print(
            f"  graph[{i}]={name}: nodes={graph_node_features[i].size(0)} "
            f"edges={graph_edge_index[i].size(1)}"
        )


if __name__ == "__main__":
    main()
