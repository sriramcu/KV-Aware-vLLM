#!/usr/bin/env python3
"""Build training data for rag_query_gnn_predictor.py from LinearRAG artifacts.

Outputs a .pt file with keys:
- token_ids [N, L]
- attention_mask [N, L]
- token_labels [N, L]
- node_features [V, D]
- edge_index [2, E]
- edge_weight [E]
- retrieved_node_ids [N, K]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import igraph as ig
import pandas as pd
import torch
from transformers import AutoTokenizer
from llm_attention_supervision import load_attention_supervision


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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-name", required=True)
    p.add_argument("--predictions-json",
                   required=True,
                   help="LinearRAG results/*/predictions.json")
    p.add_argument("--linearrag-import-dir",
                   default="LinearRAG/import",
                   help="Directory containing <dataset>/LinearRAG.graphml")
    p.add_argument("--attention-supervision",
                   type=str,
                   default=None,
                   help="Optional attention supervision file for prompt tokens")
    p.add_argument("--tokenizer",
                   required=True,
                   help="HF tokenizer name/path used for prompt token IDs")
    p.add_argument("--max-seq-len", type=int, default=64)
    p.add_argument("--block-size", type=int, default=16,
                   help="Logical vLLM block size in tokens")
    p.add_argument("--max-retrieved", type=int, default=5)
    p.add_argument("--output",
                   default="gnn_train_data.pt",
                   help="Output .pt path")
    args = p.parse_args()

    ds_dir = Path(args.linearrag_import_dir) / args.dataset_name
    graph_path = ds_dir / "LinearRAG.graphml"
    passage_parquet = ds_dir / "passage_embedding.parquet"
    entity_parquet = ds_dir / "entity_embedding.parquet"

    if not graph_path.exists():
        raise FileNotFoundError(f"Missing graph: {graph_path}")
    if not Path(args.predictions_json).exists():
        raise FileNotFoundError(f"Missing predictions: {args.predictions_json}")

    with open(args.predictions_json, "r", encoding="utf-8") as f:
        predictions = json.load(f)
    if not isinstance(predictions, list) or not predictions:
        raise ValueError("predictions.json must be a non-empty list")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError("Tokenizer must define a pad_token_id or eos_token_id")

    g = ig.Graph.Read_GraphML(str(graph_path))
    node_names = g.vs["name"]
    node_name_to_idx = {n: i for i, n in enumerate(node_names)}

    passage_emb = load_embeddings(passage_parquet)
    entity_emb = load_embeddings(entity_parquet)
    any_emb = next(iter(passage_emb.values()), None)
    if any_emb is None:
        any_emb = next(iter(entity_emb.values()), None)
    if any_emb is None:
        raise ValueError("No embeddings found in passage/entity parquet files")
    node_dim = len(any_emb)

    node_features = torch.zeros((len(node_names), node_dim), dtype=torch.float32)
    for i, node_name in enumerate(node_names):
        emb = passage_emb.get(node_name)
        if emb is None:
            emb = entity_emb.get(node_name)
        if emb is not None:
            node_features[i] = torch.tensor(emb, dtype=torch.float32)

    passage_text_to_hash = {}
    if passage_parquet.exists():
        dfp = pd.read_parquet(passage_parquet)
        passage_text_to_hash = dict(zip(dfp["text"].tolist(), dfp["hash_id"].tolist()))

    prompt_texts = [
        build_prompt_text(x.get("sorted_passage", []), x.get("question", ""))
        for x in predictions
    ]
    attn_records = None
    if args.attention_supervision:
        attn_records = load_attention_supervision(args.attention_supervision)
        if len(attn_records) != len(predictions):
            raise ValueError("attention supervision length does not match predictions")

    token_id_rows: List[List[int]] = []
    attention_rows: List[List[int]] = []
    label_rows: List[List[float]] = []
    block_label_rows: List[List[float]] = []
    block_mask_rows: List[List[int]] = []
    retrieved_rows: List[List[int]] = []

    for i, (item, prompt_text) in enumerate(zip(predictions, prompt_texts)):
        token_ids, attention, prompt_toks = encode_prompt_text(prompt_text, tokenizer, args.max_seq_len)
        if attn_records is not None:
            raw_scores = attn_records[i]["prompt_token_scores"]
            if len(raw_scores) < len(prompt_toks):
                raise ValueError(
                    f"Prompt token length mismatch at sample {i}: "
                    f"supervision={len(raw_scores)} tokenizer={len(prompt_toks)}"
                )
            raw_scores = raw_scores[:len(prompt_toks)]
            token_labels = [
                float(raw_scores[j]) if attention[j] == 1 and j < len(raw_scores) else 0.0
                for j in range(min(len(prompt_toks), args.max_seq_len))
            ]
        else:
            passage_blob = " ".join(item.get("sorted_passage", [])).lower()
            question_blob = item.get("question", "").lower()
            token_labels = [
                1.0 if (attention[j] == 1 and prompt_toks[j] in f"{passage_blob} {question_blob}") else 0.0
                for j in range(min(len(prompt_toks), args.max_seq_len))
            ]
        if len(token_labels) < args.max_seq_len:
            token_labels.extend([0.0] * (args.max_seq_len - len(token_labels)))
        block_labels, block_mask = build_block_labels(token_labels, attention,
                                                      args.max_seq_len,
                                                      args.block_size)

        retrieved = []
        if "final_passage_node_ids" in item:
            for nid in item["final_passage_node_ids"][:args.max_retrieved]:
                nid = int(nid)
                if 0 <= nid < len(node_names):
                    retrieved.append(nid)
        elif "final_passage_hash_ids" in item:
            for hid in item["final_passage_hash_ids"][:args.max_retrieved]:
                vidx = node_name_to_idx.get(hid)
                if vidx is not None:
                    retrieved.append(vidx)
        else:
            # Backward compatibility: map from passage text.
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

    # Duplicate edges in both directions to support bidirectional message passing.
    edges = g.get_edgelist()
    weights = g.es["weight"] if "weight" in g.es.attributes() else [1.0] * len(edges)
    src = []
    dst = []
    ew = []
    for (u, v), w in zip(edges, weights):
        src.extend([u, v])
        dst.extend([v, u])
        ew.extend([float(w), float(w)])

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(ew, dtype=torch.float32)

    out = {
        "token_ids": torch.tensor(token_id_rows, dtype=torch.long),
        "attention_mask": torch.tensor(attention_rows, dtype=torch.bool),
        "token_labels": torch.tensor(label_rows, dtype=torch.float32),
        "block_labels": torch.tensor(block_label_rows, dtype=torch.float32),
        "block_attention_mask": torch.tensor(block_mask_rows, dtype=torch.bool),
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "retrieved_node_ids": torch.tensor(retrieved_rows, dtype=torch.long),
        "tokenizer_name_or_path": args.tokenizer,
        "tokenizer_vocab_size": int(len(tokenizer)),
        "pad_token_id": int(tokenizer.pad_token_id),
        "vllm_block_size": int(args.block_size),
        "label_source": ("last_layer_attention"
                          if attn_records is not None else
                          "prompt_overlap"),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, output)
    print(f"saved -> {output}")
    print(
        f"samples={out['token_ids'].size(0)} nodes={node_features.size(0)} "
        f"edges={edge_index.size(1)} tokenizer_vocab={int(len(tokenizer))}"
    )


if __name__ == "__main__":
    main()
