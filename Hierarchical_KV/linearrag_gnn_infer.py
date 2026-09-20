#!/usr/bin/env python3
"""Run LinearRAG retrieval + GNN block importance inference in one pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import igraph as ig
import pandas as pd
import torch
from transformers import AutoTokenizer

try:
    import huggingface_hub as _hf_hub
    if not hasattr(_hf_hub, "HfFolder"):
        class _CompatHfFolder:
            @staticmethod
            def get_token():
                return None
        _hf_hub.HfFolder = _CompatHfFolder
    if not hasattr(_hf_hub, "Repository"):
        class _CompatRepository:
            def __init__(self, *args, **kwargs):
                raise NotImplementedError(
                    "huggingface_hub.Repository is unavailable in this environment"
                )
        _hf_hub.Repository = _CompatRepository
    if not hasattr(_hf_hub, "cached_download"):
        _hf_hub.cached_download = _hf_hub.hf_hub_download
except Exception:
    pass

from sentence_transformers import SentenceTransformer

from rag_query_gnn_predictor import GraphConditionedTokenRanker

def build_prompt_text(passages: List[str], question: str) -> str:
    parts = [p for p in passages if p]
    if question:
        parts.append(question)
    return "\n".join(parts)


def load_questions(args: argparse.Namespace) -> List[Dict[str, str]]:
    if args.question:
        return [{"question": args.question, "answer": ""}]
    with open(args.questions_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("questions-json must contain a list")
    return [{"question": x["question"], "answer": x.get("answer", "")} for x in data]


def load_passages(dataset_root: Path, dataset_name: str) -> List[str]:
    chunks_path = dataset_root / dataset_name / "chunks.json"
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return [f"{idx}:{chunk}" for idx, chunk in enumerate(chunks)]


def encode_prompt_text(text: str, tokenizer, max_seq_len: int) -> tuple[torch.Tensor, torch.Tensor, List[str]]:
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_seq_len,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
    )
    valid_len = int(enc["attention_mask"][0].sum().item())
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0, :valid_len].tolist())
    return enc["input_ids"][0].long(), enc["attention_mask"][0].bool(), tokens


def load_embeddings(parquet_path: Path) -> Dict[str, List[float]]:
    if not parquet_path.exists():
        return {}
    df = pd.read_parquet(parquet_path)
    return dict(zip(df["hash_id"].tolist(), df["embedding"].tolist()))


def load_passage_table(import_root: Path, dataset_name: str) -> pd.DataFrame:
    passage_parquet = import_root / dataset_name / "passage_embedding.parquet"
    if not passage_parquet.exists():
        raise FileNotFoundError(f"Missing passage embeddings: {passage_parquet}")
    return pd.read_parquet(passage_parquet)


def dense_retrieve_from_import(import_root: Path,
                               dataset_name: str,
                               questions: List[Dict[str, str]],
                               embedding_model,
                               retrieval_top_k: int) -> List[Dict[str, object]]:
    df = load_passage_table(import_root, dataset_name)
    passage_texts = df["text"].tolist()
    passage_embeddings = np.asarray(df["embedding"].tolist(), dtype=np.float32)

    retrieval_results: List[Dict[str, object]] = []
    for item in questions:
        question = item["question"]
        q_emb = np.asarray(
            embedding_model.encode(question,
                                   normalize_embeddings=True,
                                   show_progress_bar=False),
            dtype=np.float32,
        )
        scores = np.dot(passage_embeddings, q_emb)
        top_idx = np.argsort(scores)[::-1][:retrieval_top_k]
        retrieval_results.append({
            "question": question,
            "sorted_passage": [passage_texts[i] for i in top_idx],
            "sorted_passage_scores": [float(scores[i]) for i in top_idx],
            "gold_answer": item.get("answer", ""),
            "retrieval_mode": "dense_import_fallback",
        })
    return retrieval_results


def load_graph_from_linearrag_import(import_root: Path,
                                     dataset_name: str,
                                     device: str
                                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, int], Dict[str, str]]:
    ds_dir = import_root / dataset_name
    graph_path = ds_dir / "LinearRAG.graphml"
    passage_parquet = ds_dir / "passage_embedding.parquet"
    entity_parquet = ds_dir / "entity_embedding.parquet"
    if not graph_path.exists():
        raise FileNotFoundError(f"Missing graph: {graph_path}")

    g = ig.Graph.Read_GraphML(str(graph_path))
    node_names = g.vs["name"]
    node_name_to_idx = {str(name): i for i, name in enumerate(node_names)}
    passage_emb = load_embeddings(passage_parquet)
    entity_emb = load_embeddings(entity_parquet)
    passage_df = pd.read_parquet(passage_parquet) if passage_parquet.exists() else pd.DataFrame()
    passage_text_to_hash = {}
    if not passage_df.empty:
        passage_text_to_hash = {
            str(text): str(hash_id)
            for text, hash_id in zip(passage_df["text"].tolist(), passage_df["hash_id"].tolist())
        }
    any_emb = next(iter(passage_emb.values()), None)
    if any_emb is None:
        any_emb = next(iter(entity_emb.values()), None)
    if any_emb is None:
        raise ValueError(f"No embeddings found in {ds_dir}")

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

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(ew, dtype=torch.float32)
    return (
        node_features.to(device),
        edge_index.to(device),
        edge_weight.to(device),
        node_name_to_idx,
        passage_text_to_hash,
    )


def derive_retrieved_nodes(sorted_passage: List[str],
                           passage_text_to_hash: Dict[str, str],
                           node_name_to_idx: Dict[str, int],
                           max_retrieved: int) -> Tuple[List[str], List[int]]:
    passage_hash_ids: List[str] = []
    node_ids: List[int] = []
    for passage_text in sorted_passage[:max_retrieved]:
        passage_hash_id = passage_text_to_hash.get(passage_text)
        if passage_hash_id is None:
            continue
        passage_hash_ids.append(passage_hash_id)
        node_ids.append(node_name_to_idx.get(passage_hash_id, -1))
    if len(node_ids) < max_retrieved:
        node_ids.extend([-1] * (max_retrieved - len(node_ids)))
    return passage_hash_ids, node_ids


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-name", required=True)
    p.add_argument("--dataset-root", default="/mnt/shared/gpfs/home/seliny2/vllm/Hierarchical_KV/LinearRAG/dataset")
    p.add_argument("--linearrag-dir", default="/mnt/shared/gpfs/home/seliny2/vllm/Hierarchical_KV/LinearRAG")
    p.add_argument("--linearrag-import-dir", default="/mnt/shared/gpfs/home/seliny2/vllm/Hierarchical_KV/LinearRAG/import")
    p.add_argument("--embedding-model",
                   default="sentence-transformers/all-mpnet-base-v2")
    p.add_argument("--spacy-model", default="en_core_web_trf")
    p.add_argument("--retrieval-top-k", type=int, default=5)
    p.add_argument("--gnn-ckpt", required=True)
    p.add_argument("--gnn-data",
                   required=True,
                   help=("Training .pt used for tokenizer/max_retrieved. "
                         "Graph tensors are loaded from LinearRAG/import."))
    p.add_argument("--max-seq-len", type=int, default=8192)
    p.add_argument("--top-k-tokens", type=int, default=20)
    p.add_argument("--question", type=str, default=None)
    p.add_argument("--questions-json", type=str, default=None)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--retrieval-mode",
                   choices=["auto", "linearrag", "dense"],
                   default="auto",
                   help="Retrieval path before GNN token scoring")
    p.add_argument("--output", default="linearrag_gnn_predictions.json")
    args = p.parse_args()

    if not args.question and not args.questions_json:
        raise ValueError("provide either --question or --questions-json")

    linearrag_dir = Path(args.linearrag_dir).resolve()
    sys.path.insert(0, str(linearrag_dir))
    from src.LinearRAG import LinearRAG
    from src.config import LinearRAGConfig

    questions = load_questions(args)
    passages = load_passages(Path(args.dataset_root), args.dataset_name)

    emb_model = SentenceTransformer(args.embedding_model, device=args.device)
    retrieval = None
    if args.retrieval_mode != "dense":
        try:
            cfg = LinearRAGConfig(
                dataset_name=args.dataset_name,
                embedding_model=emb_model,
                spacy_model=args.spacy_model,
                llm_model=None,
                retrieval_top_k=args.retrieval_top_k,
            )
            rag = LinearRAG(cfg)
            rag.index(passages)
            retrieval = rag.retrieve(questions)
        except Exception:
            if args.retrieval_mode == "linearrag":
                raise
    if retrieval is None:
        retrieval = dense_retrieve_from_import(
            import_root=Path(args.linearrag_import_dir),
            dataset_name=args.dataset_name,
            questions=questions,
            embedding_model=emb_model,
            retrieval_top_k=args.retrieval_top_k,
        )

    gnn_data = torch.load(args.gnn_data, map_location="cpu")
    tokenizer_name_or_path = "meta-llama/Meta-Llama-3-8B"
    if not tokenizer_name_or_path:
        raise ValueError("gnn-data must contain tokenizer_name_or_path")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        pad_token_id = gnn_data.get("pad_token_id")
        if pad_token_id is not None:
            tokenizer.pad_token_id = int(pad_token_id)
        elif tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError("Tokenizer must define a pad_token_id or eos_token_id")
    train_seq_len = int(gnn_data["token_ids"].size(1))
    if args.max_seq_len > train_seq_len:
        print(
            f"[warn] requested max_seq_len={args.max_seq_len} exceeds training "
            f"sequence length {train_seq_len}; capping to {train_seq_len}"
        )
        args.max_seq_len = train_seq_len

    node_features, edge_index, edge_weight, node_name_to_idx, passage_text_to_hash = load_graph_from_linearrag_import(
        Path(args.linearrag_import_dir), args.dataset_name, args.device)
    max_retrieved = int(gnn_data["retrieved_node_ids"].size(1))

    ckpt = torch.load(args.gnn_ckpt, map_location="cpu")
    cfg = ckpt["config"]
    block_size = int(cfg.get("vllm_block_size", gnn_data.get("vllm_block_size", 16)))
    model = GraphConditionedTokenRanker(
        vocab_size=int(cfg["vocab_size"]),
        node_dim=node_features.size(1),
        hidden_dim=int(cfg["hidden_dim"]),
        gnn_layers=int(cfg["gnn_layers"]),
        tfm_layers=int(cfg["tfm_layers"]),
        num_heads=int(cfg["num_heads"]),
        dropout=float(cfg["dropout"]),
        importance_levels=int(cfg.get("importance_levels", 2)),
        block_size=block_size,
        prediction_unit=str(cfg.get("prediction_unit", "block")),
    ).to(args.device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    out = []
    for item in retrieval:
        prompt_text = build_prompt_text(item.get("sorted_passage", []),
                                        item["question"])
        token_ids, attn, prompt_tokens = encode_prompt_text(prompt_text, tokenizer, args.max_seq_len)
        final_passage_hash_ids, r_nodes = derive_retrieved_nodes(
            item.get("sorted_passage", []),
            passage_text_to_hash,
            node_name_to_idx,
            max_retrieved,
        )
        retrieved = torch.tensor(r_nodes, dtype=torch.long, device=args.device)

        block_payload = []
        importance_levels = int(cfg.get("importance_levels", 2))
        if importance_levels > 2:
            logits = model(
                token_ids.to(args.device).unsqueeze(0),
                attn.to(args.device).unsqueeze(0),
                node_features,
                edge_index,
                edge_weight,
                retrieved.unsqueeze(0),
            ).squeeze(0)
            probs = torch.softmax(logits, dim=-1)
            conf, levels = probs.max(dim=-1)
            num_valid_tokens = int(attn.sum().item())
            num_blocks = len(levels)
            for b in range(num_blocks):
                start = b * block_size
                end = min(start + block_size, num_valid_tokens)
                if start >= num_valid_tokens:
                    break
                block_payload.append({
                    "block_index": int(b),
                    "token_start": int(start),
                    "token_end": int(end),
                    "tokens": prompt_tokens[start:end],
                    "importance_level": int(levels[b]),
                    "confidence": float(conf[b]),
                    "class_probabilities": [
                        float(x) for x in probs[b].detach().cpu().tolist()
                    ],
                    "kv_cache_tier": int(levels[b]),
                })
            tier_names = (["low", "medium", "high"]
                          if importance_levels == 3 else
                          ["drop", "disk", "cpu", "gpu"])
            tier_counts = {
                tier_names[i] if i < len(tier_names) else f"level_{i}":
                int((levels == i).sum().item())
                for i in range(importance_levels)
            }
        else:
            logits = model(
                token_ids.to(args.device).unsqueeze(0),
                attn.to(args.device).unsqueeze(0),
                node_features,
                edge_index,
                edge_weight,
                retrieved.unsqueeze(0),
            ).squeeze(0)
            valid_scores = torch.sigmoid(logits).tolist()
            num_valid_tokens = int(attn.sum().item())
            for b, s in enumerate(valid_scores):
                start = b * block_size
                end = min(start + block_size, num_valid_tokens)
                if start >= num_valid_tokens:
                    break
                block_payload.append({
                    "block_index": int(b),
                    "token_start": int(start),
                    "token_end": int(end),
                    "tokens": prompt_tokens[start:end],
                    "score": float(s),
                })
            tier_counts = None
        out.append({
            "question": item["question"],
            "sorted_passage": item["sorted_passage"],
            "prompt_text": prompt_text,
            "final_passage_hash_ids": final_passage_hash_ids,
            "final_passage_node_ids": r_nodes,
            "vllm_block_size": block_size,
            "block_importance": block_payload,
            "kv_cache_summary": tier_counts,
        })

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"saved -> {out_path}")

def predict_prompt_blocks(
    prompt_text: str,
    sorted_passage: list[str],
    model,
    tokenizer,
    node_features,
    edge_index,
    edge_weight,
    passage_text_to_hash,
    node_name_to_idx,
    max_retrieved: int,
    max_seq_len: int,
    block_size: int,
    device: str,
):
    token_ids, attn, prompt_tokens = encode_prompt_text(
        prompt_text, tokenizer, max_seq_len
    )

    _, r_nodes = derive_retrieved_nodes(
        sorted_passage,
        passage_text_to_hash,
        node_name_to_idx,
        max_retrieved,
    )

    retrieved = torch.tensor(r_nodes, dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(
            token_ids.to(device).unsqueeze(0),
            attn.to(device).unsqueeze(0),
            node_features,
            edge_index,
            edge_weight,
            retrieved.unsqueeze(0),
        ).squeeze(0)

    return logits

if __name__ == "__main__":
    main()