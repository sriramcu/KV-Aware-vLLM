#!/usr/bin/env python3
"""
infer_from_checkpoint.py
────────────────────────
Load a saved model checkpoint + TokenRankData checkpoint and run
3-level token importance prediction on every sample in a chosen split.

Usage
─────
# Run on test split (default)
python infer_from_checkpoint.py \
    --model-ckpt  token_ranker_model.pt \
    --data-ckpt   my_token_rank_data.pt \
    --split       test \
    --batch-size  32 \
    --output      results.json

# Run on all samples regardless of split
python infer_from_checkpoint.py \
    --model-ckpt token_ranker_model.pt \
    --data-ckpt  my_token_rank_data.pt \
    --split      all
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class SAGEConvWeighted(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin_self  = nn.Linear(in_dim, out_dim, bias=True)
        self.lin_neigh = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, edge_index, edge_weight):
        src, dst   = edge_index
        num_nodes  = x.size(0)
        msg        = x[src] * edge_weight.unsqueeze(-1)
        neigh_sum  = torch.zeros_like(x)
        neigh_sum.index_add_(0, dst, msg)
        deg        = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
        deg.index_add_(0, dst, edge_weight)
        deg        = deg.clamp_min(1e-12).unsqueeze(-1)
        return self.lin_self(x) + self.lin_neigh(neigh_sum / deg)


class GraphEncoder(nn.Module):
    def __init__(self, node_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.node_in = nn.Linear(node_dim, hidden_dim)
        self.layers  = nn.ModuleList(
            [SAGEConvWeighted(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.dropout = dropout

    def forward(self, node_features, edge_index, edge_weight):
        h = self.node_in(node_features)
        for layer in self.layers:
            h = layer(h, edge_index, edge_weight)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return F.normalize(h, p=2, dim=-1)


class GraphConditionedTokenRanker(nn.Module):
    def __init__(self, vocab_size, node_dim, hidden_dim, gnn_layers,
                 tfm_layers, num_heads, dropout, importance_levels=2):
        super().__init__()
        self.importance_levels = importance_levels
        self.graph_encoder  = GraphEncoder(node_dim, hidden_dim, gnn_layers, dropout)
        self.token_embed    = nn.Embedding(vocab_size, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=4 * hidden_dim, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.token_encoder  = nn.TransformerEncoder(enc_layer, num_layers=tfm_layers)
        self.graph_to_token = nn.Linear(hidden_dim, hidden_dim)
        out_dim = 1 if importance_levels == 2 else importance_levels
        self.score_head     = nn.Linear(hidden_dim, out_dim)

    @staticmethod
    def _pool_retrieved_nodes(node_repr, retrieved_node_ids):
        valid    = retrieved_node_ids >= 0
        safe_ids = retrieved_node_ids.clamp_min(0)
        gathered = node_repr[safe_ids] * valid.unsqueeze(-1)
        denom    = valid.sum(dim=1, keepdim=True).clamp_min(1).to(gathered.dtype)
        return gathered.sum(dim=1) / denom

    def forward(self, token_ids, attention_mask, node_features,
                edge_index, edge_weight, retrieved_node_ids):
        node_repr  = self.graph_encoder(node_features, edge_index, edge_weight)
        graph_ctx  = self._pool_retrieved_nodes(node_repr, retrieved_node_ids)
        tok        = self.token_embed(token_ids)
        tok        = tok + self.graph_to_token(graph_ctx).unsqueeze(1)
        tok        = self.token_encoder(tok, src_key_padding_mask=~attention_mask.bool())
        logits     = self.score_head(tok)
        if self.importance_levels == 2:
            logits = logits.squeeze(-1)
        return logits


@dataclass
class TokenRankData:
    token_ids:           torch.Tensor
    attention_mask:      torch.Tensor
    token_labels:        torch.Tensor
    node_features:       Optional[torch.Tensor]
    edge_index:          Optional[torch.Tensor]
    edge_weight:         Optional[torch.Tensor]
    retrieved_node_ids:  torch.Tensor
    train_idx:           torch.Tensor
    val_idx:             torch.Tensor
    test_idx:            torch.Tensor
    graph_node_features: Optional[List[torch.Tensor]] = None
    graph_edge_index:    Optional[List[torch.Tensor]] = None
    graph_edge_weight:   Optional[List[torch.Tensor]] = None
    sample_graph_id:     Optional[torch.Tensor]       = None
    graph_names:         Optional[List[str]]          = None

LEVEL_NAMES = {0: "LOW", 1: "MED", 2: "HIGH"}


def _is_multigraph(data: TokenRankData) -> bool:
    return (
        data.graph_node_features is not None
        and data.graph_edge_index  is not None
        and data.graph_edge_weight is not None
        and data.sample_graph_id   is not None
    )


def load_model_checkpoint(
    path: str, device: torch.device
) -> Tuple[GraphConditionedTokenRanker, dict]:
    ckpt = torch.load(path, map_location="cpu")
    if "config" not in ckpt or "state_dict" not in ckpt:
        raise ValueError(
            f"Model checkpoint '{path}' must contain 'config' and 'state_dict'. "
            "Ensure it was saved with --output-model from the training script."
        )
    cfg = ckpt["config"]
    model = GraphConditionedTokenRanker(
        vocab_size        = cfg["vocab_size"],
        node_dim          = cfg.get("node_dim", 64),
        hidden_dim        = cfg["hidden_dim"],
        gnn_layers        = cfg["gnn_layers"],
        tfm_layers        = cfg["tfm_layers"],
        num_heads         = cfg["num_heads"],
        dropout           = cfg["dropout"],
        importance_levels = cfg.get("importance_levels", 2),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    importance_levels = cfg.get("importance_levels", 2)
    print(f"[model]  loaded from '{path}'  "
          f"(importance_levels={importance_levels}, "
          f"hidden_dim={cfg['hidden_dim']}, "
          f"vocab_size={cfg['vocab_size']})")
    return model, cfg


def load_data_checkpoint(path: str) -> TokenRankData:
    """
    Loads a TokenRankData object saved with torch.save().
    Handles both:
      - torch.save(data, path)          → direct TokenRankData
      - torch.save(data.__dict__, path) → plain dict
    """
    raw = torch.load(path, map_location="cpu")

    if isinstance(raw, TokenRankData):
        data = raw
    elif isinstance(raw, dict):
        # Reconstruct from dict — handle both full-field and minimal saves
        required = {"token_ids", "attention_mask", "token_labels", "retrieved_node_ids"}
        missing  = required - raw.keys()
        if missing:
            raise ValueError(f"Data checkpoint missing required keys: {missing}")

        # Synthesise split indices if not present
        num_samples = raw["token_ids"].size(0)
        if all(k in raw for k in ("train_idx", "val_idx", "test_idx")):
            train_idx = raw["train_idx"]
            val_idx   = raw["val_idx"]
            test_idx  = raw["test_idx"]
        else:
            print("[data]   no split indices found — using all samples as test split")
            train_idx = torch.zeros(0, dtype=torch.long)
            val_idx   = torch.zeros(0, dtype=torch.long)
            test_idx  = torch.arange(num_samples, dtype=torch.long)

        data = TokenRankData(
            token_ids           = raw["token_ids"],
            attention_mask      = raw["attention_mask"],
            token_labels        = raw["token_labels"],
            node_features       = raw.get("node_features"),
            edge_index          = raw.get("edge_index"),
            edge_weight         = raw.get("edge_weight"),
            retrieved_node_ids  = raw["retrieved_node_ids"],
            train_idx           = train_idx,
            val_idx             = val_idx,
            test_idx            = test_idx,
            graph_node_features = raw.get("graph_node_features"),
            graph_edge_index    = raw.get("graph_edge_index"),
            graph_edge_weight   = raw.get("graph_edge_weight"),
            sample_graph_id     = raw.get("sample_graph_id"),
            graph_names         = raw.get("graph_names"),
        )
    else:
        raise TypeError(
            f"Unexpected type in data checkpoint: {type(raw)}. "
            "Expected TokenRankData or dict."
        )

    num_samples = data.token_ids.size(0)
    print(f"[data]   loaded from '{path}'  "
          f"({num_samples} samples, "
          f"seq_len={data.token_ids.size(1)}, "
          f"multigraph={_is_multigraph(data)})")
    return data


def get_split_indices(data: TokenRankData, split: str) -> torch.Tensor:
    if split == "train":
        idx = data.train_idx
    elif split == "val":
        idx = data.val_idx
    elif split == "test":
        idx = data.test_idx
    elif split == "all":
        idx = torch.arange(data.token_ids.size(0), dtype=torch.long)
    else:
        raise ValueError(f"split must be one of: train | val | test | all. Got '{split}'")

    if idx.numel() == 0:
        print(f"[warn]   split='{split}' has 0 samples — falling back to all samples")
        idx = torch.arange(data.token_ids.size(0), dtype=torch.long)
    return idx


def _forward_batch(
    model:  GraphConditionedTokenRanker,
    data:   TokenRankData,
    bidx:   torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Forward pass for a batch of sample indices. Handles single / multi-graph."""
    token_ids  = data.token_ids[bidx].to(device)
    attn       = data.attention_mask[bidx].to(device).bool()
    retrieved  = data.retrieved_node_ids[bidx].to(device)

    if not _is_multigraph(data):
        return model(
            token_ids, attn,
            data.node_features.to(device),
            data.edge_index.to(device),
            data.edge_weight.to(device),
            retrieved,
        )

    # Per-sample graph (multigraph mode)
    rows = []
    for local_i, sample_i in enumerate(bidx.tolist()):
        gid = int(data.sample_graph_id[sample_i].item())
        logits_i = model(
            token_ids[local_i:local_i + 1],
            attn[local_i:local_i + 1],
            data.graph_node_features[gid].to(device),
            data.graph_edge_index[gid].to(device),
            data.graph_edge_weight[gid].to(device),
            retrieved[local_i:local_i + 1],
        ).squeeze(0)
        rows.append(logits_i)
    return torch.stack(rows, dim=0)


@torch.no_grad()
def run_inference(
    model:      GraphConditionedTokenRanker,
    data:       TokenRankData,
    split_idx:  torch.Tensor,
    batch_size: int,
    device:     torch.device,
) -> List[dict]:
    """
    Returns a list of per-sample result dicts:
        {
          "sample_idx"  : int,
          "token_ids"   : List[int],
          "levels"      : List[int],       # 0=LOW  1=MED  2=HIGH
          "confidence"  : List[float],
          "labels"      : List[int] | None,  # ground-truth if available
          "token_acc"   : float | None,
        }
    """
    model.eval()
    results = []

    for start in tqdm(range(0, split_idx.numel(), batch_size), desc="Inferring"):
        bidx   = split_idx[start : start + batch_size]
        attn   = data.attention_mask[bidx].bool()          # CPU for label logic
        labels = data.token_labels[bidx]

        logits = _forward_batch(model, data, bidx, device) # [B, L, 3] or [B, L]

        #  3-level decoding 
        if logits.dim() == 3:                              # [B, L, 3]  ✓ 3-level
            probs  = torch.softmax(logits.cpu(), dim=-1)   # [B, L, 3]
            conf, pred_levels = probs.max(dim=-1)          # [B, L]
        elif logits.dim() == 2:                            # [B, L]  binary model
            print("[warn] model output is binary (2-level). "
                  "Thresholding to LOW / HIGH only.")
            probs_bin   = torch.sigmoid(logits.cpu())
            pred_levels = (probs_bin >= 0.5).long()
            conf        = torch.where(pred_levels == 1, probs_bin, 1.0 - probs_bin)
        else:
            raise RuntimeError(f"Unexpected logits shape: {logits.shape}")

        # Ground-truth levels (if token_labels are provided)
        has_labels = (labels.abs().sum() > 0)

        for i, sample_i in enumerate(bidx.tolist()):
            valid_mask = attn[i]                          # [L]  bool

            lvl_i  = pred_levels[i][valid_mask].tolist()
            conf_i = conf[i][valid_mask].tolist()
            tids_i = data.token_ids[sample_i][valid_mask].tolist()

            # Ground-truth
            gt_levels, acc = None, None
            if has_labels:
                raw_labels = labels[i]
                if torch.is_floating_point(raw_labels):
                    gt_levels = _float_labels_to_levels(raw_labels[valid_mask]).tolist()
                else:
                    gt_levels = raw_labels[valid_mask].long().clamp(0, 2).tolist()
                correct = sum(p == g for p, g in zip(lvl_i, gt_levels))
                acc     = correct / max(len(lvl_i), 1)

            results.append({
                "sample_idx": sample_i,
                "token_ids":  tids_i,
                "levels":     lvl_i,
                "confidence": [round(c, 4) for c in conf_i],
                "labels":     gt_levels,
                "token_acc":  round(acc, 4) if acc is not None else None,
            })

    return results


def _float_labels_to_levels(
    labels: torch.Tensor,
    low: float  = 0.33,
    high: float = 0.66,
) -> torch.Tensor:
    """Map float labels in [0,1] → 0/1/2 levels."""
    out = torch.zeros_like(labels, dtype=torch.long)
    out = torch.where(labels >= high, torch.full_like(out, 2), out)
    out = torch.where((labels >= low) & (labels < high), torch.full_like(out, 1), out)
    return out


def compute_summary(results: List[dict]) -> dict:
    total_tokens  = 0
    level_counts  = {0: 0, 1: 0, 2: 0}
    accs          = [r["token_acc"] for r in results if r["token_acc"] is not None]

    for r in results:
        for lvl in r["levels"]:
            level_counts[lvl] += 1
            total_tokens      += 1

    summary = {
        "num_samples"         : len(results),
        "total_valid_tokens"  : total_tokens,
        "level_distribution"  : {
            LEVEL_NAMES[k]: {
                "count": v,
                "pct":   round(100 * v / max(total_tokens, 1), 2),
            }
            for k, v in level_counts.items()
        },
    }
    if accs:
        summary["mean_token_acc"] = round(sum(accs) / len(accs), 4)
        summary["samples_with_labels"] = len(accs)

    return summary


def print_sample(result: dict, top_k: int = 20) -> None:
    """Pretty-print one sample's high-importance tokens."""
    print(f"\n── Sample {result['sample_idx']} ──")
    tids  = result["token_ids"]
    lvls  = result["levels"]
    confs = result["confidence"]
    gts   = result["labels"]

    # All tokens table
    print(f"  {'Pos':>4}  {'TokenID':>8}  {'Pred':>4}  {'Conf':>6}"
          + ("  GT" if gts else ""))
    for pos, (tid, lvl, conf) in enumerate(zip(tids, lvls, confs)):
        gt_str = f"  {LEVEL_NAMES[gts[pos]]}" if gts else ""
        marker = " ◀" if lvl == 2 else ""
        print(f"  {pos:>4}  {tid:>8}  {LEVEL_NAMES[lvl]:>4}  {conf:>6.3f}{gt_str}{marker}")

    # Top-k HIGH tokens
    high_items = [(pos, conf) for pos, (lvl, conf) in enumerate(zip(lvls, confs)) if lvl == 2]
    high_items.sort(key=lambda x: x[1], reverse=True)
    print(f"\n  Top-{top_k} HIGH tokens (by confidence):")
    for rank, (pos, conf) in enumerate(high_items[:top_k], 1):
        print(f"    {rank:>3}. pos={pos:<5} token_id={tids[pos]:<8} conf={conf:.4f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="3-level token importance inference from saved checkpoints"
    )
    p.add_argument("--model-ckpt",  required=True,
                   help="Path to model checkpoint (.pt) saved by training script")
    p.add_argument("--data-ckpt",   required=True,
                   help="Path to TokenRankData checkpoint (.pt / .pth)")
    p.add_argument("--split",       default="test",
                   choices=["train", "val", "test", "all"],
                   help="Which data split to run inference on (default: test)")
    p.add_argument("--batch-size",  type=int, default=32)
    p.add_argument("--device",      type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output",      type=str, default="inference_results.json",
                   help="Path to save per-sample JSON results")
    p.add_argument("--print-samples", type=int, default=3,
                   help="Number of sample breakdowns to print (0 = none)")
    p.add_argument("--low-threshold",  type=float, default=0.33,
                   help="Float label threshold for LOW  → MED boundary")
    p.add_argument("--high-threshold", type=float, default=0.66,
                   help="Float label threshold for MED  → HIGH boundary")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device(args.device)

    model, cfg = load_model_checkpoint(args.model_ckpt, device)
    data       = load_data_checkpoint(args.data_ckpt)
    split_idx  = get_split_indices(data, args.split)

    importance_levels = cfg.get("importance_levels", 2)
    if importance_levels != 3:
        print(f"[warn] checkpoint was trained with importance_levels={importance_levels}. "
              "Predictions will be binary (LOW/HIGH) — MED level will be absent.")

    print(f"\nRunning inference on split='{args.split}' "
          f"({split_idx.numel()} samples) …\n")

    results = run_inference(model, data, split_idx, args.batch_size, device)

    summary = compute_summary(results)
    print("\n── Summary ──────────────────────────────────────────────────────")
    print(json.dumps(summary, indent=2))

    for i in range(min(args.print_samples, len(results))):
        print_sample(results[i], top_k=20)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "results": results}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {len(results)} results → {out_path}")


if __name__ == "__main__":
    main()