"""Inference-only Short-Q ranker compatible with Yingbing/shortq-kv-ranker-8k."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class SAGEConvWeighted(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim, bias=True)
        self.lin_neigh = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        msg = x[src] * edge_weight.unsqueeze(-1)
        neigh_sum = torch.zeros_like(x)
        neigh_sum.index_add_(0, dst, msg)
        deg = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        deg.index_add_(0, dst, edge_weight)
        return self.lin_self(x) + self.lin_neigh(neigh_sum / deg.clamp_min(1e-12).unsqueeze(-1))


class GraphEncoder(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.node_in = nn.Linear(node_dim, hidden_dim)
        self.layers = nn.ModuleList([SAGEConvWeighted(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.dropout = dropout

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        h = self.node_in(node_features)
        for layer in self.layers:
            h = F.dropout(F.relu(layer(h, edge_index, edge_weight)), p=self.dropout, training=self.training)
        return F.normalize(h, p=2, dim=-1)


def gather_query_blocks(h: torch.Tensor, q_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    counts = q_mask.sum(dim=1)
    max_q = max(1, int(counts.max().item()))
    out = h.new_zeros((h.size(0), max_q, h.size(-1)))
    pad = torch.ones((h.size(0), max_q), dtype=torch.bool, device=h.device)
    for i in range(h.size(0)):
        vals = h[i][q_mask[i]]
        n = vals.size(0)
        if n:
            out[i, :n] = vals
            pad[i, :n] = False
    return out, pad


def shortq_block_scores(q: torch.Tensor, k: torch.Tensor, q_pad: torch.Tensor, k_mask: torch.Tensor) -> torch.Tensor:
    logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.size(-1))
    logits = logits.masked_fill(q_pad.unsqueeze(-1), torch.finfo(logits.dtype).min)
    logits = logits.masked_fill(~k_mask.unsqueeze(1), torch.finfo(logits.dtype).min)
    weights = torch.nan_to_num(torch.softmax(logits, dim=-1), nan=0.0)
    weights = weights.masked_fill(q_pad.unsqueeze(-1), 0)
    denom = (~q_pad).sum(dim=1).clamp_min(1).to(weights.dtype).unsqueeze(-1)
    return weights.sum(dim=1) / denom


class ShortQBlockKVRanker(nn.Module):
    def __init__(self, llm_feat_dim: int, node_dim: int, hidden_dim: int, gnn_layers: int,
                 num_heads: int, dropout: float, importance_levels: int = 4,
                 max_retrieved: int = 5, max_blocks: int = 512):
        super().__init__()
        if importance_levels != 4:
            raise ValueError("first placement integration expects a four-class checkpoint")
        self.importance_levels = importance_levels
        self.max_retrieved = max_retrieved
        self.graph_encoder = GraphEncoder(node_dim, hidden_dim, gnn_layers, dropout)
        self.block_proj = nn.Sequential(nn.LayerNorm(llm_feat_dim), nn.Linear(llm_feat_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.recency_emb = nn.Embedding(max_blocks, hidden_dim)
        self.passage_emb = nn.Embedding(max_retrieved + 1, hidden_dim)
        self.shortq_proj = nn.Linear(1, hidden_dim)
        self.block_to_graph = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.fusion_gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.block_mlp = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 4 * hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * hidden_dim, hidden_dim), nn.Dropout(dropout))
        self.class_head = nn.Linear(hidden_dim, importance_levels)
        self.rank_head = nn.Linear(hidden_dim, 1)

    def encode_graph(self, node_features: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        return self.graph_encoder(node_features, edge_index, edge_weight)

    def forward_blocks(self, block_features: torch.Tensor, block_mask: torch.Tensor,
                       recency: torch.Tensor, passage_ids: torch.Tensor,
                       node_repr: torch.Tensor, retrieved_node_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = block_mask.bool()
        h = self.block_proj(block_features)
        h = h + self.recency_emb(recency.clamp(min=0, max=self.recency_emb.num_embeddings - 1))
        h = h + self.passage_emb(passage_ids.clamp(min=0, max=self.max_retrieved))
        q_mask = mask & (passage_ids == self.max_retrieved)
        empty_q = ~q_mask.any(dim=1)
        if empty_q.any():
            q_mask = q_mask.clone()
            last = mask.long().cumsum(dim=1)
            last = last == last.max(dim=1, keepdim=True).values
            q_mask[empty_q] = mask[empty_q] & last[empty_q]
        q, q_pad = gather_query_blocks(h, q_mask)
        h = h + self.shortq_proj(shortq_block_scores(q, h, q_pad, mask).unsqueeze(-1))
        valid_nodes = retrieved_node_ids >= 0
        safe_ids = retrieved_node_ids.clamp_min(0)
        gathered = node_repr[safe_ids] * valid_nodes.unsqueeze(-1)
        empty = ~valid_nodes.any(dim=1)
        if empty.any():
            valid_nodes = valid_nodes.clone(); gathered = gathered.clone()
            valid_nodes[empty, 0] = True; gathered[empty, 0] = 0
        cross, _ = self.block_to_graph(query=h, key=gathered, value=gathered,
                                       key_padding_mask=~valid_nodes, need_weights=False)
        gate = torch.sigmoid(self.fusion_gate(torch.cat([h, cross], dim=-1)))
        h = h + gate * cross
        h = h + self.block_mlp(h)
        h = h.masked_fill((~mask).unsqueeze(-1), 0)
        return self.class_head(h), self.rank_head(h).squeeze(-1)


def pool_hidden_to_blocks(hidden: torch.Tensor, token_mask: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, seq_len, dim = hidden.shape
    nblocks = (seq_len + block_size - 1) // block_size
    pad_len = nblocks * block_size - seq_len
    if pad_len:
        hidden = F.pad(hidden, (0, 0, 0, pad_len))
        token_mask = F.pad(token_mask, (0, pad_len))
    hidden = hidden.view(bsz, nblocks, block_size, dim)
    bm = token_mask.view(bsz, nblocks, block_size).bool()
    mask_f = bm.unsqueeze(-1).to(hidden.dtype)
    mean = (hidden * mask_f).sum(dim=2) / mask_f.sum(dim=2).clamp_min(1.0)
    maxp = torch.nan_to_num(hidden.masked_fill(~bm.unsqueeze(-1), float("-inf")).max(dim=2).values,
                            nan=0.0, posinf=0.0, neginf=0.0)
    return torch.cat([mean, maxp], dim=-1), bm.any(dim=-1)


def recency_and_passage_ids(block_mask: torch.Tensor, max_retrieved: int) -> tuple[torch.Tensor, torch.Tensor]:
    mask = block_mask.bool()
    lengths = mask.sum(dim=1)
    pos = mask.long().cumsum(dim=1) - 1
    recency = (lengths.unsqueeze(1) - 1 - pos).clamp(min=0).masked_fill(~mask, 0)
    q_n = (lengths // 8).clamp(min=1)
    body_n = (lengths - q_n).clamp(min=1)
    in_body = mask & (pos < body_n.unsqueeze(1))
    body_ids = torch.div(pos * max_retrieved, body_n.unsqueeze(1).clamp_min(1), rounding_mode="floor").clamp(max=max_retrieved - 1)
    passage = torch.where(in_body, body_ids, torch.full_like(body_ids, max_retrieved)).masked_fill(~mask, 0)
    return recency, passage


def load_ranker_from_checkpoint(path: str | Path, device: torch.device) -> tuple[ShortQBlockKVRanker, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    if cfg.get("architecture") not in (None, "shortq_block_mlp"):
        raise ValueError(f"checkpoint architecture={cfg.get('architecture')} is not Short-Q")
    model = ShortQBlockKVRanker(
        llm_feat_dim=int(cfg["llm_feat_dim"]), node_dim=int(cfg["node_dim"]),
        hidden_dim=int(cfg["hidden_dim"]), gnn_layers=int(cfg["gnn_layers"]),
        num_heads=int(cfg["num_heads"]), dropout=float(cfg["dropout"]),
        importance_levels=int(cfg["importance_levels"]), max_retrieved=int(cfg["max_retrieved"]),
        max_blocks=int(cfg["max_blocks"]),
    )
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device).eval()
    return model, cfg
