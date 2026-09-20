#!/usr/bin/env python3
"""Graph-conditioned prompt block importance ranker.

This script trains a model that outputs block-level importance tiers for each
prompt sequence, while conditioning on an already-retrieved graph/subgraph.

Supported data format (.pt / .pth / .npz):
Required keys:
- token_ids: int64 [num_samples, seq_len]
- attention_mask: int64/bool [num_samples, seq_len] (1 = valid token)
- token_labels: float32 [num_samples, seq_len]
    Token-level supervision used to derive block labels.
- node_features: float32 [num_nodes, node_dim]
- edge_index: int64 [2, num_edges]
- retrieved_node_ids: int64 [num_samples, max_retrieved]
    Padded with -1 for missing entries.

Optional keys:
- edge_weight: float32 [num_edges]
- block_labels: float32 [num_samples, num_blocks]
- block_attention_mask: bool [num_samples, num_blocks]
- train_idx / val_idx / test_idx: int64 [num_split]
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


@dataclass
class TokenRankData:
    token_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_labels: torch.Tensor
    block_labels: Optional[torch.Tensor]
    block_attention_mask: Optional[torch.Tensor]
    node_features: Optional[torch.Tensor]
    edge_index: Optional[torch.Tensor]
    edge_weight: Optional[torch.Tensor]
    retrieved_node_ids: torch.Tensor
    train_idx: torch.Tensor
    val_idx: torch.Tensor
    test_idx: torch.Tensor
    graph_node_features: Optional[List[torch.Tensor]] = None
    graph_edge_index: Optional[List[torch.Tensor]] = None
    graph_edge_weight: Optional[List[torch.Tensor]] = None
    sample_graph_id: Optional[torch.Tensor] = None
    graph_names: Optional[List[str]] = None
    tokenizer_name_or_path: Optional[str] = None
    tokenizer_vocab_size: Optional[int] = None
    pad_token_id: Optional[int] = None
    vllm_block_size: Optional[int] = None


class SAGEConvWeighted(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim, bias=True)
        self.lin_neigh = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        num_nodes = x.size(0)

        msg = x[src] * edge_weight.unsqueeze(-1)
        neigh_sum = torch.zeros_like(x)
        neigh_sum.index_add_(0, dst, msg)

        deg = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
        deg.index_add_(0, dst, edge_weight)
        deg = deg.clamp_min(1e-12).unsqueeze(-1)

        neigh_mean = neigh_sum / deg
        return self.lin_self(x) + self.lin_neigh(neigh_mean)


class GraphEncoder(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int, num_layers: int,
                 dropout: float):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.node_in = nn.Linear(node_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [SAGEConvWeighted(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.dropout = dropout

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor) -> torch.Tensor:
        h = self.node_in(node_features)
        for layer in self.layers:
            h = layer(h, edge_index, edge_weight)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return F.normalize(h, p=2, dim=-1)


class GraphConditionedTokenRanker(nn.Module):
    """Token-level scorer conditioned on retrieved graph context.

    Query tokens are fused with the retrieved subgraph through token-to-node
    cross-attention so the output can be used as a token-level KV cache policy.
    """

    def __init__(
        self,
        vocab_size: int,
        node_dim: int,
        hidden_dim: int,
        gnn_layers: int,
        tfm_layers: int,
        num_heads: int,
        dropout: float,
        importance_levels: int = 2,
        block_size: int = 16,
        prediction_unit: str = "block",
    ):
        super().__init__()
        if importance_levels not in (2, 3, 4):
            raise ValueError("importance_levels must be 2, 3, or 4")
        if prediction_unit not in ("token", "block"):
            raise ValueError("prediction_unit must be 'token' or 'block'")
        self.importance_levels = importance_levels
        self.block_size = block_size
        self.prediction_unit = prediction_unit
        self.graph_encoder = GraphEncoder(node_dim, hidden_dim, gnn_layers, dropout)

        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.token_encoder = nn.TransformerEncoder(encoder_layer,
                                                   num_layers=tfm_layers)

        self.token_to_graph_attn = nn.MultiheadAttention(hidden_dim,
                                                         num_heads,
                                                         dropout=dropout,
                                                         batch_first=True)
        self.graph_to_token = nn.Linear(hidden_dim, hidden_dim)
        self.fusion_gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.block_token_score = nn.Linear(hidden_dim, 1)
        out_dim = 1 if importance_levels == 2 else importance_levels
        self.score_head = nn.Linear(hidden_dim, out_dim)

    def _pool_tokens_to_blocks(self, tok: torch.Tensor,
                               attention_mask: torch.Tensor
                               ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, hidden = tok.shape
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        pad_len = num_blocks * self.block_size - seq_len
        if pad_len > 0:
            tok = F.pad(tok, (0, 0, 0, pad_len))
            attention_mask = F.pad(attention_mask, (0, pad_len))
        tok = tok.view(bsz, num_blocks, self.block_size, hidden)
        mask = attention_mask.view(bsz, num_blocks, self.block_size).bool()

        scores = self.block_token_score(tok).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))

        pooled = torch.zeros((bsz, num_blocks, hidden), dtype=tok.dtype, device=tok.device)
        for i in range(bsz):
            for j in range(num_blocks):
                valid_idx = mask[i, j].nonzero(as_tuple=True)[0]
                if valid_idx.numel() == 0:
                    continue
                keep_n = max(1, (int(valid_idx.numel()) + 9) // 10)
                top_local = scores[i, j, valid_idx].topk(keep_n).indices
                chosen = valid_idx[top_local]
                pooled[i, j] = tok[i, j, chosen].mean(dim=0)

        block_mask = mask.any(dim=-1)
        return pooled, block_mask

    @staticmethod
    def _gather_retrieved_nodes(node_repr: torch.Tensor,
                                retrieved_node_ids: torch.Tensor
                                ) -> Tuple[torch.Tensor, torch.Tensor]:
        # retrieved_node_ids: [B, K], padded with -1
        valid = retrieved_node_ids >= 0
        safe_ids = retrieved_node_ids.clamp_min(0)
        gathered = node_repr[safe_ids]  # [B, K, H]
        gathered = gathered * valid.unsqueeze(-1)
        # MultiheadAttention cannot handle rows where every key is masked.
        empty_rows = ~valid.any(dim=1)
        if empty_rows.any():
            valid = valid.clone()
            gathered = gathered.clone()
            valid[empty_rows, 0] = True
            gathered[empty_rows, 0] = 0
        return gathered, valid

    @staticmethod
    def _pool_retrieved_nodes(gathered: torch.Tensor,
                              valid: torch.Tensor) -> torch.Tensor:
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1).to(gathered.dtype)
        pooled = gathered.sum(dim=1) / denom
        return pooled

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        retrieved_node_ids: torch.Tensor,
    ) -> torch.Tensor:
        node_repr = self.graph_encoder(node_features, edge_index, edge_weight)
        retrieved_nodes, retrieved_valid = self._gather_retrieved_nodes(
            node_repr, retrieved_node_ids)
        graph_ctx = self._pool_retrieved_nodes(retrieved_nodes, retrieved_valid)

        tok = self.token_embed(token_ids)
        tok = tok + self.graph_to_token(graph_ctx).unsqueeze(1)

        # Let each prompt token read from the retrieved subgraph directly.
        cross_out, _ = self.token_to_graph_attn(
            query=tok,
            key=retrieved_nodes,
            value=retrieved_nodes,
            key_padding_mask=~retrieved_valid,
            need_weights=False,
        )
        gate = torch.sigmoid(self.fusion_gate(torch.cat([tok, cross_out], dim=-1)))
        tok = tok + gate * cross_out

        # src_key_padding_mask expects True for padding positions
        padding_mask = ~attention_mask.bool()
        tok = self.token_encoder(tok, src_key_padding_mask=padding_mask)

        rep = (self._pool_tokens_to_blocks(tok, attention_mask)[0]
               if self.prediction_unit == "block" else tok)
        logits = self.score_head(rep)
        if self.importance_levels == 2:
            logits = logits.squeeze(-1)
        return logits


def _use_block_targets(args: argparse.Namespace, data: TokenRankData) -> bool:
    if getattr(args, "prediction_unit", "auto") == "token":
        return False
    if getattr(args, "prediction_unit", "auto") == "block":
        if data.block_labels is None or data.block_attention_mask is None:
            raise ValueError("prediction-unit=block requires block_labels and block_attention_mask in the dataset")
        return True
    return data.block_labels is not None and data.block_attention_mask is not None


@torch.no_grad()
def evaluate(model: nn.Module, data: TokenRankData, idx: torch.Tensor,
             batch_size: int, device: torch.device,
             positive_threshold: float,
             args: Optional[argparse.Namespace] = None) -> Dict[str, float]:
    model.eval()

    all_losses = []
    all_recall_at_10 = []
    all_token_acc = []
    all_pred_levels: List[torch.Tensor] = []
    all_target_levels: List[torch.Tensor] = []

    for start in range(0, idx.numel(), batch_size):
        bidx = idx[start:start + batch_size]
        attn = data.attention_mask[bidx].to(device).bool()
        labels = data.token_labels[bidx].to(device)
        logits = _forward_batch(model, data, bidx, device)

        eval_args = args if args is not None else argparse.Namespace(
            prediction_unit="token",
            importance_levels=2 if logits.dim() == 2 else logits.size(-1),
            loss_type="bce",
            pos_weight=1.0,
            focal_alpha=0.25,
            focal_gamma=2.0,
            label_low_threshold=0.33,
            label_high_threshold=0.66,
            label_disk_threshold=0.25,
            label_cpu_threshold=0.50,
            label_gpu_threshold=0.75,
            multi_level_label_mode="adaptive",
            rank_disk_ratio=0.25,
            rank_cpu_ratio=0.35,
            rank_gpu_ratio=0.15,
            adaptive_disk_std=0.25,
            adaptive_cpu_std=0.75,
            adaptive_gpu_std=1.25,
            adaptive_min_disk_ratio=0.20,
            adaptive_max_disk_ratio=0.45,
            adaptive_min_cpu_ratio=0.20,
            adaptive_max_cpu_ratio=0.35,
            adaptive_min_gpu_ratio=0.05,
            adaptive_max_gpu_ratio=0.15,
        )
        if _use_block_targets(eval_args, data):
            valid = data.block_attention_mask[bidx].to(device).bool()
            labels = data.block_labels[bidx].to(device)
        else:
            valid = attn
        loss = compute_loss(logits, labels, valid, eval_args)
        all_losses.append(loss.item())

        if logits.dim() == 2:
            # Binary ranking metric: recall@10 over positive tokens.
            probs = torch.sigmoid(logits)
            for i in range(probs.size(0)):
                valid_i = valid[i]
                p_i = probs[i][valid_i]
                y_i = labels[i][valid_i]
                pos = (y_i >= positive_threshold).nonzero(as_tuple=True)[0]

                if pos.numel() == 0:
                    continue

                topk = min(10, p_i.numel())
                pred_idx = p_i.topk(topk).indices
                hit = torch.isin(pos, pred_idx).float().mean().item()
                all_recall_at_10.append(hit)
        else:
            pred_levels = logits.argmax(dim=-1)
            target_levels = labels_to_importance_levels(labels, eval_args, valid)
            pred_valid = pred_levels[valid]
            target_valid = target_levels[valid]
            acc = (pred_valid == target_valid).float().mean().item()
            all_token_acc.append(acc)
            all_pred_levels.append(pred_valid.detach().cpu())
            all_target_levels.append(target_valid.detach().cpu())

    if all_token_acc:
        metrics = {
            "ce": float(sum(all_losses) / max(len(all_losses), 1)),
            "token_acc": float(sum(all_token_acc) / max(len(all_token_acc), 1)),
        }
        if all_pred_levels and all_target_levels:
            pred_cat = torch.cat(all_pred_levels)
            target_cat = torch.cat(all_target_levels)
            num_levels = int(getattr(args, "importance_levels",
                                     int(pred_cat.max().item()) + 1 if pred_cat.numel() > 0 else 3))
            conf = torch.zeros((num_levels, num_levels), dtype=torch.long)
            for t, p in zip(target_cat.tolist(), pred_cat.tolist()):
                conf[int(t), int(p)] += 1
            metrics["confusion_matrix"] = conf.tolist()
            class_names = _importance_class_names(num_levels)
            for i, name in enumerate(class_names):
                tp = float(conf[i, i].item())
                fp = float(conf[:, i].sum().item() - conf[i, i].item())
                fn = float(conf[i, :].sum().item() - conf[i, i].item())
                precision = tp / max(tp + fp, 1.0)
                recall = tp / max(tp + fn, 1.0)
                f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
                support = int(conf[i, :].sum().item())
                metrics[f"{name}_precision"] = precision
                metrics[f"{name}_recall"] = recall
                metrics[f"{name}_f1"] = f1
                metrics[f"{name}_support"] = support
            metrics["macro_f1"] = float(sum(metrics[f"{n}_f1"] for n in class_names) / len(class_names))
            metrics.update(_multi_level_distance_metrics(pred_cat, target_cat, num_levels))
        return metrics
    return {
        "bce": float(sum(all_losses) / max(len(all_losses), 1)),
        "recall@10": float(sum(all_recall_at_10) / max(len(all_recall_at_10), 1)),
    }


def make_splits(num_samples: int,
                seed: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_samples, generator=g)
    n_train = int(0.8 * num_samples)
    n_val = int(0.1 * num_samples)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    return train_idx, val_idx, test_idx


def _to_tensor(x, dtype=None) -> torch.Tensor:
    t = torch.as_tensor(x)
    return t.to(dtype=dtype) if dtype is not None else t


def _is_multigraph(data: TokenRankData) -> bool:
    return (data.graph_node_features is not None
            and data.graph_edge_index is not None
            and data.graph_edge_weight is not None
            and data.sample_graph_id is not None)


def _forward_batch(model: nn.Module, data: TokenRankData, bidx: torch.Tensor,
                   device: torch.device) -> torch.Tensor:
    token_ids = data.token_ids[bidx].to(device)
    attn = data.attention_mask[bidx].to(device).bool()
    retrieved = data.retrieved_node_ids[bidx].to(device)

    if not _is_multigraph(data):
        assert data.node_features is not None
        assert data.edge_index is not None
        assert data.edge_weight is not None
        return model(token_ids, attn, data.node_features.to(device),
                     data.edge_index.to(device), data.edge_weight.to(device),
                     retrieved)

    logits_rows = []
    assert data.sample_graph_id is not None
    assert data.graph_node_features is not None
    assert data.graph_edge_index is not None
    assert data.graph_edge_weight is not None
    for local_i, sample_i in enumerate(bidx.tolist()):
        gid = int(data.sample_graph_id[sample_i].item())
        node_features = data.graph_node_features[gid].to(device)
        edge_index = data.graph_edge_index[gid].to(device)
        edge_weight = data.graph_edge_weight[gid].to(device)
        logits_i = model(token_ids[local_i:local_i + 1], attn[local_i:local_i + 1],
                         node_features, edge_index, edge_weight,
                         retrieved[local_i:local_i + 1]).squeeze(0)
        logits_rows.append(logits_i)
    return torch.stack(logits_rows, dim=0)


def _importance_class_names(levels: int) -> List[str]:
    if levels == 3:
        return ["low", "medium", "high"]
    if levels == 4:
        return ["drop", "disk", "cpu", "gpu"]
    raise ValueError(f"class names only supported for multi-class levels, got {levels}")


def _multi_level_distance_metrics(pred: torch.Tensor,
                                  target: torch.Tensor,
                                  num_levels: int) -> Dict[str, float]:
    if pred.numel() == 0:
        return {
            "adjacent_acc": 0.0,
            "mean_class_distance": 0.0,
            "normalized_distance_error": 0.0,
        }

    dist = (pred.long() - target.long()).abs().float()
    max_dist = float(max(num_levels - 1, 1))
    return {
        "adjacent_acc": float((dist <= 1).float().mean().item()),
        "mean_class_distance": float(dist.mean().item()),
        "normalized_distance_error": float((dist / max_dist).mean().item()),
    }


def labels_to_importance_levels(labels: torch.Tensor,
                                args: argparse.Namespace,
                                valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if torch.is_floating_point(labels):
        mode = getattr(args, "multi_level_label_mode",
                       getattr(args, "three_level_label_mode", "fixed"))
        num_levels = int(getattr(args, "importance_levels", 3))
        if mode == "rank":
            squeeze = False
            if labels.dim() == 1:
                labels = labels.unsqueeze(0)
                if valid_mask is not None:
                    valid_mask = valid_mask.unsqueeze(0)
                squeeze = True
            if valid_mask is None:
                valid_mask = torch.ones_like(labels, dtype=torch.bool)
            y = torch.zeros_like(labels, dtype=torch.long)
            if num_levels == 3:
                rank_ratios = [
                    float(getattr(args, "rank_medium_ratio", 0.2)),
                    float(getattr(args, "rank_high_ratio", 0.1)),
                ]
            elif num_levels == 4:
                rank_ratios = [
                    float(getattr(args, "rank_disk_ratio", 0.25)),
                    float(getattr(args, "rank_cpu_ratio", 0.35)),
                    float(getattr(args, "rank_gpu_ratio", 0.15)),
                ]
            else:
                raise ValueError(f"rank mode only supports 3 or 4 levels, got {num_levels}")
            for i in range(labels.size(0)):
                valid_i = valid_mask[i]
                idx = valid_i.nonzero(as_tuple=True)[0]
                n = int(idx.numel())
                if n == 0:
                    continue
                scores = labels[i, idx]
                order = torch.argsort(scores, descending=True)
                offset = 0
                for class_idx, ratio in zip(range(num_levels - 1, 0, -1),
                                            reversed(rank_ratios)):
                    class_n = min(n - offset,
                                  max(1, int(round(n * ratio)))) if ratio > 0 else 0
                    if class_n <= 0:
                        continue
                    y[i, idx[order[offset:offset + class_n]]] = class_idx
                    offset += class_n
            return y.squeeze(0) if squeeze else y
        if mode == "adaptive":
            squeeze = False
            if labels.dim() == 1:
                labels = labels.unsqueeze(0)
                if valid_mask is not None:
                    valid_mask = valid_mask.unsqueeze(0)
                squeeze = True
            if valid_mask is None:
                valid_mask = torch.ones_like(labels, dtype=torch.bool)
            y = torch.zeros_like(labels, dtype=torch.long)
            if num_levels == 3:
                adaptive_stds = [
                    float(getattr(args, "adaptive_medium_std", 0.5)),
                    float(getattr(args, "adaptive_high_std", 1.0)),
                ]
                min_ratios = [
                    float(getattr(args, "adaptive_min_medium_ratio", 0.10)),
                    float(getattr(args, "adaptive_min_high_ratio", 0.05)),
                ]
                max_ratios = [
                    float(getattr(args, "adaptive_max_medium_ratio", 0.30)),
                    float(getattr(args, "adaptive_max_high_ratio", 0.10)),
                ]
            elif num_levels == 4:
                adaptive_stds = [
                    float(getattr(args, "adaptive_disk_std", 0.25)),
                    float(getattr(args, "adaptive_cpu_std", 0.75)),
                    float(getattr(args, "adaptive_gpu_std", 1.25)),
                ]
                min_ratios = [
                    float(getattr(args, "adaptive_min_disk_ratio", 0.20)),
                    float(getattr(args, "adaptive_min_cpu_ratio", 0.20)),
                    float(getattr(args, "adaptive_min_gpu_ratio", 0.05)),
                ]
                max_ratios = [
                    float(getattr(args, "adaptive_max_disk_ratio", 0.45)),
                    float(getattr(args, "adaptive_max_cpu_ratio", 0.35)),
                    float(getattr(args, "adaptive_max_gpu_ratio", 0.15)),
                ]
            else:
                raise ValueError(f"adaptive mode only supports 3 or 4 levels, got {num_levels}")
            for i in range(labels.size(0)):
                idx = valid_mask[i].nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    continue
                scores = labels[i, idx]
                mean = scores.mean()
                std = scores.std(unbiased=False)
                y_i = torch.zeros_like(scores, dtype=torch.long)
                n = int(idx.numel())
                order = torch.argsort(scores, descending=True)
                class_thrs = [mean + std_mult * std for std_mult in adaptive_stds]
                next_upper = None
                for class_idx, thr in reversed(list(enumerate(class_thrs, start=1))):
                    if next_upper is None:
                        mask = scores >= thr
                    else:
                        mask = (scores >= thr) & (scores < next_upper)
                    y_i = torch.where(mask, torch.full_like(y_i, class_idx), y_i)
                    next_upper = thr

                for class_idx in range(1, num_levels):
                    min_ratio = min_ratios[class_idx - 1]
                    max_ratio = max_ratios[class_idx - 1]
                    min_count = max(1, int(round(n * min_ratio))) if min_ratio > 0 else 0
                    max_count = n if max_ratio <= 0 else max(min_count, int(round(n * max_ratio)))
                    class_pos = (y_i == class_idx).nonzero(as_tuple=True)[0]
                    if class_pos.numel() < min_count:
                        promote = order[:min_count]
                        y_i[promote] = class_idx
                    class_pos = (y_i == class_idx).nonzero(as_tuple=True)[0]
                    if class_pos.numel() > max_count:
                        keep = set(order[:max_count].tolist())
                        for pos in class_pos.tolist():
                            if pos in keep:
                                continue
                            y_i[pos] = class_idx - 1

                y[i, idx] = y_i
            return y.squeeze(0) if squeeze else y
        y = torch.zeros_like(labels, dtype=torch.long)
        if num_levels == 3:
            low = float(args.label_low_threshold)
            high = float(args.label_high_threshold)
            y = torch.where(labels >= high, torch.full_like(y, 2), y)
            mid = (labels >= low) & (labels < high)
            y = torch.where(mid, torch.full_like(y, 1), y)
        elif num_levels == 4:
            disk = float(getattr(args, "label_disk_threshold", 0.25))
            cpu = float(getattr(args, "label_cpu_threshold", 0.50))
            gpu = float(getattr(args, "label_gpu_threshold", 0.75))
            y = torch.where(labels >= gpu, torch.full_like(y, 3), y)
            y = torch.where((labels >= cpu) & (labels < gpu), torch.full_like(y, 2), y)
            y = torch.where((labels >= disk) & (labels < cpu), torch.full_like(y, 1), y)
        else:
            raise ValueError(f"fixed mode only supports 3 or 4 levels, got {num_levels}")
        return y
    y = labels.long()
    return y.clamp(min=0, max=int(getattr(args, "importance_levels", 3)) - 1)


def compute_loss(logits: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor,
                 args: argparse.Namespace) -> torch.Tensor:
    if args.importance_levels in (3, 4):
        labels_v = labels_to_importance_levels(labels, args, valid)[valid]
        logits_v = logits[valid]  # [N, C]
        class_weights = getattr(args, "multi_level_class_weights_tensor", None)
        if class_weights is not None:
            class_weights = class_weights.to(logits_v.device)
        return F.cross_entropy(logits_v, labels_v, weight=class_weights)

    logits_v = logits[valid]
    labels_v = labels[valid]
    if args.loss_type == "bce":
        pos_weight = torch.tensor(float(args.pos_weight), device=logits.device)
        return F.binary_cross_entropy_with_logits(logits_v,
                                                  labels_v,
                                                  pos_weight=pos_weight)
    if args.loss_type == "focal":
        bce = F.binary_cross_entropy_with_logits(logits_v, labels_v, reduction="none")
        pt = torch.exp(-bce)
        alpha = float(args.focal_alpha)
        alpha_t = alpha * labels_v + (1.0 - alpha) * (1.0 - labels_v)
        loss = alpha_t * ((1.0 - pt)**float(args.focal_gamma)) * bce
        return loss.mean()
    raise ValueError(f"unsupported loss_type={args.loss_type}")


def make_balanced_train_perm(train_idx: torch.Tensor,
                             sample_graph_id: Optional[torch.Tensor],
                             seed: int, epoch: int) -> torch.Tensor:
    if sample_graph_id is None:
        return train_idx[torch.randperm(train_idx.numel())]

    gid_to_indices: Dict[int, List[int]] = {}
    for i in train_idx.tolist():
        gid = int(sample_graph_id[i].item())
        gid_to_indices.setdefault(gid, []).append(i)
    if not gid_to_indices:
        return train_idx

    g = torch.Generator().manual_seed(seed + epoch)
    max_len = max(len(v) for v in gid_to_indices.values())
    mixed: List[int] = []
    for items in gid_to_indices.values():
        t = torch.tensor(items, dtype=torch.long)
        if t.numel() < max_len:
            extra = t[torch.randint(0, t.numel(), (max_len - t.numel(),), generator=g)]
            t = torch.cat([t, extra], dim=0)
        t = t[torch.randperm(t.numel(), generator=g)]
        mixed.extend(t.tolist())
    mixed_t = torch.tensor(mixed, dtype=torch.long)
    mixed_t = mixed_t[torch.randperm(mixed_t.numel(), generator=g)]
    return mixed_t


def evaluate_per_dataset(model: nn.Module, data: TokenRankData, idx: torch.Tensor,
                         batch_size: int, device: torch.device,
                         positive_threshold: float,
                         args: Optional[argparse.Namespace] = None) -> Dict[str, Dict[str, float]]:
    if data.sample_graph_id is None:
        return {}
    names = data.graph_names
    if not names:
        names = [f"graph_{i}" for i in range(int(data.sample_graph_id.max().item()) + 1)]
    result: Dict[str, Dict[str, float]] = {}
    gids = data.sample_graph_id[idx]
    for gid in sorted(set(gids.tolist())):
        mask = gids == gid
        sub_idx = idx[mask]
        if sub_idx.numel() == 0:
            continue
        name = names[int(gid)] if int(gid) < len(names) else f"graph_{int(gid)}"
        result[name] = evaluate(model, data, sub_idx, batch_size, device,
                                positive_threshold, args)
    return result


def _load_npz(path: Path, seed: int) -> TokenRankData:
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("Loading .npz requires numpy. Install numpy or use .pt data.") from exc

    with np.load(path, allow_pickle=False) as d:
        token_ids = _to_tensor(d["token_ids"], torch.long)
        attention_mask = _to_tensor(d["attention_mask"], torch.bool)
        token_labels = _to_tensor(d["token_labels"], torch.float32)
        block_labels = _to_tensor(d["block_labels"], torch.float32) if "block_labels" in d else None
        block_attention_mask = _to_tensor(d["block_attention_mask"], torch.bool) if "block_attention_mask" in d else None
        node_features = _to_tensor(d["node_features"], torch.float32)
        edge_index = _to_tensor(d["edge_index"], torch.long)
        retrieved_node_ids = _to_tensor(d["retrieved_node_ids"], torch.long)

        if "edge_weight" in d:
            edge_weight = _to_tensor(d["edge_weight"], torch.float32)
        else:
            edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32)

        if all(k in d for k in ("train_idx", "val_idx", "test_idx")):
            train_idx = _to_tensor(d["train_idx"], torch.long)
            val_idx = _to_tensor(d["val_idx"], torch.long)
            test_idx = _to_tensor(d["test_idx"], torch.long)
        else:
            train_idx, val_idx, test_idx = make_splits(token_ids.size(0), seed)

    return TokenRankData(token_ids=token_ids,
                         attention_mask=attention_mask,
                         token_labels=token_labels,
                         block_labels=block_labels,
                         block_attention_mask=block_attention_mask,
                         node_features=node_features,
                         edge_index=edge_index,
                         edge_weight=edge_weight,
                         retrieved_node_ids=retrieved_node_ids,
                         train_idx=train_idx,
                         val_idx=val_idx,
                         test_idx=test_idx,
                         graph_node_features=None,
                         graph_edge_index=None,
                         graph_edge_weight=None,
                         sample_graph_id=None,
                         graph_names=None,
                         tokenizer_name_or_path=d.get("tokenizer_name_or_path"),
                         tokenizer_vocab_size=d.get("tokenizer_vocab_size"),
                         pad_token_id=d.get("pad_token_id"),
                         vllm_block_size=d.get("vllm_block_size"))


def _load_pt(path: Path, seed: int) -> TokenRankData:
    d = torch.load(path, map_location="cpu")

    token_ids = _to_tensor(d["token_ids"], torch.long)
    attention_mask = _to_tensor(d["attention_mask"], torch.bool)
    token_labels = _to_tensor(d["token_labels"], torch.float32)
    block_labels = _to_tensor(d["block_labels"], torch.float32) if "block_labels" in d else None
    block_attention_mask = _to_tensor(d["block_attention_mask"], torch.bool) if "block_attention_mask" in d else None
    retrieved_node_ids = _to_tensor(d["retrieved_node_ids"], torch.long)
    is_multigraph = all(k in d for k in (
        "graph_node_features",
        "graph_edge_index",
        "graph_edge_weight",
        "sample_graph_id",
    ))

    if is_multigraph:
        graph_node_features = [
            _to_tensor(x, torch.float32) for x in d["graph_node_features"]
        ]
        graph_edge_index = [_to_tensor(x, torch.long) for x in d["graph_edge_index"]]
        graph_edge_weight = [
            _to_tensor(x, torch.float32) for x in d["graph_edge_weight"]
        ]
        sample_graph_id = _to_tensor(d["sample_graph_id"], torch.long)
        graph_names = list(d.get("graph_names", []))
        node_features = None
        edge_index = None
        edge_weight = None
    else:
        node_features = _to_tensor(d["node_features"], torch.float32)
        edge_index = _to_tensor(d["edge_index"], torch.long)
        edge_weight = _to_tensor(
            d.get("edge_weight", torch.ones(edge_index.size(1))), torch.float32)
        graph_node_features = None
        graph_edge_index = None
        graph_edge_weight = None
        sample_graph_id = None
        graph_names = None

    if all(k in d for k in ("train_idx", "val_idx", "test_idx")):
        train_idx = _to_tensor(d["train_idx"], torch.long)
        val_idx = _to_tensor(d["val_idx"], torch.long)
        test_idx = _to_tensor(d["test_idx"], torch.long)
    else:
        train_idx, val_idx, test_idx = make_splits(token_ids.size(0), seed)

    return TokenRankData(token_ids=token_ids,
                         attention_mask=attention_mask,
                         token_labels=token_labels,
                         block_labels=block_labels,
                         block_attention_mask=block_attention_mask,
                         node_features=node_features,
                         edge_index=edge_index,
                         edge_weight=edge_weight,
                         retrieved_node_ids=retrieved_node_ids,
                         train_idx=train_idx,
                         val_idx=val_idx,
                         test_idx=test_idx,
                         graph_node_features=graph_node_features,
                         graph_edge_index=graph_edge_index,
                         graph_edge_weight=graph_edge_weight,
                         sample_graph_id=sample_graph_id,
                         graph_names=graph_names,
                         tokenizer_name_or_path=d.get("tokenizer_name_or_path"),
                         tokenizer_vocab_size=d.get("tokenizer_vocab_size"),
                         pad_token_id=d.get("pad_token_id"),
                         vllm_block_size=d.get("vllm_block_size"))


def load_data(path: Path, seed: int) -> TokenRankData:
    if path.suffix == ".npz":
        return _load_npz(path, seed)
    if path.suffix in {".pt", ".pth"}:
        return _load_pt(path, seed)
    raise ValueError("data_path must end with .npz/.pt/.pth")


def _path_candidates(raw_path: str) -> Tuple[Path, ...]:
    p = Path(raw_path).expanduser()
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd()
    repo_roots = [cwd, script_dir]
    common_data_roots = [
        cwd / "dataset_linear-rag",
        script_dir / "dataset_linear-rag",
        cwd / "dataset",
        script_dir / "dataset",
        cwd / "LinearRAG" / "dataset",
        script_dir / "LinearRAG" / "dataset",
    ]

    if p.is_absolute():
        return (p,)

    cand = [p]
    cand.extend(root / p for root in repo_roots)
    cand.extend(root / p for root in common_data_roots)
    return tuple(dict.fromkeys(cand))


def resolve_existing_path(raw_path: str) -> Path:
    candidates = _path_candidates(raw_path)
    for c in candidates:
        if c.exists():
            return c
    tried = "\n".join(f"  - {c}" for c in candidates)
    raise FileNotFoundError(
        f"Could not find data path '{raw_path}'. Tried:\n{tried}")


def resolve_output_path(raw_path: str) -> Path:
    p = Path(raw_path).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def compute_multi_level_class_weights(data: TokenRankData,
                                      args: argparse.Namespace) -> torch.Tensor:
    if _use_block_targets(args, data):
        assert data.block_labels is not None
        assert data.block_attention_mask is not None
        labels = data.block_labels[data.train_idx]
        valid = data.block_attention_mask[data.train_idx].bool()
    else:
        labels = data.token_labels[data.train_idx]
        valid = data.attention_mask[data.train_idx].bool()
    levels = labels_to_importance_levels(labels, args, valid)
    levels_v = levels[valid]
    counts = torch.bincount(levels_v, minlength=int(args.importance_levels)).float().clamp_min(1.0)
    weights = counts.sum() / (counts.numel() * counts)
    weights = weights / weights.mean()
    return weights


def train(args: argparse.Namespace) -> Dict[str, float]:
    torch.manual_seed(args.seed)

    data_path = resolve_existing_path(args.data_path)
    data = load_data(data_path, args.seed)
    device = torch.device(args.device)

    vocab_size = max(
        int(data.tokenizer_vocab_size or 0),
        int(data.token_ids.max().item()) + 1,
        int(data.pad_token_id) + 1 if data.pad_token_id is not None else 0,
    )
    if _is_multigraph(data):
        assert data.graph_node_features is not None
        node_dim = int(data.graph_node_features[0].size(1))
    else:
        assert data.node_features is not None
        node_dim = int(data.node_features.size(1))

    model = GraphConditionedTokenRanker(vocab_size=vocab_size,
                                        node_dim=node_dim,
                                        hidden_dim=args.hidden_dim,
                                        gnn_layers=args.gnn_layers,
                                        tfm_layers=args.tfm_layers,
                                        num_heads=args.num_heads,
                                        dropout=args.dropout,
                                        importance_levels=args.importance_levels,
                                        block_size=int(data.vllm_block_size or 16),
                                        prediction_unit=("block" if _use_block_targets(args, data) else "token")).to(device)

    if args.importance_levels in (3, 4) and args.multi_level_class_weighting == "balanced":
        args.multi_level_class_weights_tensor = compute_multi_level_class_weights(data, args)
        print(f"multi_level_class_weights={args.multi_level_class_weights_tensor.tolist()}")
    else:
        args.multi_level_class_weights_tensor = None

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = None
    if args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2)
    elif args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(args.epochs, 1))

    best_val = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.balanced_sampling:
            perm = make_balanced_train_perm(data.train_idx, data.sample_graph_id,
                                            args.seed, epoch)
        else:
            perm = data.train_idx[torch.randperm(data.train_idx.numel())]

        loss_sum = 0.0
        count = 0

        for start in range(0, perm.numel(), args.batch_size):
            bidx = perm[start:start + args.batch_size]

            if _use_block_targets(args, data):
                attn = data.block_attention_mask[bidx].to(device).bool()
                labels = data.block_labels[bidx].to(device)
            else:
                attn = data.attention_mask[bidx].to(device).bool()
                labels = data.token_labels[bidx].to(device)
            logits = _forward_batch(model, data, bidx, device)

            valid = attn
            if valid.sum() == 0:
                continue

            loss = compute_loss(logits, labels, valid, args)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            bs = bidx.numel()
            loss_sum += loss.item() * bs
            count += bs

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_metrics = evaluate(model, data, data.val_idx, args.batch_size, device,
                                   args.positive_threshold, args)
            primary_key = "ce" if args.importance_levels in (3, 4) else "bce"
            val_primary = val_metrics[primary_key]
            if val_primary < (best_val - args.min_delta):
                best_val = val_primary
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if scheduler is not None:
                if args.lr_scheduler == "plateau":
                    scheduler.step(val_primary)
                else:
                    scheduler.step()
            cur_lr = optimizer.param_groups[0]["lr"]

            if args.importance_levels in (3, 4):
                class_names = _importance_class_names(args.importance_levels)
                f1_bits = " ".join(
                    f"val_{name}_f1={val_metrics.get(f'{name}_f1', float('nan')):.4f}"
                    for name in class_names[1:]
                )
                print(
                    f"epoch={epoch:04d} train_loss={loss_sum / max(count, 1):.4f} "
                    f"val_ce={val_metrics['ce']:.4f} val_token_acc={val_metrics['token_acc']:.4f} "
                    f"val_macro_f1={val_metrics.get('macro_f1', float('nan')):.4f} "
                    f"{f1_bits} "
                    f"lr={cur_lr:.2e}"
                )
            else:
                print(
                    f"epoch={epoch:04d} train_loss={loss_sum / max(count, 1):.4f} "
                    f"val_bce={val_metrics['bce']:.4f} val_recall@10={val_metrics['recall@10']:.4f} "
                    f"lr={cur_lr:.2e}"
                )
            if (args.early_stop_patience > 0
                    and epochs_without_improvement >= args.early_stop_patience):
                print(f"early stopping at epoch={epoch:04d}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_metrics = evaluate(model, data, data.val_idx, args.batch_size, device,
                           args.positive_threshold, args)
    test_metrics = evaluate(model, data, data.test_idx, args.batch_size, device,
                            args.positive_threshold, args)
    val_by_dataset = evaluate_per_dataset(model, data, data.val_idx, args.batch_size,
                                          device, args.positive_threshold, args)
    test_by_dataset = evaluate_per_dataset(model, data, data.test_idx,
                                           args.batch_size, device,
                                           args.positive_threshold, args)

    metrics = {
        "best_val_primary": best_val,
        **{f"val_{k}": v
           for k, v in val_metrics.items()},
        **{f"test_{k}": v
           for k, v in test_metrics.items()},
    }
    if val_by_dataset:
        metrics["val_by_dataset"] = val_by_dataset
    if test_by_dataset:
        metrics["test_by_dataset"] = test_by_dataset

    if args.output_model:
        out_model = resolve_output_path(args.output_model)
        ckpt = {
            "state_dict": model.state_dict(),
            "config": {
                "hidden_dim": args.hidden_dim,
                "gnn_layers": args.gnn_layers,
                "tfm_layers": args.tfm_layers,
                "num_heads": args.num_heads,
                "dropout": args.dropout,
                "vocab_size": vocab_size,
                "max_seq_len": int(data.token_ids.size(1)),
                "vllm_block_size": int(data.vllm_block_size or 16),
                "prediction_unit": model.prediction_unit,
                "tokenizer_name_or_path": data.tokenizer_name_or_path,
                "pad_token_id": data.pad_token_id,
                "importance_levels": args.importance_levels,
                "label_low_threshold": args.label_low_threshold,
                "label_high_threshold": args.label_high_threshold,
                "label_disk_threshold": args.label_disk_threshold,
                "label_cpu_threshold": args.label_cpu_threshold,
                "label_gpu_threshold": args.label_gpu_threshold,
                "three_level_label_mode": args.three_level_label_mode,
                "multi_level_label_mode": args.multi_level_label_mode,
                "rank_medium_ratio": args.rank_medium_ratio,
                "rank_high_ratio": args.rank_high_ratio,
                "rank_disk_ratio": args.rank_disk_ratio,
                "rank_cpu_ratio": args.rank_cpu_ratio,
                "rank_gpu_ratio": args.rank_gpu_ratio,
                "adaptive_disk_std": args.adaptive_disk_std,
                "adaptive_cpu_std": args.adaptive_cpu_std,
                "adaptive_gpu_std": args.adaptive_gpu_std,
                "adaptive_min_disk_ratio": args.adaptive_min_disk_ratio,
                "adaptive_max_disk_ratio": args.adaptive_max_disk_ratio,
                "adaptive_min_cpu_ratio": args.adaptive_min_cpu_ratio,
                "adaptive_max_cpu_ratio": args.adaptive_max_cpu_ratio,
                "adaptive_min_gpu_ratio": args.adaptive_min_gpu_ratio,
                "adaptive_max_gpu_ratio": args.adaptive_max_gpu_ratio,
                "three_level_class_weighting": args.three_level_class_weighting,
                "multi_level_class_weighting": args.multi_level_class_weighting,
                "loss_type": args.loss_type,
                "pos_weight": args.pos_weight,
                "focal_gamma": args.focal_gamma,
                "focal_alpha": args.focal_alpha,
            },
        }
        torch.save(ckpt, out_model)
        print(f"saved model -> {out_model}")

    if args.output_metrics:
        out_metrics = resolve_output_path(args.output_metrics)
        with open(out_metrics, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"saved metrics -> {out_metrics}")

    return metrics


@torch.no_grad()
def rank_tokens_for_sample(model: nn.Module, token_ids: torch.Tensor,
                           attention_mask: torch.Tensor,
                           retrieved_node_ids: torch.Tensor,
                           node_features: torch.Tensor,
                           edge_index: torch.Tensor,
                           edge_weight: torch.Tensor,
                           top_k: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return top-k block positions and scores for one sample.

    Shapes:
    - token_ids: [seq_len]
    - attention_mask: [seq_len]
    - retrieved_node_ids: [max_retrieved]
    """
    model.eval()
    print("attention_mask.shape: ", attention_mask.shape)
    print("token_ids.shape: ", token_ids.shape)
    print("node_features.shape: ", node_features.shape)
    print("edge_index.shape: ", edge_index.shape)
    print("edge_weight.shape: ", edge_weight.shape)
    print("retrieved_node_ids.shape: ", retrieved_node_ids.shape)

    logits = model(token_ids.unsqueeze(0), attention_mask.unsqueeze(0).bool(),
                   node_features, edge_index, edge_weight,
                   retrieved_node_ids.unsqueeze(0)).squeeze(0)
    num_blocks = logits.shape[0]
    valid = attention_mask.new_ones(num_blocks, dtype=torch.bool)
    if logits.dim() == 1:
        scores = torch.sigmoid(logits[valid])
    else:
        probs = torch.softmax(logits[valid], dim=-1)
        scores = probs[:, -1]

    k = min(top_k, scores.numel())
    top_pos_in_valid = scores.topk(k).indices
    valid_pos = valid.nonzero(as_tuple=True)[0]
    block_positions = valid_pos[top_pos_in_valid]
    block_scores = scores[top_pos_in_valid]
    return block_positions, block_scores


@torch.no_grad()
def predict_importance_levels_for_sample(
        model: nn.Module,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        retrieved_node_ids: torch.Tensor,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return predicted level and confidence for every valid block.

    Outputs:
    - levels: [num_valid_blocks] int64
    - conf: [num_valid_blocks] float32
    """
    model.eval()
    print("attention_mask.shape: ", attention_mask.shape)
    print("token_ids.shape: ", token_ids.shape)
    print("node_features.shape: ", node_features.shape)
    print("edge_index.shape: ", edge_index.shape)
    print("edge_weight.shape: ", edge_weight.shape)
    print("retrieved_node_ids.shape: ", retrieved_node_ids.shape)
    logits = model(token_ids.unsqueeze(0), attention_mask.unsqueeze(0).bool(),
                   node_features, edge_index, edge_weight,
                   retrieved_node_ids.unsqueeze(0)).squeeze(0)
    num_blocks = logits.shape[0] if logits.dim() == 2 else logits.shape[0]
    valid = torch.ones(num_blocks, dtype=torch.bool, device=logits.device)
    if logits.dim() == 1:
        probs = torch.sigmoid(logits[valid])
        levels = (probs >= 0.5).long()
        return levels, probs
    probs = torch.softmax(logits[valid], dim=-1)
    conf, levels = probs.max(dim=-1)
    return levels, conf


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Graph-conditioned prompt block importance ranker")
    p.add_argument("--data-path",
                   type=str,
                   required=True,
                   help=("Path to .npz/.pt/.pth. Relative paths are resolved "
                         "against CWD, script dir, and common dataset roots."))
    p.add_argument("--device",
                   type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--gnn-layers", type=int, default=2)
    p.add_argument("--tfm-layers", type=int, default=2)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--importance-levels", type=int, choices=[2, 3, 4], default=2)
    p.add_argument("--prediction-unit",
                   type=str,
                   choices=["auto", "token", "block"],
                   default="auto")
    p.add_argument("--label-low-threshold", type=float, default=0.33)
    p.add_argument("--label-high-threshold", type=float, default=0.66)
    p.add_argument("--three-level-label-mode",
                   type=str,
                   choices=["fixed", "rank", "adaptive"],
                   default="adaptive")
    p.add_argument("--multi-level-label-mode",
                   type=str,
                   choices=["fixed", "rank", "adaptive"],
                   default="adaptive")
    p.add_argument("--rank-medium-ratio", type=float, default=0.20)
    p.add_argument("--rank-high-ratio", type=float, default=0.10)
    p.add_argument("--rank-disk-ratio", type=float, default=0.25)
    p.add_argument("--rank-cpu-ratio", type=float, default=0.35)
    p.add_argument("--rank-gpu-ratio", type=float, default=0.15)
    p.add_argument("--adaptive-medium-std", type=float, default=0.5)
    p.add_argument("--adaptive-high-std", type=float, default=1.0)
    p.add_argument("--adaptive-disk-std", type=float, default=0.25)
    p.add_argument("--adaptive-cpu-std", type=float, default=0.75)
    p.add_argument("--adaptive-gpu-std", type=float, default=1.25)
    p.add_argument("--adaptive-min-medium-ratio", type=float, default=0.10)
    p.add_argument("--adaptive-max-medium-ratio", type=float, default=0.30)
    p.add_argument("--adaptive-min-high-ratio", type=float, default=0.05)
    p.add_argument("--adaptive-max-high-ratio", type=float, default=0.10)
    p.add_argument("--adaptive-min-disk-ratio", type=float, default=0.20)
    p.add_argument("--adaptive-max-disk-ratio", type=float, default=0.45)
    p.add_argument("--adaptive-min-cpu-ratio", type=float, default=0.20)
    p.add_argument("--adaptive-max-cpu-ratio", type=float, default=0.35)
    p.add_argument("--adaptive-min-gpu-ratio", type=float, default=0.05)
    p.add_argument("--adaptive-max-gpu-ratio", type=float, default=0.15)
    p.add_argument("--label-disk-threshold", type=float, default=0.25)
    p.add_argument("--label-cpu-threshold", type=float, default=0.50)
    p.add_argument("--label-gpu-threshold", type=float, default=0.75)
    p.add_argument("--three-level-class-weighting",
                   type=str,
                   choices=["none", "balanced"],
                   default="balanced")
    p.add_argument("--multi-level-class-weighting",
                   type=str,
                   choices=["none", "balanced"],
                   default="balanced")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--positive-threshold", type=float, default=0.5)
    p.add_argument("--loss-type", type=str, choices=["bce", "focal"], default="bce")
    p.add_argument("--pos-weight", type=float, default=1.0)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--focal-alpha", type=float, default=0.25)
    p.add_argument("--balanced-sampling", action="store_true")
    p.add_argument("--lr-scheduler",
                   type=str,
                   choices=["none", "plateau", "cosine"],
                   default="plateau")
    p.add_argument("--early-stop-patience", type=int, default=5)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-model", type=str, default="token_ranker_model.pt")
    p.add_argument("--output-metrics",
                   type=str,
                   default="token_ranker_metrics.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train(args)

    print("final metrics:")
    for k, v in metrics.items():
        if isinstance(v, dict) or isinstance(v, list):
            print(f"  {k}:")
            print(json.dumps(v, indent=2))
        else:
            print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    main()
