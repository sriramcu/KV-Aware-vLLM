#!/usr/bin/env python3
"""Utilities for loading token-level supervision from LLM attention dumps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


def normalize_attention_scores(scores: List[float]) -> List[float]:
    if not scores:
        return []
    s = torch.tensor(scores, dtype=torch.float32)
    if s.numel() == 1:
        return [1.0]
    s_min = float(s.min().item())
    s_max = float(s.max().item())
    if s_max - s_min < 1e-8:
        return [0.0 for _ in scores]
    return ((s - s_min) / (s_max - s_min)).tolist()


def _records_from_raw(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "items" in data and isinstance(data["items"], list):
            return data["items"]
        if "prompt_token_scores" in data:
            scores = data["prompt_token_scores"]
            if isinstance(scores, torch.Tensor):
                scores = scores.tolist()
            if scores and isinstance(scores[0], (int, float)):
                return [{"prompt_token_scores": scores}]
            return [{"prompt_token_scores": row} for row in scores]
        if "query_token_scores" in data:
            scores = data["query_token_scores"]
            if isinstance(scores, torch.Tensor):
                scores = scores.tolist()
            if scores and isinstance(scores[0], (int, float)):
                return [{"query_token_scores": scores}]
            return [{"query_token_scores": row} for row in scores]
        if "last_layer_attention" in data and "query_token_mask" in data:
            attn = data["last_layer_attention"]
            qmask = data["query_token_mask"]
            tmask = data.get("target_token_mask")
            if not isinstance(attn, torch.Tensor):
                attn = torch.as_tensor(attn)
            if not isinstance(qmask, torch.Tensor):
                qmask = torch.as_tensor(qmask).bool()
            if tmask is not None and not isinstance(tmask, torch.Tensor):
                tmask = torch.as_tensor(tmask).bool()
            records: List[Dict[str, Any]] = []
            for i in range(attn.size(0)):
                a = attn[i]  # [H, T, S] or [T, S]
                if a.dim() == 2:
                    a = a.unsqueeze(0)
                source_scores = a.mean(dim=0)  # [T, S]
                if tmask is not None:
                    tgt_valid = tmask[i]
                    source_scores = source_scores[tgt_valid]
                source_scores = source_scores.mean(dim=0)  # [S]
                query_scores = source_scores[qmask[i]].tolist()
                records.append({
                    "query_token_scores": normalize_attention_scores(query_scores)
                })
            return records
    raise ValueError("Unsupported attention supervision format")


def load_attention_supervision(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if p.suffix == ".json":
        with open(p, "r", encoding="utf-8") as f:
            raw = json.load(f)
    elif p.suffix == ".jsonl":
        with open(p, "r", encoding="utf-8") as f:
            raw = [json.loads(line) for line in f if line.strip()]
    elif p.suffix in {".pt", ".pth"}:
        raw = torch.load(p, map_location="cpu")
    else:
        raise ValueError(f"Unsupported attention supervision file: {p}")

    records = _records_from_raw(raw)
    out: List[Dict[str, Any]] = []
    for rec in records:
        scores = rec.get("prompt_token_scores")
        token_key = "prompt_tokens"
        score_key = "prompt_token_scores"
        if scores is None:
            scores = rec.get("query_token_scores")
            token_key = "query_tokens"
            score_key = "query_token_scores"
        if scores is None:
            raise ValueError(
                "Each attention supervision record needs prompt_token_scores or query_token_scores")
        if isinstance(scores, torch.Tensor):
            scores = scores.tolist()
        out.append({
            "question": rec.get("question"),
            "prompt_tokens": rec.get(token_key),
            "prompt_token_scores": normalize_attention_scores(
                [float(x) for x in scores]),
            "query_tokens": rec.get("query_tokens"),
            "query_token_scores": rec.get(score_key),
        })
    return out


def maybe_find_attention_file(run_dir: Path,
                              filename: str = "query_attention_scores.pt"
                              ) -> Optional[Path]:
    cand = run_dir / filename
    if cand.exists():
        return cand
    for alt in ("prompt_attention_scores.pt",
                "prompt_attention_scores.json",
                "prompt_attention_scores.jsonl",
                "query_attention_scores.json",
                "query_attention_scores.jsonl",
                "last_layer_prompt_attention.pt",
                "last_layer_query_attention.pt"):
        cand = run_dir / alt
        if cand.exists():
            return cand
    return None
