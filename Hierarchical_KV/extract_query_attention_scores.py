#!/usr/bin/env python3
"""Convert raw last-layer attention dumps into query token supervision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_attention_supervision import load_attention_supervision


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True,
                   help="Raw .pt/.json/.jsonl attention dump")
    p.add_argument("--output", required=True,
                   help="Output .json file with query_token_scores")
    args = p.parse_args()

    records = load_attention_supervision(args.input)
    out = []
    for rec in records:
        out.append({
            "question": rec.get("question"),
            "query_tokens": rec.get("query_tokens"),
            "query_token_scores": rec["query_token_scores"],
        })

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
