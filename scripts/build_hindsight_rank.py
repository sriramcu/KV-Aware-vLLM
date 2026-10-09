#!/usr/bin/env python3
"""Generate budget-independent hindsight ranking from Q650 occurrence JSONL.

Use a historical trace covering the exact intended prompt population. This is
not appropriate as a causal production predictor. No runtime cache preloading.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Hierarchical_KV.shortq_placement.hindsight_oracle import rank_from_occurrences

p = argparse.ArgumentParser()
p.add_argument("--placement-trace", required=True)
p.add_argument("--output", required=True)
args = p.parse_args()
rows = [json.loads(line) for line in Path(args.placement_trace).read_text().splitlines() if line.strip()]
result = rank_from_occurrences(rows)
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
print("oracle_unique_chunks", result["unique_chunks"],
      "occurrences", result["chunk_occurrences"])
