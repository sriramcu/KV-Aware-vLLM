#!/usr/bin/env python3
"""Measure exact shared-prefix opportunity in a LinearRAG dataset.

This reproduces the current KV-Aware-vLLM prompt path:
  questions.json -> dense top-k retrieval from passage_embedding.parquet
  -> fixed system prompt + retrieved passages + question
  -> Llama chat template -> token IDs

It reports:
  * exact common token prefix across all prompts
  * duplicate first-1/2/... retrieved-passage prefixes
  * best prefix match against any earlier request (order-aware)
  * adjacent-request prefix overlap
  * an all-pairs upper bound (best match against any other request)
  * raw and LMCache chunk-aligned token coverage

The chunk-aligned numbers are opportunity estimates, not guaranteed runtime hit rates:
eviction, request timing, and cache lifecycle can reduce realized reuse.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


SYSTEM_PROMPT = (
    "As an advanced reading comprehension assistant, your task is to analyze "
    "text passages and corresponding questions meticulously. Your response "
    'start after "Thought: ", where you will methodically break down the '
    'reasoning process, illustrating how you arrive at conclusions. Conclude '
    'with "Answer: " to present a concise, definitive response, devoid of '
    "additional elaborations."
)


@dataclass
class TrieNode:
    children: dict[int, "TrieNode"] = field(default_factory=dict)
    count: int = 0
    last_index: int | None = None


def find_repo(explicit: str | None) -> Path:
    if explicit:
        repo = Path(explicit).expanduser().resolve()
        if not (repo / "Hierarchical_KV" / "LinearRAG").is_dir():
            raise FileNotFoundError(f"Not a KV-Aware-vLLM repo: {repo}")
        return repo

    candidates = [Path.cwd(), Path(__file__).resolve().parent]
    candidates += list(Path(__file__).resolve().parents)
    for candidate in candidates:
        if (candidate / "Hierarchical_KV" / "LinearRAG").is_dir():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not locate KV-Aware-vLLM. Run from the repo or pass --repo."
    )


def percentile(values: Sequence[int], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else 0.0


def lcp_len(a: Sequence[int], b: Sequence[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def global_lcp(prompts: Sequence[Sequence[int]]) -> int:
    if not prompts:
        return 0
    prefix = list(prompts[0])
    for prompt in prompts[1:]:
        prefix = prefix[: lcp_len(prefix, prompt)]
        if not prefix:
            break
    return len(prefix)


def insert(root: TrieNode, tokens: Sequence[int], index: int) -> None:
    root.count += 1
    root.last_index = index
    node = root
    for token in tokens:
        node = node.children.setdefault(int(token), TrieNode())
        node.count += 1
        node.last_index = index


def best_prior_lcps(
    prompts: Sequence[Sequence[int]], order: Sequence[int]
) -> tuple[list[int], list[int | None]]:
    """Longest exact prefix match with any request earlier in this order."""
    root = TrieNode()
    lengths = [0] * len(prompts)
    matched_indices: list[int | None] = [None] * len(prompts)

    for idx in order:
        node = root
        depth = 0
        match_idx: int | None = None
        for token in prompts[idx]:
            child = node.children.get(int(token))
            if child is None:
                break
            node = child
            depth += 1
            match_idx = node.last_index
        lengths[idx] = depth
        matched_indices[idx] = match_idx
        insert(root, prompts[idx], idx)

    return lengths, matched_indices


def best_any_other_lcps(prompts: Sequence[Sequence[int]]) -> list[int]:
    """Longest exact prefix match against any other request (order-free upper bound)."""
    root = TrieNode()
    for idx, tokens in enumerate(prompts):
        insert(root, tokens, idx)

    out: list[int] = []
    for tokens in prompts:
        node = root
        depth = 0
        for token in tokens:
            child = node.children.get(int(token))
            if child is None or child.count < 2:
                break
            node = child
            depth += 1
        out.append(depth)
    return out


def adjacent_lcps(prompts: Sequence[Sequence[int]], order: Sequence[int]) -> list[int]:
    values = [0] * len(prompts)
    for pos in range(1, len(order)):
        current = order[pos]
        previous = order[pos - 1]
        values[current] = lcp_len(prompts[current], prompts[previous])
    return values


def passage_signature(passages: Sequence[str], n: int) -> str:
    prefix = "\n".join(passages[:n])
    return hashlib.sha1(prefix.encode("utf-8")).hexdigest()


def reordered_indices(records: Sequence[dict], n: int = 2) -> list[int]:
    return sorted(
        range(len(records)),
        key=lambda i: (
            passage_signature(records[i]["sorted_passage"], n=n),
            records[i]["question"],
        ),
    )


def chunk_align(value: int, chunk_size: int) -> int:
    return (value // chunk_size) * chunk_size


def summarize_overlap(
    name: str,
    overlap: Sequence[int],
    prompt_lengths: Sequence[int],
    chunk_size: int,
) -> None:
    aligned = [chunk_align(x, chunk_size) for x in overlap]
    total_tokens = sum(prompt_lengths)
    raw_total = sum(overlap)
    aligned_total = sum(aligned)
    n = len(overlap)

    print(f"\n{name}")
    print("-" * len(name))
    print(
        f"raw LCP/request:       mean={np.mean(overlap):8.1f}  "
        f"p50={percentile(overlap, 50):8.1f}  "
        f"p95={percentile(overlap, 95):8.1f}  max={max(overlap, default=0):6d}"
    )
    print(
        f"aligned LCP/request:   mean={np.mean(aligned):8.1f}  "
        f"p50={percentile(aligned, 50):8.1f}  "
        f"p95={percentile(aligned, 95):8.1f}  max={max(aligned, default=0):6d}"
    )
    print(
        f"requests >=1 chunk:   {sum(x >= chunk_size for x in overlap):6d}/{n} "
        f"({100.0 * sum(x >= chunk_size for x in overlap) / max(n, 1):6.2f}%)"
    )
    print(
        f"requests >=2 chunks:  {sum(x >= 2 * chunk_size for x in overlap):6d}/{n} "
        f"({100.0 * sum(x >= 2 * chunk_size for x in overlap) / max(n, 1):6.2f}%)"
    )
    print(
        f"raw token coverage:   {raw_total:12,d}/{total_tokens:,} "
        f"({100.0 * raw_total / max(total_tokens, 1):6.2f}%)"
    )
    print(
        f"{chunk_size}-aligned coverage: {aligned_total:12,d}/{total_tokens:,} "
        f"({100.0 * aligned_total / max(total_tokens, 1):6.2f}%)"
    )


def print_passage_prefix_groups(records: Sequence[dict], max_k: int) -> None:
    print("\nExact retrieved-passage-prefix duplication")
    print("-------------------------------------------")
    for k in range(1, max_k + 1):
        counts = Counter(
            tuple(record["sorted_passage"][:k])
            for record in records
            if len(record["sorted_passage"]) >= k
        )
        repeated = [count for count in counts.values() if count > 1]
        requests_in_repeated = sum(repeated)
        print(
            f"first {k} passage(s): repeated_groups={len(repeated):5d}, "
            f"requests_in_groups={requests_in_repeated:6d}/{len(records)} "
            f"({100.0 * requests_in_repeated / max(len(records), 1):6.2f}%), "
            f"largest_group={max(repeated, default=1):4d}"
        )


def load_and_retrieve(
    questions_path: Path,
    passage_parquet: Path,
    embedding_model_path: str,
    max_questions: int | None,
    retrieval_top_k: int,
    device: str,
    embedding_batch_size: int,
) -> list[dict]:
    with questions_path.open("r", encoding="utf-8") as f:
        questions_data = json.load(f)
    if not isinstance(questions_data, list):
        raise ValueError(f"Expected a JSON list in {questions_path}")
    if max_questions is not None:
        questions_data = questions_data[:max_questions]

    try:
        import pandas as pd
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Activate the KV-Aware virtualenv first; this script needs pandas, "
            "pyarrow, and sentence-transformers."
        ) from exc

    df = pd.read_parquet(passage_parquet, columns=["text", "embedding"])
    passage_texts = df["text"].astype(str).tolist()
    passage_embeddings = np.asarray(df["embedding"].tolist(), dtype=np.float32)

    model = SentenceTransformer(embedding_model_path, device=device)
    questions = [str(item["question"]) for item in questions_data]
    q_embeddings = np.asarray(
        model.encode(
            questions,
            batch_size=embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        ),
        dtype=np.float32,
    )

    # This matches the current dense retrieval fallback: dot product followed by
    # descending score order.
    scores = q_embeddings @ passage_embeddings.T
    top_indices = np.argsort(scores, axis=1)[:, ::-1][:, :retrieval_top_k]

    records: list[dict] = []
    for row, item in enumerate(questions_data):
        idxs = top_indices[row]
        records.append(
            {
                "question": str(item["question"]),
                "answer": str(item.get("answer", "")),
                "sorted_passage": [passage_texts[int(i)] for i in idxs],
                "scores": [float(scores[row, int(i)]) for i in idxs],
            }
        )
    return records


def tokenize_prompts(records: Sequence[dict], llm_model: str) -> list[list[int]]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Activate the KV-Aware virtualenv first; this script needs transformers."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(llm_model, use_fast=True)
    prompts: list[list[int]] = []

    for record in records:
        prompt_user = "".join(f"{passage}\n" for passage in record["sorted_passage"])
        prompt_user += f"Question: {record['question']}\n Thought: "
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_user},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        # The chat template already carries Llama special tokens.
        prompts.append(
            tokenizer.encode(prompt_text, add_special_tokens=False)
        )
    return prompts


def write_csv(
    path: Path,
    records: Sequence[dict],
    prompt_lengths: Sequence[int],
    original_best: Sequence[int],
    original_match: Sequence[int | None],
    reordered_best: Sequence[int],
    reordered_match: Sequence[int | None],
    any_best: Sequence[int],
    chunk_size: int,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "request_index",
                "question",
                "prompt_tokens",
                "best_prior_original_tokens",
                "best_prior_original_aligned",
                "best_prior_original_match_index",
                "best_prior_reordered_tokens",
                "best_prior_reordered_aligned",
                "best_prior_reordered_match_index",
                "best_any_other_tokens",
                "best_any_other_aligned",
                "first_passage_sha1",
                "first_two_passages_sha1",
            ],
        )
        writer.writeheader()
        for i, record in enumerate(records):
            writer.writerow(
                {
                    "request_index": i,
                    "question": record["question"],
                    "prompt_tokens": prompt_lengths[i],
                    "best_prior_original_tokens": original_best[i],
                    "best_prior_original_aligned": chunk_align(original_best[i], chunk_size),
                    "best_prior_original_match_index": original_match[i],
                    "best_prior_reordered_tokens": reordered_best[i],
                    "best_prior_reordered_aligned": chunk_align(reordered_best[i], chunk_size),
                    "best_prior_reordered_match_index": reordered_match[i],
                    "best_any_other_tokens": any_best[i],
                    "best_any_other_aligned": chunk_align(any_best[i], chunk_size),
                    "first_passage_sha1": passage_signature(record["sorted_passage"], 1),
                    "first_two_passages_sha1": passage_signature(record["sorted_passage"], 2),
                }
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", help="KV-Aware-vLLM repository root")
    p.add_argument("--dataset-name", default="hotpotqa")
    p.add_argument("--max-questions", type=int, default=250)
    p.add_argument("--retrieval-top-k", type=int, default=5)
    p.add_argument("--chunk-size", type=int, default=512)
    p.add_argument("--llm-model", default="meta-llama/Llama-3.3-70B-Instruct")
    p.add_argument("--embedding-model", help="Override LinearRAG embedding model path")
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Embedding device; auto uses CUDA when torch reports it available",
    )
    p.add_argument("--embedding-batch-size", type=int, default=64)
    p.add_argument("--csv", type=Path, help="Optional per-request CSV output")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_questions is not None and args.max_questions <= 0:
        raise ValueError("--max-questions must be positive")
    if args.retrieval_top_k <= 0:
        raise ValueError("--retrieval-top-k must be positive")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    repo = find_repo(args.repo)
    linearrag = repo / "Hierarchical_KV" / "LinearRAG"
    questions_path = linearrag / "dataset" / args.dataset_name / "questions.json"
    passage_parquet = linearrag / "import" / args.dataset_name / "passage_embedding.parquet"
    embedding_model = args.embedding_model or str(linearrag / "model" / "all-mpnet-base-v2")

    for path in (questions_path, passage_parquet):
        if not path.exists():
            raise FileNotFoundError(path)

    print(f"repo:              {repo}")
    print(f"dataset:           {args.dataset_name}")
    print(f"questions:         {questions_path}")
    print(f"passage embeddings:{passage_parquet}")
    print(f"embedding model:   {embedding_model}")
    print(f"LLM tokenizer:     {args.llm_model}")
    print(f"retrieval top-k:   {args.retrieval_top_k}")
    print(f"LMCache chunk:     {args.chunk_size}")

    device = args.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    records = load_and_retrieve(
        questions_path=questions_path,
        passage_parquet=passage_parquet,
        embedding_model_path=embedding_model,
        max_questions=args.max_questions,
        retrieval_top_k=args.retrieval_top_k,
        device=device,
        embedding_batch_size=args.embedding_batch_size,
    )
    prompts = tokenize_prompts(records, args.llm_model)
    prompt_lengths = [len(tokens) for tokens in prompts]

    original_order = list(range(len(records)))
    first_two_reorder = reordered_indices(records, n=2)

    original_best, original_match = best_prior_lcps(prompts, original_order)
    reordered_best, reordered_match = best_prior_lcps(prompts, first_two_reorder)
    original_adjacent = adjacent_lcps(prompts, original_order)
    reordered_adjacent = adjacent_lcps(prompts, first_two_reorder)
    any_best = best_any_other_lcps(prompts)

    total_prompt_tokens = sum(prompt_lengths)
    common = global_lcp(prompts)

    print("\nDataset / prompt size")
    print("---------------------")
    print(f"requests:            {len(records):,}")
    print(f"total prompt tokens: {total_prompt_tokens:,}")
    print(
        f"tokens/request:      mean={np.mean(prompt_lengths):.1f}, "
        f"p50={percentile(prompt_lengths, 50):.1f}, "
        f"p95={percentile(prompt_lengths, 95):.1f}, "
        f"min={min(prompt_lengths, default=0)}, max={max(prompt_lengths, default=0)}"
    )
    print(
        f"global common prefix:{common:,} raw tokens; "
        f"{chunk_align(common, args.chunk_size):,} chunk-aligned tokens"
    )

    print_passage_prefix_groups(records, max_k=args.retrieval_top_k)

    summarize_overlap(
        "Upper bound: best exact prefix with any other request",
        any_best,
        prompt_lengths,
        args.chunk_size,
    )
    summarize_overlap(
        "Original order: best prefix with any earlier request",
        original_best,
        prompt_lengths,
        args.chunk_size,
    )
    summarize_overlap(
        "Original order: adjacent-request overlap",
        original_adjacent,
        prompt_lengths,
        args.chunk_size,
    )
    summarize_overlap(
        "Current first-two-passage reorder: best prefix with any earlier request",
        reordered_best,
        prompt_lengths,
        args.chunk_size,
    )
    summarize_overlap(
        "Current first-two-passage reorder: adjacent-request overlap",
        reordered_adjacent,
        prompt_lengths,
        args.chunk_size,
    )

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(
            args.csv,
            records,
            prompt_lengths,
            original_best,
            original_match,
            reordered_best,
            reordered_match,
            any_best,
            args.chunk_size,
        )
        print(f"\nper-request CSV: {args.csv.resolve()}")

    print(
        "\nInterpretation: use the 512-aligned coverage from 'best prefix with any "
        "earlier request' as a retention-free LMCache opportunity estimate. "
        "Actual hits can be lower because of eviction, request timing, and cache lifecycle."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
