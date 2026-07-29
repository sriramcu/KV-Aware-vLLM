#!/usr/bin/env python3
"""Create a compact, human-readable profile of LinearRAG datasets.

The report is designed to be uploaded or pasted into ChatGPT for future
hypothesis generation. It shows:
  * questions.json structure and representative question/answer records
  * passage_embedding.parquet structure without dumping embedding vectors
  * dense top-k retrieval examples using the same retrieval rule as LinearRAG
  * the actual Llama chat-prompt shape and token/chunk counts
  * examples of repeated first-passage and first-two-passage retrieval families

Example:
  python profile_linearrag_datasets.py \
      --repo /path/to/KV-Aware-vLLM \
      --datasets hotpotqa 2wikimultihop musique medical \
      --max-questions 250 \
      --output linearrag_dataset_profile.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


SYSTEM_PROMPT = (
    "As an advanced reading comprehension assistant, your task is to analyze "
    "text passages and corresponding questions meticulously. Your response "
    'start after "Thought: ", where you will methodically break down the '
    'reasoning process, illustrating how you arrive at conclusions. Conclude '
    'with "Answer: " to present a concise, definitive response, devoid of '
    "additional elaborations."
)


def find_repo(explicit: str | None) -> Path:
    if explicit:
        repo = Path(explicit).expanduser().resolve()
        if not (repo / "Hierarchical_KV" / "LinearRAG").is_dir():
            raise FileNotFoundError(f"Not a KV-Aware-vLLM repository: {repo}")
        return repo

    candidates = [Path.cwd(), Path(__file__).resolve().parent]
    candidates.extend(Path(__file__).resolve().parents)
    for candidate in candidates:
        if (candidate / "Hierarchical_KV" / "LinearRAG").is_dir():
            return candidate.resolve()
    raise FileNotFoundError("Run inside the repository or pass --repo.")


def choose_device(value: str) -> str:
    if value != "auto":
        return value
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def md_escape(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def excerpt(text: Any, limit: int) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 1)].rstrip() + "…"


def fenced(text: Any) -> str:
    # Avoid accidentally closing the Markdown fence.
    return str(text).replace("```", "` ` `")


def signature(passages: Sequence[str], depth: int) -> str:
    raw = "\n".join(passages[:depth]).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


def load_questions(path: Path, max_questions: int | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON list in {path}")

    all_items = raw
    items = raw if max_questions is None else raw[:max_questions]
    normalized: list[dict[str, Any]] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict) or "question" not in item:
            raise ValueError(f"Question record {i} must be an object with a 'question' field")
        normalized.append(
            {
                "source_index": i,
                "question": str(item["question"]),
                "answer": str(item.get("answer", "")),
                "raw": item,
            }
        )

    key_counts = Counter()
    value_types: dict[str, Counter[str]] = defaultdict(Counter)
    for item in all_items:
        if not isinstance(item, dict):
            value_types["<record>"][type(item).__name__] += 1
            continue
        for key, value in item.items():
            key_counts[str(key)] += 1
            value_types[str(key)][type(value).__name__] += 1

    schema = {
        "total_records": len(all_items),
        "analyzed_records": len(normalized),
        "key_counts": key_counts,
        "value_types": value_types,
    }
    return normalized, schema


def load_passages(path: Path):
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas and pyarrow are required") from exc

    frame = pd.read_parquet(path)
    required = {"text", "embedding"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    passage_texts = frame["text"].astype(str).tolist()
    passage_embeddings = np.asarray(frame["embedding"].tolist(), dtype=np.float32)
    if passage_embeddings.ndim != 2:
        raise ValueError(f"Expected a 2-D embedding matrix, got {passage_embeddings.shape}")
    return frame, passage_texts, passage_embeddings


def dense_retrieve(
    questions: Sequence[dict[str, Any]],
    passage_texts: Sequence[str],
    passage_embeddings: np.ndarray,
    embedding_model_path: str,
    device: str,
    top_k: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("sentence-transformers is required") from exc

    model = SentenceTransformer(embedding_model_path, device=device)
    question_texts = [item["question"] for item in questions]
    query_embeddings = np.asarray(
        model.encode(
            question_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        ),
        dtype=np.float32,
    )

    # Matches LinearRAG's dense fallback: normalized query dot passage embedding,
    # sorted in descending order.
    scores = query_embeddings @ passage_embeddings.T
    top_indices = np.argsort(scores, axis=1)[:, ::-1][:, :top_k]

    records: list[dict[str, Any]] = []
    for row, question in enumerate(questions):
        indices = [int(i) for i in top_indices[row]]
        records.append(
            {
                **question,
                "passage_indices": indices,
                "passages": [passage_texts[i] for i in indices],
                "scores": [float(scores[row, i]) for i in indices],
            }
        )
    return records


def load_tokenizer(model_name: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required") from exc
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


def build_prompt(record: dict[str, Any], tokenizer) -> tuple[str, list[int], str]:
    prompt_user = "".join(f"{passage}\n" for passage in record["passages"])
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
    token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    return prompt_text, token_ids, prompt_user


def select_sample_indices(records: Sequence[dict[str, Any]], count: int, seed: int) -> list[int]:
    if not records or count <= 0:
        return []

    candidates = [
        0,
        min(range(len(records)), key=lambda i: len(records[i]["question"])),
        max(range(len(records)), key=lambda i: len(records[i]["question"])),
    ]
    rng = random.Random(seed)
    shuffled = list(range(len(records)))
    rng.shuffle(shuffled)
    candidates.extend(shuffled)

    selected: list[int] = []
    for idx in candidates:
        if idx not in selected:
            selected.append(idx)
        if len(selected) >= min(count, len(records)):
            break
    return selected


def repeated_groups(
    records: Sequence[dict[str, Any]], depth: int
) -> list[tuple[tuple[str, ...], list[int]]]:
    groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        passages = record["passages"]
        if len(passages) >= depth:
            groups[tuple(passages[:depth])].append(idx)
    repeated = [(prefix, members) for prefix, members in groups.items() if len(members) > 1]
    repeated.sort(key=lambda item: (-len(item[1]), item[1][0]))
    return repeated


def add_question_schema(lines: list[str], schema: dict[str, Any], records: Sequence[dict[str, Any]]) -> None:
    lines.extend(
        [
            "### `questions.json`",
            "",
            f"- Total records in file: **{schema['total_records']:,}**",
            f"- Records analyzed/retrieved: **{schema['analyzed_records']:,}**",
            "",
            "| Field | Records containing field | Observed Python types |",
            "|---|---:|---|",
        ]
    )
    for key, count in sorted(schema["key_counts"].items()):
        types = ", ".join(
            f"{name} ({n})" for name, n in schema["value_types"][key].most_common()
        )
        lines.append(f"| `{md_escape(key)}` | {count:,} | {md_escape(types)} |")

    if records:
        lines.extend(
            [
                "",
                "First raw record (embedding-free):",
                "",
                "```json",
                fenced(json.dumps(records[0]["raw"], indent=2, ensure_ascii=False)),
                "```",
            ]
        )


def add_passage_schema(lines: list[str], frame, passage_embeddings: np.ndarray) -> None:
    text_lengths = [len(str(x)) for x in frame["text"].tolist()]
    lines.extend(
        [
            "",
            "### `passage_embedding.parquet`",
            "",
            f"- Passage rows: **{len(frame):,}**",
            f"- Embedding dimension: **{passage_embeddings.shape[1]:,}**",
            (
                "- Passage characters: "
                f"mean **{statistics.mean(text_lengths):.1f}**, "
                f"p50 **{percentile(text_lengths, 50):.0f}**, "
                f"p95 **{percentile(text_lengths, 95):.0f}**, "
                f"min **{min(text_lengths, default=0)}**, max **{max(text_lengths, default=0)}**"
            ),
            "",
            "| Column | Pandas dtype | Example (embeddings suppressed) |",
            "|---|---|---|",
        ]
    )
    for column in frame.columns:
        if column == "embedding":
            example = f"[{passage_embeddings.shape[1]} floating-point values]"
        else:
            example = excerpt(frame[column].iloc[0], 180) if len(frame) else ""
        lines.append(
            f"| `{md_escape(column)}` | `{md_escape(frame[column].dtype)}` | {md_escape(example)} |"
        )


def add_sample_records(
    lines: list[str],
    records: Sequence[dict[str, Any]],
    tokenizer,
    sample_count: int,
    seed: int,
    chunk_size: int,
    passage_chars: int,
    prompt_preview_chars: int,
) -> None:
    lines.extend(["", "## Representative retrieval examples", ""])
    indices = select_sample_indices(records, sample_count, seed)
    for sample_number, idx in enumerate(indices, start=1):
        record = records[idx]
        prompt_text, token_ids, _ = build_prompt(record, tokenizer)
        full_chunks = len(token_ids) // chunk_size
        remainder = len(token_ids) % chunk_size
        lines.extend(
            [
                f"### Sample {sample_number}: request {idx}",
                "",
                f"**Question:** {record['question']}",
                "",
                f"**Gold answer:** {record['answer'] or '(empty / unavailable)'}",
                "",
                (
                    f"**Actual LLM prompt size:** {len(token_ids):,} tokens = "
                    f"{full_chunks} complete {chunk_size}-token chunks + {remainder} trailing tokens"
                ),
                "",
            ]
        )
        for rank, (passage_index, score, passage) in enumerate(
            zip(record["passage_indices"], record["scores"], record["passages"]),
            start=1,
        ):
            lines.extend(
                [
                    f"**Retrieved passage {rank}** — corpus row `{passage_index}`, score `{score:.6f}`",
                    "",
                    "> " + excerpt(passage, passage_chars).replace("\n", "\n> "),
                    "",
                ]
            )

        if sample_number == 1:
            lines.extend(
                [
                    "Prompt preview after applying the Llama chat template:",
                    "",
                    "```text",
                    fenced(excerpt(prompt_text, prompt_preview_chars)),
                    "```",
                    "",
                ]
            )


def add_group_examples(
    lines: list[str],
    records: Sequence[dict[str, Any]],
    depths: Iterable[int],
    examples_per_depth: int,
    member_limit: int,
    passage_chars: int,
) -> None:
    lines.extend(
        [
            "",
            "## Repeated retrieval-prefix families",
            "",
            (
                "These are concrete examples of different questions receiving the exact same "
                "first retrieved passage(s), in the same order."
            ),
        ]
    )

    for depth in depths:
        groups = repeated_groups(records, depth)
        requests_in_groups = sum(len(members) for _, members in groups)
        lines.extend(
            [
                "",
                f"### Shared first {depth} passage{'s' if depth != 1 else ''}",
                "",
                (
                    f"Repeated groups: **{len(groups):,}**; requests in repeated groups: "
                    f"**{requests_in_groups:,}/{len(records):,}**"
                ),
            ]
        )
        if not groups:
            lines.extend(["", "No repeated groups were found."])
            continue

        for group_number, (prefix, members) in enumerate(groups[:examples_per_depth], start=1):
            lines.extend(
                [
                    "",
                    f"#### Family {group_number} — {len(members)} requests — signature `{signature(prefix, depth)}`",
                    "",
                    "Member questions:",
                    "",
                ]
            )
            for idx in members[:member_limit]:
                lines.append(f"- Request {idx}: {records[idx]['question']}")
            if len(members) > member_limit:
                lines.append(f"- …and {len(members) - member_limit} more")

            for passage_number, passage in enumerate(prefix, start=1):
                lines.extend(
                    [
                        "",
                        f"Common passage {passage_number}:",
                        "",
                        "> " + excerpt(passage, passage_chars).replace("\n", "\n> "),
                    ]
                )


def build_dataset_report(
    dataset: str,
    repo: Path,
    max_questions: int | None,
    retrieval_top_k: int,
    embedding_model_path: str,
    device: str,
    embedding_batch_size: int,
    tokenizer,
    chunk_size: int,
    sample_questions: int,
    seed: int,
    group_depths: Sequence[int],
    group_examples: int,
    group_member_limit: int,
    passage_chars: int,
    prompt_preview_chars: int,
) -> list[str]:
    linearrag = repo / "Hierarchical_KV" / "LinearRAG"
    questions_path = linearrag / "dataset" / dataset / "questions.json"
    passage_path = linearrag / "import" / dataset / "passage_embedding.parquet"
    for path in (questions_path, passage_path):
        if not path.exists():
            raise FileNotFoundError(path)

    questions, question_schema = load_questions(questions_path, max_questions)
    frame, passage_texts, passage_embeddings = load_passages(passage_path)
    records = dense_retrieve(
        questions=questions,
        passage_texts=passage_texts,
        passage_embeddings=passage_embeddings,
        embedding_model_path=embedding_model_path,
        device=device,
        top_k=retrieval_top_k,
        batch_size=embedding_batch_size,
    )

    lines = [
        f"# Dataset: `{dataset}`",
        "",
        "## Files and run configuration",
        "",
        f"- Questions: `{questions_path}`",
        f"- Passage table: `{passage_path}`",
        f"- Dense embedding model: `{embedding_model_path}`",
        f"- Retrieval top-k: **{retrieval_top_k}**",
        f"- Questions retrieved for this report: **{len(records):,}**",
        f"- LMCache chunk size used for prompt summaries: **{chunk_size} tokens**",
        "",
        "## Source format",
        "",
    ]
    add_question_schema(lines, question_schema, questions)
    add_passage_schema(lines, frame, passage_embeddings)
    add_sample_records(
        lines,
        records,
        tokenizer,
        sample_questions,
        seed,
        chunk_size,
        passage_chars,
        prompt_preview_chars,
    )
    add_group_examples(
        lines,
        records,
        group_depths,
        group_examples,
        group_member_limit,
        passage_chars,
    )
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", help="KV-Aware-vLLM repository root")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["hotpotqa"],
        help="One or more LinearRAG dataset directory names",
    )
    parser.add_argument("--max-questions", type=int, default=250)
    parser.add_argument("--retrieval-top-k", type=int, default=5)
    parser.add_argument("--embedding-model", help="Override all-mpnet-base-v2 path")
    parser.add_argument("--llm-model", default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--sample-questions", type=int, default=4)
    parser.add_argument("--group-depths", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--group-examples", type=int, default=3)
    parser.add_argument("--group-member-limit", type=int, default=8)
    parser.add_argument("--passage-chars", type=int, default=900)
    parser.add_argument("--prompt-preview-chars", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, help="Markdown output path; stdout if omitted")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_questions is not None and args.max_questions <= 0:
        raise ValueError("--max-questions must be positive")
    for name in (
        "retrieval_top_k",
        "embedding_batch_size",
        "chunk_size",
        "sample_questions",
        "group_examples",
        "group_member_limit",
        "passage_chars",
        "prompt_preview_chars",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if any(depth <= 0 or depth > args.retrieval_top_k for depth in args.group_depths):
        raise ValueError("--group-depths values must be between 1 and retrieval-top-k")

    repo = find_repo(args.repo)
    linearrag = repo / "Hierarchical_KV" / "LinearRAG"
    embedding_model = args.embedding_model or str(linearrag / "model" / "all-mpnet-base-v2")
    device = choose_device(args.device)
    tokenizer = load_tokenizer(args.llm_model)

    combined: list[str] = [
        "# LinearRAG dataset content profile",
        "",
        (
            "This report intentionally suppresses embedding vectors and instead shows the "
            "dataset schemas, semantic retrieval examples, actual prompt sizes, and concrete "
            "shared-retrieval families."
        ),
        "",
        f"Repository: `{repo}`",
        f"Embedding device: `{device}`",
        "",
    ]

    for position, dataset in enumerate(args.datasets):
        if position:
            combined.extend(["", "---", ""])
        combined.extend(
            build_dataset_report(
                dataset=dataset,
                repo=repo,
                max_questions=args.max_questions,
                retrieval_top_k=args.retrieval_top_k,
                embedding_model_path=embedding_model,
                device=device,
                embedding_batch_size=args.embedding_batch_size,
                tokenizer=tokenizer,
                chunk_size=args.chunk_size,
                sample_questions=args.sample_questions,
                seed=args.seed + position,
                group_depths=args.group_depths,
                group_examples=args.group_examples,
                group_member_limit=args.group_member_limit,
                passage_chars=args.passage_chars,
                prompt_preview_chars=args.prompt_preview_chars,
            )
        )

    text = "\n".join(combined).rstrip() + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
