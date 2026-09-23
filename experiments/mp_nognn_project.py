#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
No-GNN project driver for the standalone LMCache MP architecture.

Preserves the useful non-GNN behavior of the legacy
Hierarchical_KV/cpu_offload_lmcache_sriram.py workflow:

  dataset questions
    -> LinearRAG dense retrieval from passage_embedding.parquet
    -> legacy chat-template prompt construction
    -> reproducible request ordering
    -> bounded submission waves
    -> cold pass
    -> reverse/same warm replay
    -> per-request and phase metrics
    -> cold/warm output-consistency check

Intentionally absent:
  * GNN loading/inference
  * graph loading
  * importance-tier sidecar generation
  * VLLM_KV_IMPORTANCE_* plumbing
  * in-process vLLM LLM(...)
  * in-process LMCache construction/destruction

The vLLM server and standalone LMCache server are launched by the sbatch file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import requests
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
HKV_ROOT = REPO_ROOT / "Hierarchical_KV"
LINEARRAG_ROOT = HKV_ROOT / "LinearRAG"


@dataclass
class RequestRecord:
    source_index: int
    prompt: str
    prompt_record: dict[str, Any]


class RequestPolicy(Protocol):
    name: str

    def request_payload_extras(
        self,
        *,
        phase: str,
        source_index: int,
        prompt_record: dict[str, Any],
    ) -> dict[str, Any]:
        ...


class NoGNNPolicy:
    """No prediction and no placement metadata."""

    name = "none"

    def request_payload_extras(
        self,
        *,
        phase: str,
        source_index: int,
        prompt_record: dict[str, Any],
    ) -> dict[str, Any]:
        return {}


def load_questions(args: argparse.Namespace) -> list[dict[str, str]]:
    """
    Dataset-first behavior.

    Normal project use:
        --dataset_name hotpotqa

    resolves automatically to:
        <dataset_root>/hotpotqa/questions.json

    --questions_json remains available as an explicit override.
    """
    if args.question:
        return [{"question": args.question, "answer": ""}]

    questions_path = (
        Path(args.questions_json)
        if args.questions_json
        else Path(args.dataset_root) / args.dataset_name / "questions.json"
    )

    if not questions_path.exists():
        raise FileNotFoundError(f"Missing questions file: {questions_path}")

    with questions_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{questions_path} must contain a JSON list")

    out: list[dict[str, str]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict) or "question" not in item:
            raise ValueError(
                f"{questions_path}: entry {idx} does not contain 'question'"
            )
        out.append(
            {
                "question": str(item["question"]),
                "answer": str(item.get("answer", "")),
            }
        )
    return out


def load_embedding_model(model_path: Path):
    """
    Prefer the exact helper already shipped by the copied LinearRAG tree.
    Fall back to SentenceTransformer directly if run.py has unrelated import
    side effects in a particular vendored snapshot.
    """
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(LINEARRAG_ROOT))

    try:
        from Hierarchical_KV.LinearRAG.run import (  # type: ignore
            load_embedding_model as legacy_loader,
        )

        return legacy_loader(str(model_path))
    except Exception as first_exc:
        print(
            "[SC_EMBEDDING_LOADER_FALLBACK] "
            f"legacy LinearRAG loader failed: {first_exc!r}",
            flush=True,
        )
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as second_exc:
            raise RuntimeError(
                "Could not import either LinearRAG.load_embedding_model or "
                "sentence_transformers.SentenceTransformer"
            ) from second_exc

        return SentenceTransformer(str(model_path), device="cpu")


def dense_retrieve_from_import(
    *,
    import_root: Path,
    dataset_name: str,
    questions: list[dict[str, str]],
    embedding_model,
    retrieval_top_k: int,
) -> list[dict[str, Any]]:
    """
    Self-contained copy of the dense-import fallback behavior used by the
    legacy project. No GNN code is imported.
    """
    passage_parquet = import_root / dataset_name / "passage_embedding.parquet"
    if not passage_parquet.exists():
        raise FileNotFoundError(f"Missing passage embeddings: {passage_parquet}")

    df = pd.read_parquet(passage_parquet)
    for required in ("text", "embedding"):
        if required not in df.columns:
            raise ValueError(
                f"{passage_parquet} is missing required column {required!r}"
            )

    passage_texts = df["text"].tolist()
    passage_embeddings = np.asarray(
        df["embedding"].tolist(),
        dtype=np.float32,
    )

    results: list[dict[str, Any]] = []

    for item in questions:
        question = item["question"]
        q_emb = np.asarray(
            embedding_model.encode(
                question,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )

        scores = np.dot(passage_embeddings, q_emb)
        top_idx = np.argsort(scores)[::-1][:retrieval_top_k]

        results.append(
            {
                "question": question,
                "sorted_passage": [passage_texts[i] for i in top_idx],
                "sorted_passage_scores": [float(scores[i]) for i in top_idx],
                "gold_answer": item.get("answer", ""),
                "retrieval_mode": "dense_import_fallback",
            }
        )

    return results


def passage_prefix_signature(sorted_passage, n=2):
    prefix = "\n".join(sorted_passage[:n])
    return hashlib.sha1(prefix.encode("utf-8")).hexdigest()


def passage_signature(passage):
    return hashlib.sha1(passage.encode("utf-8")).hexdigest()


def hierarchical_passage_prefix_key(prompt_record, depth=2):
    sorted_passage = prompt_record.get("sorted_passage") or []
    signatures = tuple(
        passage_signature(sorted_passage[index])
        if index < len(sorted_passage)
        else ""
        for index in range(depth)
    )
    return (*signatures, prompt_record.get("question", ""))


def reorder_requests(
    llm_inputs,
    prompt_records,
    order_mode="legacy_prefix_hash",
    prefix_sort_depth=2,
    seed=0,
):
    if len(llm_inputs) != len(prompt_records):
        raise ValueError(
            "llm_inputs and prompt_records must have identical lengths"
        )
    if prefix_sort_depth <= 0:
        raise ValueError("prefix_sort_depth must be > 0")

    pairs = list(zip(llm_inputs, prompt_records))

    if order_mode == "legacy_prefix_hash":
        pairs.sort(
            key=lambda pair: (
                passage_prefix_signature(
                    pair[1]["sorted_passage"],
                    n=prefix_sort_depth,
                ),
                pair[1]["question"],
            )
        )
    elif order_mode == "hierarchical_prefix":
        pairs.sort(
            key=lambda pair: hierarchical_passage_prefix_key(
                pair[1],
                depth=prefix_sort_depth,
            )
        )
    elif order_mode == "seeded_shuffle":
        random.Random(seed).shuffle(pairs)
    else:
        raise ValueError(f"Unknown request order mode: {order_mode}")

    return (
        [pair[0] for pair in pairs],
        [pair[1] for pair in pairs],
    )


def build_vllm_prompts_from_retrieval_results(
    retrieval_results,
    tokenizer,
):
    """
    Preserve the legacy prompt format.
    """
    system_prompt = (
        "As an advanced reading comprehension assistant, your task is to analyze "
        "text passages and corresponding questions meticulously. Your response "
        'start after "Thought: ", where you will methodically break down the '
        "reasoning process, illustrating how you arrive at conclusions. Conclude "
        'with "Answer: " to present a concise, definitive response, devoid of '
        "additional elaborations."
    )

    llm_inputs = []
    prompt_records = []

    for retrieval_result in retrieval_results:
        question = retrieval_result["question"]
        sorted_passage = retrieval_result["sorted_passage"]

        prompt_user = ""
        for passage in sorted_passage:
            prompt_user += f"{passage}\n"
        prompt_user += f"Question: {question}\n Thought: "

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_user},
        ]

        llm_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        llm_inputs.append(llm_prompt)
        prompt_records.append(
            {
                "question": question,
                "sorted_passage": sorted_passage,
                "sorted_passage_scores": retrieval_result.get(
                    "sorted_passage_scores", []
                ),
                "gold_answer": retrieval_result.get("gold_answer", ""),
                "retrieval_mode": retrieval_result.get("retrieval_mode"),
                "llm_prompt": llm_prompt,
            }
        )

    return llm_inputs, prompt_records


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    idx = max(0, math.ceil(p * len(vals)) - 1)
    return vals[idx]


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def one_completion(
    *,
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
    min_tokens: int,
    temperature: float,
    top_p: float,
    timeout_s: float,
    payload_extras: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
        "min_tokens": min_tokens,
        "max_tokens": max_tokens,
    }
    payload.update(payload_extras)

    wall_start = time.time()
    mono_start = time.monotonic()

    try:
        resp = requests.post(
            endpoint,
            json=payload,
            timeout=timeout_s,
        )
        elapsed = time.monotonic() - mono_start

        try:
            body = resp.json()
        except Exception:
            body = None

        if resp.status_code != 200:
            return {
                "success": False,
                "http_status": resp.status_code,
                "elapsed_seconds": elapsed,
                "started_wall": wall_start,
                "error": resp.text,
                "response": body,
            }

        if not isinstance(body, dict) or not body.get("choices"):
            return {
                "success": False,
                "http_status": resp.status_code,
                "elapsed_seconds": elapsed,
                "started_wall": wall_start,
                "error": f"unexpected response body: {body!r}",
                "response": body,
            }

        usage = body.get("usage") or {}
        choice = body["choices"][0]
        text = choice.get("text") or ""

        return {
            "success": True,
            "http_status": resp.status_code,
            "elapsed_seconds": elapsed,
            "started_wall": wall_start,
            "request_id": str(body.get("id", "")),
            "prompt_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "finish_reason": choice.get("finish_reason"),
            "text": text,
            "output_sha256": sha256_text(text),
        }
    except Exception as exc:
        return {
            "success": False,
            "http_status": 0,
            "elapsed_seconds": time.monotonic() - mono_start,
            "started_wall": wall_start,
            "error": repr(exc),
        }


def generate_in_submission_batches_http(
    *,
    records: list[RequestRecord],
    endpoint: str,
    model: str,
    max_tokens: int,
    min_tokens: int,
    temperature: float,
    top_p: float,
    request_timeout_s: float,
    submission_batch_size: int,
    phase_name: str,
    policy: RequestPolicy,
    result_file,
) -> tuple[list[dict[str, Any]], float]:
    """
    Match the legacy bounded-wave behavior.

    Up to submission_batch_size requests are issued concurrently. The next
    wave does not start until every request in the current wave completes.
    """
    if submission_batch_size <= 0:
        raise ValueError("submission_batch_size must be > 0")

    phase_start = time.monotonic()
    all_results: list[dict[str, Any]] = []

    total = len(records)
    total_batches = (total + submission_batch_size - 1) // submission_batch_size

    for start_idx in range(0, total, submission_batch_size):
        end_idx = min(start_idx + submission_batch_size, total)
        batch = records[start_idx:end_idx]
        batch_number = start_idx // submission_batch_size + 1

        print(
            f"[{phase_name}] Submitting batch {batch_number}/{total_batches}: "
            f"requests {start_idx}:{end_idx} ({len(batch)} requests)",
            flush=True,
        )

        batch_start = time.monotonic()
        batch_results: list[dict[str, Any] | None] = [None] * len(batch)

        with ThreadPoolExecutor(max_workers=len(batch)) as pool:
            future_to_position = {}

            for pos, record in enumerate(batch):
                extras = policy.request_payload_extras(
                    phase=phase_name,
                    source_index=record.source_index,
                    prompt_record=record.prompt_record,
                )

                future = pool.submit(
                    one_completion,
                    endpoint=endpoint,
                    model=model,
                    prompt=record.prompt,
                    max_tokens=max_tokens,
                    min_tokens=min_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    timeout_s=request_timeout_s,
                    payload_extras=extras,
                )
                future_to_position[future] = pos

            for future in as_completed(future_to_position):
                pos = future_to_position[future]
                record = batch[pos]
                result = future.result()
                result.update(
                    {
                        "phase": phase_name,
                        "phase_position": start_idx + pos,
                        "source_index": record.source_index,
                        "question": record.prompt_record["question"],
                        "policy": policy.name,
                    }
                )
                batch_results[pos] = result

        if any(result is None for result in batch_results):
            raise RuntimeError("internal error: incomplete batch result set")

        for result in batch_results:
            assert result is not None
            all_results.append(result)

            result_file.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            result_file.flush()

            metric = {
                "phase": result["phase"],
                "source_index": result["source_index"],
                "request_id": result.get("request_id", ""),
                "success": result["success"],
                "http_status": result["http_status"],
                "elapsed_seconds": result["elapsed_seconds"],
                "prompt_tokens": result.get("prompt_tokens"),
                "output_tokens": result.get("output_tokens"),
                "finish_reason": result.get("finish_reason"),
                "output_sha256": result.get("output_sha256"),
            }
            print(
                "[SC_DRIVER_REQUEST_METRIC] "
                + json.dumps(metric, sort_keys=True, separators=(",", ":")),
                flush=True,
            )

            if not result["success"]:
                raise RuntimeError(
                    f"{phase_name} request failed: "
                    f"source_index={result['source_index']} "
                    f"status={result['http_status']} "
                    f"error={result.get('error')}"
                )

        print(
            f"[{phase_name}] Finished batch {batch_number}/{total_batches} "
            f"in {time.monotonic() - batch_start:.2f} seconds",
            flush=True,
        )

    return all_results, time.monotonic() - phase_start


def emit_phase_summary(
    phase_name: str,
    results: list[dict[str, Any]],
    elapsed: float,
):
    prompt_counts = [
        result["prompt_tokens"]
        for result in results
        if result.get("prompt_tokens") is not None
    ]
    output_counts = [
        result["output_tokens"]
        for result in results
        if result.get("output_tokens") is not None
    ]
    latencies = [
        float(result["elapsed_seconds"])
        for result in results
        if result["success"]
    ]

    finish_reasons: dict[str, int] = {}
    for result in results:
        reason = str(result.get("finish_reason") or "unknown")
        finish_reasons[reason] = finish_reasons.get(reason, 0) + 1

    request_count = len(results)
    prompt_tokens = (
        sum(prompt_counts)
        if len(prompt_counts) == request_count
        else None
    )
    output_tokens = (
        sum(output_counts)
        if len(output_counts) == request_count
        else None
    )

    summary = {
        "phase": phase_name,
        "requests": request_count,
        "successful": sum(1 for result in results if result["success"]),
        "elapsed_seconds": elapsed,
        "requests_per_second": (
            request_count / elapsed if elapsed > 0 else None
        ),
        "prompt_tokens": prompt_tokens,
        "prompt_tokens_per_second": (
            prompt_tokens / elapsed
            if prompt_tokens is not None and elapsed > 0
            else None
        ),
        "output_tokens": output_tokens,
        "output_tokens_per_second": (
            output_tokens / elapsed
            if output_tokens is not None and elapsed > 0
            else None
        ),
        "request_latency_mean_seconds": (
            sum(latencies) / len(latencies) if latencies else None
        ),
        "request_latency_p50_seconds": percentile(latencies, 0.50),
        "request_latency_p95_seconds": percentile(latencies, 0.95),
        "finish_reasons": finish_reasons,
    }

    print(
        "[SC_DRIVER_PHASE_SUMMARY] "
        + json.dumps(summary, sort_keys=True, separators=(",", ":")),
        flush=True,
    )
    return summary


def emit_output_consistency(
    cold_by_source: list[dict[str, Any]],
    warm_by_source: list[dict[str, Any]],
):
    if len(cold_by_source) != len(warm_by_source):
        raise RuntimeError("cold/warm output counts differ")

    mismatches = []

    for index, (cold, warm) in enumerate(
        zip(cold_by_source, warm_by_source)
    ):
        if (
            cold.get("output_sha256") != warm.get("output_sha256")
            or cold.get("output_tokens") != warm.get("output_tokens")
            or cold.get("finish_reason") != warm.get("finish_reason")
        ):
            mismatches.append(index)

    summary = {
        "requests": len(cold_by_source),
        "mismatch_count": len(mismatches),
        "mismatch_source_indices": mismatches[:100],
        "mismatch_indices_truncated": len(mismatches) > 100,
    }

    print(
        "[SC_DRIVER_OUTPUT_CONSISTENCY] "
        + json.dumps(summary, sort_keys=True, separators=(",", ":")),
        flush=True,
    )
    return summary


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_root",
        default=str(LINEARRAG_ROOT / "dataset"),
    )
    parser.add_argument(
        "--linearrag_import_dir",
        default=str(LINEARRAG_ROOT / "import"),
    )
    parser.add_argument(
        "--embedding_model",
        default=str(LINEARRAG_ROOT / "model/all-mpnet-base-v2"),
    )

    parser.add_argument("--dataset_name", default="hotpotqa")
    parser.add_argument("--retrieval_top_k", type=int, default=5)

    parser.add_argument("--question")
    parser.add_argument("--questions_json")
    parser.add_argument("--max_questions", type=int, default=None)

    parser.add_argument(
        "--submission_batch_size",
        type=int,
        default=8,
    )

    # Preserve the historical default model in the Python program.
    parser.add_argument(
        "--llm_model",
        default="meta-llama/Llama-3.3-70B-Instruct",
    )

    parser.add_argument(
        "--request_order",
        choices=(
            "legacy_prefix_hash",
            "hierarchical_prefix",
            "seeded_shuffle",
        ),
        default="legacy_prefix_hash",
    )
    parser.add_argument("--prefix_sort_depth", type=int, default=2)
    parser.add_argument("--request_order_seed", type=int, default=0)

    parser.add_argument(
        "--warm_order",
        choices=("same", "reverse"),
        default="reverse",
    )

    parser.add_argument(
        "--server_url",
        default="http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--request_timeout_s",
        type=float,
        default=1200.0,
    )

    # Preserve legacy SamplingParams.
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--min_tokens", type=int, default=128)
    parser.add_argument("--max_tokens", type=int, default=512)

    parser.add_argument(
        "--between_phases_seconds",
        type=float,
        default=2.0,
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--metrics_after_cold_path",
        default="",
        help="Optional path for a Prometheus /metrics snapshot immediately after cold.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    questions = load_questions(args)

    if args.max_questions is not None:
        questions = questions[: args.max_questions]

    if not questions:
        raise RuntimeError("No questions loaded")

    print(
        "[SC_DRIVER_CONFIG] "
        + json.dumps(
            {
                "dataset_name": args.dataset_name,
                "dataset_root": args.dataset_root,
                "questions_json": args.questions_json,
                "max_questions": args.max_questions,
                "llm_model": args.llm_model,
                "retrieval_top_k": args.retrieval_top_k,
                "request_order": args.request_order,
                "prefix_sort_depth": args.prefix_sort_depth,
                "request_order_seed": args.request_order_seed,
                "warm_order": args.warm_order,
                "submission_batch_size": args.submission_batch_size,
                "server_url": args.server_url,
                "policy": "none",
                "gnn_enabled": False,
                "cuda_visible_devices": os.environ.get(
                    "CUDA_VISIBLE_DEVICES"
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    print(f"Using first {len(questions)} questions", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.llm_model,
        use_fast=True,
    )
    if (
        tokenizer.pad_token_id is None
        and tokenizer.eos_token_id is not None
    ):
        tokenizer.pad_token = tokenizer.eos_token

    retrieval_start = time.monotonic()
    embedding_model = load_embedding_model(Path(args.embedding_model))

    retrieval_results = dense_retrieve_from_import(
        import_root=Path(args.linearrag_import_dir),
        dataset_name=args.dataset_name,
        questions=questions,
        embedding_model=embedding_model,
        retrieval_top_k=args.retrieval_top_k,
    )
    retrieval_elapsed = time.monotonic() - retrieval_start

    llm_inputs, prompt_records = (
        build_vllm_prompts_from_retrieval_results(
            retrieval_results,
            tokenizer,
        )
    )

    llm_inputs, prompt_records = reorder_requests(
        llm_inputs,
        prompt_records,
        order_mode=args.request_order,
        prefix_sort_depth=args.prefix_sort_depth,
        seed=args.request_order_seed,
    )

    num_requests = len(llm_inputs)

    if args.warm_order == "reverse":
        warm_source_indices = list(
            range(num_requests - 1, -1, -1)
        )
    else:
        warm_source_indices = list(range(num_requests))

    print(
        "Request ordering configuration: "
        f"request_order={args.request_order}, "
        f"prefix_sort_depth={args.prefix_sort_depth}, "
        f"request_order_seed={args.request_order_seed}, "
        f"warm_order={args.warm_order}, "
        f"requests={num_requests}",
        flush=True,
    )
    print(
        f"retrieval and vLLM prompt construction finished "
        f"in {retrieval_elapsed:.2f}s",
        flush=True,
    )

    with (out_dir / "prompt_records.jsonl").open(
        "w",
        encoding="utf-8",
    ) as f:
        for source_index, record in enumerate(prompt_records):
            row = dict(record)
            row["source_index"] = source_index
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "dataset_name": args.dataset_name,
        "num_requests": num_requests,
        "retrieval_top_k": args.retrieval_top_k,
        "retrieval_elapsed_seconds": retrieval_elapsed,
        "llm_model": args.llm_model,
        "request_order": args.request_order,
        "prefix_sort_depth": args.prefix_sort_depth,
        "request_order_seed": args.request_order_seed,
        "warm_order": args.warm_order,
        "submission_batch_size": args.submission_batch_size,
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "min_tokens": args.min_tokens,
            "max_tokens": args.max_tokens,
        },
        "policy": "none",
        "gnn_enabled": False,
    }

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    policy: RequestPolicy = NoGNNPolicy()
    endpoint = args.server_url.rstrip("/") + "/v1/completions"

    cold_records = [
        RequestRecord(
            source_index=index,
            prompt=llm_inputs[index],
            prompt_record=prompt_records[index],
        )
        for index in range(num_requests)
    ]

    warm_records = [
        RequestRecord(
            source_index=source_index,
            prompt=llm_inputs[source_index],
            prompt_record=prompt_records[source_index],
        )
        for source_index in warm_source_indices
    ]

    print("Cold run starting...", flush=True)

    with (out_dir / "cold_results.jsonl").open(
        "w",
        encoding="utf-8",
    ) as f:
        cold_results, cold_elapsed = (
            generate_in_submission_batches_http(
                records=cold_records,
                endpoint=endpoint,
                model=args.llm_model,
                max_tokens=args.max_tokens,
                min_tokens=args.min_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                request_timeout_s=args.request_timeout_s,
                submission_batch_size=args.submission_batch_size,
                phase_name="cold",
                policy=policy,
                result_file=f,
            )
        )

    print(
        f"first generation took {cold_elapsed:.2f} seconds.",
        flush=True,
    )
    cold_summary = emit_phase_summary(
        "cold",
        cold_results,
        cold_elapsed,
    )

    if args.metrics_after_cold_path:
        metrics_url = args.server_url.rstrip("/") + "/metrics"
        response = requests.get(metrics_url, timeout=30.0)
        response.raise_for_status()
        metrics_path = Path(args.metrics_after_cold_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(response.text, encoding="utf-8")
        print(f"[SC_METRICS_AFTER_COLD] path={metrics_path}", flush=True)

    if args.between_phases_seconds > 0:
        print(
            f"Waiting {args.between_phases_seconds:.2f}s "
            "before warm replay...",
            flush=True,
        )
        time.sleep(args.between_phases_seconds)

    print("Warm run starting...", flush=True)

    with (out_dir / "warm_results_execution_order.jsonl").open(
        "w",
        encoding="utf-8",
    ) as f:
        warm_results_exec, warm_elapsed = (
            generate_in_submission_batches_http(
                records=warm_records,
                endpoint=endpoint,
                model=args.llm_model,
                max_tokens=args.max_tokens,
                min_tokens=args.min_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                request_timeout_s=args.request_timeout_s,
                submission_batch_size=args.submission_batch_size,
                phase_name="warm",
                policy=policy,
                result_file=f,
            )
        )

    print(
        f"Second generation took {warm_elapsed:.2f} seconds.",
        flush=True,
    )
    warm_summary = emit_phase_summary(
        "warm",
        warm_results_exec,
        warm_elapsed,
    )

    cold_by_source: list[dict[str, Any] | None] = (
        [None] * num_requests
    )
    warm_by_source: list[dict[str, Any] | None] = (
        [None] * num_requests
    )

    for result in cold_results:
        cold_by_source[int(result["source_index"])] = result

    for result in warm_results_exec:
        warm_by_source[int(result["source_index"])] = result

    if any(result is None for result in cold_by_source):
        raise RuntimeError(
            "Failed to restore cold outputs to source order"
        )
    if any(result is None for result in warm_by_source):
        raise RuntimeError(
            "Failed to restore warm outputs to source order"
        )

    cold_final = [
        result
        for result in cold_by_source
        if result is not None
    ]
    warm_final = [
        result
        for result in warm_by_source
        if result is not None
    ]

    consistency = emit_output_consistency(
        cold_final,
        warm_final,
    )

    with (out_dir / "warm_results_source_order.jsonl").open(
        "w",
        encoding="utf-8",
    ) as f:
        for result in warm_final:
            f.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    summary = {
        "manifest": manifest,
        "cold": cold_summary,
        "warm": warm_summary,
        "output_consistency": consistency,
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print("[SC_DRIVER_METRICS_DONE]", flush=True)
    print(json.dumps(summary, indent=2), flush=True)

    print("\nWarm outputs in cold/source order:", flush=True)
    for result in warm_final:
        print(
            f"source_index={result['source_index']} "
            f"output={result.get('text', '')!r}",
            flush=True,
        )


if __name__ == "__main__":
    main()
