#!/usr/bin/env python3

import argparse
import math
import os

# Set BEFORE importing vLLM.
os.environ["VLLM_USE_V1"] = "1"
os.environ["PYTHONHASHSEED"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# We want ordinary single-process Prometheus collection in the frontend
# for this standalone sanity experiment.
os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from local_repro.cache_sanity_metrics import (
    snapshot_local_prefix,
    wait_for_counter,
)


MODEL = "meta-llama/Llama-3.3-70B-Instruct"

# Deliberate construction:
#
#   4096 cacheable tokens
#   + 1 token that vLLM recomputes to obtain logits.
#
PROMPT_LEN = 4097
CACHE_ELIGIBLE_TOKENS = 4096


def make_exact_length_prompt(tokenizer, length: int) -> dict:
    base = tokenizer.encode(
        "This is a controlled KV cache sanity test. "
        "Every request in this workload is exactly identical. ",
        add_special_tokens=False,
    )

    if not base:
        raise RuntimeError("Tokenizer unexpectedly produced no tokens.")

    repeats = math.ceil(length / len(base))
    token_ids = (base * repeats)[:length]

    assert len(token_ids) == length
    return {"prompt_token_ids": token_ids}


def pct(x: float) -> str:
    return f"{100.0 * x:.4f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-requests", type=int, default=250)
    args = parser.parse_args()

    n = args.num_requests

    if n < 2:
        raise ValueError("--num-requests must be at least 2")

    print("=" * 72)
    print("IDENTICAL-REQUEST GPU PREFIX-CACHE SANITY TEST")
    print("=" * 72)
    print(f"requests              = {n}")
    print(f"prompt tokens/request = {PROMPT_LEN}")
    print(f"cache-eligible tokens = {CACHE_ELIGIBLE_TOKENS}")
    print("submission            = strictly sequential")
    print("LMCache               = OFF")
    print("vLLM prefix cache     = ON")
    print("measurement           = native vLLM Prometheus counters")
    print()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL,
        use_fast=True,
    )
    prompt = make_exact_length_prompt(tokenizer, PROMPT_LEN)

    llm = LLM(
        model=MODEL,
        tensor_parallel_size=2,
        distributed_executor_backend="mp",
        quantization="fp8",
        dtype="bfloat16",
        max_model_len=8000,
        gpu_memory_utilization=0.65,

        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_num_seqs=1,

        # CRITICAL: enables native scheduler/cache statistics.
        disable_log_stats=False,
    )

    sampling = SamplingParams(
        temperature=0,
        min_tokens=1,
        max_tokens=1,
    )

    # ================================================================
    # REQUEST 0
    # ================================================================
    print("Request 0: compulsory cold request...", flush=True)

    llm.generate(
        [prompt],
        sampling,
        use_tqdm=False,
    )

    # Wait until native stats have observed request 0.
    wait_for_counter(
        "vllm:prefix_cache_queries",
        PROMPT_LEN,
    )

    first_queries, first_hits = snapshot_local_prefix()

    print(
        f"After request 0: "
        f"queries={first_queries:,} "
        f"hits={first_hits:,}",
        flush=True,
    )

    # ================================================================
    # REQUESTS 1..N-1
    # ================================================================
    print(
        f"Requests 1..{n - 1}: repeated identical requests...",
        flush=True,
    )

    for i in range(1, n):
        llm.generate(
            [prompt],
            sampling,
            use_tqdm=False,
        )

        if i % 25 == 0:
            print(
                f"  completed {i}/{n - 1}",
                flush=True,
            )

    expected_final_queries = n * PROMPT_LEN

    wait_for_counter(
        "vllm:prefix_cache_queries",
        expected_final_queries,
    )

    final_queries, final_hits = snapshot_local_prefix()

    # ================================================================
    # DELTA AFTER REQUEST 0
    # ================================================================
    post_queries = final_queries - first_queries
    post_hits = final_hits - first_hits

    expected_post_queries = (n - 1) * PROMPT_LEN
    expected_post_hits = (n - 1) * CACHE_ELIGIBLE_TOKENS

    cache_eligible_rate = (
        post_hits / expected_post_hits
        if expected_post_hits
        else 0.0
    )

    post_whole_prompt_share = (
        post_hits / post_queries
        if post_queries
        else 0.0
    )

    whole_cold_rate = (
        final_hits / final_queries
        if final_queries
        else 0.0
    )

    first_request_ok = (
        first_queries == PROMPT_LEN
        and first_hits == 0
    )

    post_requests_ok = (
        post_queries == expected_post_queries
        and post_hits == expected_post_hits
    )

    passed = first_request_ok and post_requests_ok

    print()
    print("=" * 72)
    print("RESULT")
    print("=" * 72)

    print("Request 0:")
    print(f"  native GPU queries = {first_queries:,}")
    print(f"  native GPU hits    = {first_hits:,}")

    print()
    print("Requests 1..N-1:")
    print(f"  native GPU queries = {post_queries:,}")
    print(f"  native GPU hits    = {post_hits:,}")
    print(
        f"  expected hits      = {expected_post_hits:,}"
    )

    print()
    print(
        "Post-first cache-eligible GPU hit rate = "
        f"{pct(cache_eligible_rate)}"
    )
    print(
        "Post-first whole-prompt GPU hit share = "
        f"{pct(post_whole_prompt_share)}"
    )
    print(
        "Whole cold-run native GPU hit rate    = "
        f"{pct(whole_cold_rate)}"
    )

    print()
    print("Expected:")
    print("  request 0 cache hit                = 0")
    print(
        "  requests 1..N-1 eligible hit rate = "
        "100.0000%"
    )
    print(
        "  post-first whole-prompt share      = "
        f"{pct(CACHE_ELIGIBLE_TOKENS / PROMPT_LEN)}"
    )

    print()

    if passed:
        print(
            "PASS: after the first request, every cache-eligible "
            "prompt token was served from the vLLM GPU prefix cache."
        )
    else:
        print("FAIL: native vLLM cache counters differ from theory.")

        print(
            f"  expected post queries={expected_post_queries:,}, "
            f"observed={post_queries:,}"
        )
        print(
            f"  expected post hits={expected_post_hits:,}, "
            f"observed={post_hits:,}"
        )

        raise SystemExit(1)

    print("=" * 72)


if __name__ == "__main__":
    main()