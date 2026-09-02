#!/usr/bin/env python3

import argparse
import math
import os
import shutil
import time
from pathlib import Path


# ================================================================
# ENVIRONMENT -- MUST BE SET BEFORE IMPORTING vLLM / LMCache
# ================================================================
os.environ["VLLM_USE_V1"] = "1"
os.environ["PYTHONHASHSEED"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_RPC_TIMEOUT"] = "1200000"

# Standalone single-process Prometheus registry in frontend.
os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)

# No GNN / importance policy.
os.environ["VLLM_KV_IMPORTANCE_ENABLE"] = "0"

# ================================================================
# FORCE THIS TO BE A CPU-ONLY LMCACHE TEST
# ================================================================
os.environ["LMCACHE_CHUNK_SIZE"] = "512"
os.environ["LMCACHE_LOCAL_CPU"] = "True"
os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "20"
os.environ["LMCACHE_LOOKUP_TIMEOUT_MS"] = "10000"

# Explicitly remove disk.
os.environ.pop("LMCACHE_LOCAL_DISK", None)
os.environ.pop("LMCACHE_MAX_LOCAL_DISK_SIZE", None)
os.environ["SC_LMCACHE_LOCAL_DISK_ENABLE"] = "0"

# Remove experimental admission/cancellation behavior from this functional test.
os.environ["SC_LMCACHE_SCHEDULER_LOOKUP_ADMISSION_ENABLE"] = "0"
os.environ["SC_LMCACHE_WORKER_LOOKUP_ADMISSION_ENABLE"] = "0"
os.environ["SC_LMCACHE_COLD_WARM_PUT_BARRIER_ENABLE"] = "0"
os.environ["SC_LMCACHE_DISK_PUT_ADMISSION_ENABLE"] = "0"
os.environ["SC_LMCACHE_PVTSC_ENABLE"] = "0"

job_id = os.environ.get("SLURM_JOB_ID", "manual")

config_path = Path(
    f"/tmp/lmcache_cpu_sanity_{job_id}.yaml"
)

# This experiment never needs a disk directory.
config_path.write_text(
    """\
chunk_size: 512
local_cpu: true
max_local_cpu_size: 20.0
local_disk: null
max_local_disk_size: 0.0
enable_async_loading: true
enable_kv_events: true
pre_caching_hash_algorithm: builtin
lookup_timeout_ms: 10000
""",
    encoding="utf-8",
)

os.environ["LMCACHE_CONFIG_FILE"] = str(config_path)


# ================================================================
# IMPORTS AFTER ENVIRONMENT CONFIGURATION
# ================================================================
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

from local_repro.cache_sanity_metrics import (
    snapshot_external_prefix,
    snapshot_local_prefix,
    wait_for_counter,
)


MODEL = "meta-llama/Llama-3.3-70B-Instruct"

PROMPT_LEN = 4097
LMCACHE_ELIGIBLE_TOKENS = 4096
CHUNK_SIZE = 512


def make_exact_length_prompt(tokenizer, length: int) -> dict:
    base = tokenizer.encode(
        "This is a controlled LMCache CPU sanity test. "
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

    assert LMCACHE_ELIGIBLE_TOKENS % CHUNK_SIZE == 0

    print("=" * 72)
    print("IDENTICAL-REQUEST CPU-ONLY LMCACHE SANITY TEST")
    print("=" * 72)
    print(f"requests                 = {n}")
    print(f"prompt tokens/request    = {PROMPT_LEN}")
    print(f"LMCache-eligible tokens  = {LMCACHE_ELIGIBLE_TOKENS}")
    print(
        f"LMCache chunks/request   = "
        f"{LMCACHE_ELIGIBLE_TOKENS // CHUNK_SIZE}"
    )
    print("vLLM prefix cache        = OFF")
    print("LMCache CPU              = ON")
    print("LMCache disk             = OFF")
    print("submission               = strictly sequential")
    print("measurement              = native vLLM external-cache counters")
    print()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL,
        use_fast=True,
    )

    prompt = make_exact_length_prompt(
        tokenizer,
        PROMPT_LEN,
    )

    kv_transfer = KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_both",
    )

    llm = LLM(
        model=MODEL,
        tensor_parallel_size=2,
        distributed_executor_backend="mp",
        quantization="fp8",
        dtype="bfloat16",
        max_model_len=8000,
        gpu_memory_utilization=0.65,

        # CRITICAL:
        # prevent the vLLM GPU prefix cache from satisfying the request.
        enable_prefix_caching=False,

        enable_chunked_prefill=True,
        max_num_seqs=1,
        kv_transfer_config=kv_transfer,

        # CRITICAL:
        # enables native scheduler + external-cache statistics.
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
    print(
        "Request 0: compulsory CPU miss / population request...",
        flush=True,
    )

    llm.generate(
        [prompt],
        sampling,
        use_tqdm=False,
    )

    # Native external lookup accounting is scheduler-side.
    wait_for_counter(
        "vllm:external_prefix_cache_queries",
        PROMPT_LEN,
    )

    first_queries, first_hits = snapshot_external_prefix()

    print(
        f"After request 0: "
        f"external_queries={first_queries:,} "
        f"external_hits={first_hits:,}",
        flush=True,
    )

    # CPU storage is cheap relative to disk, but request completion and the
    # connector's final store bookkeeping can overlap slightly. Give the first
    # request's CPU store a clean completion boundary before probing reuse.
    time.sleep(1.0)

    # ================================================================
    # REQUESTS 1..N-1
    # ================================================================
    print(
        f"Requests 1..{n - 1}: repeated CPU-hit requests...",
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

    expected_final_external_queries = n * PROMPT_LEN

    wait_for_counter(
        "vllm:external_prefix_cache_queries",
        expected_final_external_queries,
    )

    final_queries, final_hits = snapshot_external_prefix()

    # GPU prefix caching should contribute zero because it was disabled.
    local_queries, local_hits = snapshot_local_prefix()

    # ================================================================
    # POST-FIRST DELTA
    # ================================================================
    post_queries = final_queries - first_queries
    post_hits = final_hits - first_hits

    expected_post_queries = (n - 1) * PROMPT_LEN
    expected_post_hits = (
        (n - 1) * LMCACHE_ELIGIBLE_TOKENS
    )

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

    gpu_prefix_ok = local_hits == 0

    passed = (
        first_request_ok
        and post_requests_ok
        and gpu_prefix_ok
    )

    print()
    print("=" * 72)
    print("RESULT")
    print("=" * 72)

    print("Request 0:")
    print(f"  external queries = {first_queries:,}")
    print(f"  external hits    = {first_hits:,}")

    print()
    print("Requests 1..N-1:")
    print(f"  external queries = {post_queries:,}")
    print(f"  external hits    = {post_hits:,}")
    print(f"  expected hits    = {expected_post_hits:,}")

    print()
    print(
        "Post-first cache-eligible CPU hit rate = "
        f"{pct(cache_eligible_rate)}"
    )
    print(
        "Post-first whole-prompt CPU hit share = "
        f"{pct(post_whole_prompt_share)}"
    )
    print(
        "Whole cold-run external hit rate      = "
        f"{pct(whole_cold_rate)}"
    )

    print()
    print("GPU-prefix-cache sanity:")
    print(f"  local prefix queries = {local_queries:,}")
    print(f"  local prefix hits    = {local_hits:,}")

    print()
    print("Expected:")
    print("  request 0 external hit             = 0")
    print(
        "  requests 1..N-1 eligible CPU rate = "
        "100.0000%"
    )
    print("  GPU prefix hits                    = 0")

    print()

    if passed:
        print(
            "PASS: with GPU prefix caching disabled and disk absent, "
            "every LMCache-eligible token after request 0 was served "
            "from LMCache CPU."
        )
    else:
        print("FAIL: native cache counters differ from theory.")

        print(
            f"  expected post queries={expected_post_queries:,}, "
            f"observed={post_queries:,}"
        )
        print(
            f"  expected post hits={expected_post_hits:,}, "
            f"observed={post_hits:,}"
        )
        print(
            f"  expected GPU-prefix hits=0, "
            f"observed={local_hits:,}"
        )

        raise SystemExit(1)

    print("=" * 72)


if __name__ == "__main__":
    main()