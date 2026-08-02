# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import contextlib
import json
import os
import re
import sys
import time
import random
from dataclasses import asdict
from pathlib import Path

import requests
import torch
from transformers import AutoTokenizer

from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.v1.cache_engine import LMCacheEngineBuilder

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs

import hashlib

# ===== SC DRIVER RESOURCE MONITOR START =====
import os as _sc_os
import time as _sc_time
import threading as _sc_threading
import shutil as _sc_shutil
import subprocess as _sc_subprocess


def _sc_top_processes():
    try:
        return _sc_subprocess.check_output(
            [
                "bash",
                "-lc",
                "ps -u $USER -o pid,ppid,rss,vsz,stat,etime,cmd --sort=-rss | head -20",
            ],
            stderr=_sc_subprocess.STDOUT,
            timeout=3,
            text=True,
        ).replace("\n", " || ")
    except Exception as e:
        return repr(e)

def _sc_disk_usage(path):
    try:
        u = _sc_shutil.disk_usage(path)
        return f"{path}: used={u.used/1024**3:.1f}G free={u.free/1024**3:.1f}G total={u.total/1024**3:.1f}G"
    except Exception as e:
        return f"{path}: {e!r}"


def _sc_status():
    vals = {}
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(("VmRSS:", "VmHWM:", "VmSize:", "Threads:")):
                    k, v = line.split(":", 1)
                    vals[k] = v.strip()
    except Exception as e:
        vals["err"] = repr(e)
    return vals


def _sc_gpu():
    try:
        return _sc_subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.free,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=_sc_subprocess.STDOUT,
            timeout=3,
            text=True,
        ).replace("\n", " | ")
    except Exception as e:
        return repr(e)

def _sc_meminfo():
    try:
        keys = ("MemTotal:", "MemFree:", "MemAvailable:", "Buffers:", "Cached:", "SwapTotal:", "SwapFree:")
        out = []
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith(keys):
                    out.append(line.strip())
        return " | ".join(out)
    except Exception as e:
        return repr(e)

def _sc_env_flag(name: str, default: bool = False) -> bool:
    raw = _sc_os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"{name} has invalid boolean value {raw!r}")


def _sc_monitor_loop():
    interval_s = float(
        _sc_os.environ.get("SC_DRIVER_RESOURCE_MONITOR_INTERVAL_S", "30")
    )
    while True:
        print(
            "[SC_DRIVER_MONITOR] "
            f"pid={_sc_os.getpid()} "
            f"status={_sc_status()} "
            f"tmp={_sc_disk_usage('/tmp')} "
            f"shm={_sc_disk_usage('/dev/shm')} "
            f"lmcache={_sc_disk_usage((_sc_os.environ.get('SC_LMCACHE_DATA_DIR') or '/tmp'))} "
            f"gpu={_sc_gpu()}",
            f"top_procs={_sc_top_processes()} ",
            f"meminfo={_sc_meminfo()} ",
            flush=True,
        )
        _sc_time.sleep(interval_s)


def _sc_start_monitor():
    if not _sc_env_flag("SC_DRIVER_RESOURCE_MONITOR_ENABLE", False):
        return
    t = _sc_threading.Thread(
        target=_sc_monitor_loop,
        daemon=True,
        name="sc-driver-resource-monitor",
    )
    t.start()
# ===== SC DRIVER RESOURCE MONITOR END =====

def passage_prefix_signature(sorted_passage, n=2):
    """Legacy signature: hash the first n passages as one opaque prefix."""
    prefix = "\n".join(sorted_passage[:n])
    return hashlib.sha1(prefix.encode("utf-8")).hexdigest()


def passage_signature(passage):
    """Stable signature for one exact retrieved passage."""
    return hashlib.sha1(passage.encode("utf-8")).hexdigest()


def hierarchical_passage_prefix_key(prompt_record, depth=2):
    """
    Sort by P1, then P2, then P3, and so on.

    Unlike the legacy combined-prefix hash, this keeps requests sharing P1
    adjacent even when their P2 passages differ.
    """
    sorted_passage = prompt_record.get("sorted_passage") or []
    passage_signatures = tuple(
        passage_signature(sorted_passage[index])
        if index < len(sorted_passage)
        else ""
        for index in range(depth)
    )
    return (*passage_signatures, prompt_record.get("question", ""))


def reorder_requests(
    llm_inputs,
    prompt_records,
    order_mode="legacy_prefix_hash",
    prefix_sort_depth=2,
    seed=0,
):
    """
    Return llm_inputs and prompt_records in one of three reproducible orders.

    legacy_prefix_hash:
        Preserve the original program behavior. Hash the first
        prefix_sort_depth passages together, then sort by question.

    hierarchical_prefix:
        Sort by P1, then P2, and so on through prefix_sort_depth.

    seeded_shuffle:
        Deterministically shuffle with a fixed seed. Given identical inputs
        and the same seed, different program runs receive the same order.
    """
    if len(llm_inputs) != len(prompt_records):
        raise ValueError(
            "llm_inputs and prompt_records must have identical lengths: "
            f"{len(llm_inputs)} != {len(prompt_records)}"
        )
    if prefix_sort_depth <= 0:
        raise ValueError(
            "prefix_sort_depth must be greater than zero, "
            f"got {prefix_sort_depth}"
        )

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

    new_llm_inputs = [pair[0] for pair in pairs]
    new_prompt_records = [pair[1] for pair in pairs]

    return new_llm_inputs, new_prompt_records


HKV_ROOT = Path("/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM/Hierarchical_KV").resolve()
sys.path.insert(0, str(HKV_ROOT.parent))
sys.path.insert(0, str(HKV_ROOT))
sys.path.insert(0, str(HKV_ROOT / "LinearRAG"))

from Hierarchical_KV.linearrag_gnn_infer import (
    dense_retrieve_from_import,
    derive_retrieved_nodes,
    encode_prompt_text,
    load_graph_from_linearrag_import,
    load_questions,
)
from Hierarchical_KV.rag_query_gnn_predictor import GraphConditionedTokenRanker

from Hierarchical_KV.LinearRAG.run import load_embedding_model

vllm_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, vllm_root)

# from kvcache_monitor import (
#     install as kv_install,
#     report as kv_report,
#     reset as kv_reset,
#     to_dataframe as kv_to_dataframe,
# )
# from kvcache_visualize import visualize as kv_visualize
LM_CACHE_DISK_PATH = os.environ.get(
    "SC_LMCACHE_DATA_DIR",
    "/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/lmcache_vllm/manual",
) or "/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/lmcache_vllm/manual"
# LM_CACHE_DISK_PATH = "/scratch/sriramc2/vllm/"

# Define duplicated LMCache settings once so the generated YAML and the
# environment-variable overrides cannot accidentally diverge.
LM_CACHE_CHUNK_SIZE = 512
LM_CACHE_LOCAL_CPU = True
LM_CACHE_MAX_LOCAL_CPU_SIZE = 100.0
LM_CACHE_LOCAL_DISK = True
LM_CACHE_MAX_LOCAL_DISK_SIZE = 450.0
LM_CACHE_ENABLE_KV_EVENTS = True
LM_CACHE_PRE_CACHING_HASH_ALGORITHM = "builtin"
LM_CACHE_ENABLE_ASYNC_LOADING = True
LM_CACHE_CPU_READER_THREADS = 4
LM_CACHE_USE_EXPERIMENTAL = True
LM_CACHE_INTERNAL_API_SERVER_ENABLED = True
DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER = False
GNN_KV_BLOCK_SIZE = 16


def _bool_text(value):
    return "true" if value else "false"


def setup_environment_variables():
    def write_lmcache_config(path: str):
        config_lines = [
            f"chunk_size: {LM_CACHE_CHUNK_SIZE}",
            f"local_cpu: {_bool_text(LM_CACHE_LOCAL_CPU)}",
            f"max_local_cpu_size: {LM_CACHE_MAX_LOCAL_CPU_SIZE}",
        ]

        if LM_CACHE_LOCAL_DISK:
            config_lines.extend(
                [
                    f'local_disk: "file://{LM_CACHE_DISK_PATH}"',
                    f"max_local_disk_size: {LM_CACHE_MAX_LOCAL_DISK_SIZE}",
                ]
            )

        config_lines.extend(
            [
                f"enable_kv_events: {_bool_text(LM_CACHE_ENABLE_KV_EVENTS)}",
                (
                    "pre_caching_hash_algorithm: "
                    f"{LM_CACHE_PRE_CACHING_HASH_ALGORITHM}"
                ),
                (
                    "enable_async_loading: "
                    f"{_bool_text(LM_CACHE_ENABLE_ASYNC_LOADING)}"
                ),
            ]
        )

        config = "\n".join(config_lines)
        Path(path).write_text(config + "\n", encoding="utf-8")

    cfg_path = "./lmcache_config.yaml"
    write_lmcache_config(cfg_path)

    os.environ["LMCACHE_HOOK_ENABLE"] = "1"
    os.environ["LMCACHE_HOOK_LOG_DIR"] = os.environ.get(
        "LMCACHE_HOOK_LOG_DIR",
        "/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/lmcache_hit_hook",
    )
    os.makedirs(os.environ["LMCACHE_HOOK_LOG_DIR"], exist_ok=True)
    os.environ["LMCACHE_CONFIG_FILE"] = os.path.abspath(cfg_path)
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["PYTHONHASHSEED"] = "0"
    os.environ["VLLM_DISTRIBUTED_BACKEND"] = "nccl"
    os.environ["VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM"] = "1"
    os.environ["VLLM_RPC_TIMEOUT"] = "1200000"
    os.environ["LMCACHE_CPU_READER_THREADS"] = str(LM_CACHE_CPU_READER_THREADS)
    os.environ["VLLM_ENGINE_ITERATION_TIMEOUT_S"] = "1200"
    os.environ["VLLM_SAMPLED_TOKEN_ID_BUFFER_SIZE"] = "10"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["LMCACHE_ENABLE_ASYNC_LOADING"] = str(
        LM_CACHE_ENABLE_ASYNC_LOADING
    )
    os.environ["LMCACHE_USE_EXPERIMENTAL"] = str(LM_CACHE_USE_EXPERIMENTAL)
    os.environ["LMCACHE_CHUNK_SIZE"] = str(LM_CACHE_CHUNK_SIZE)
    os.environ["LMCACHE_LOCAL_CPU"] = str(LM_CACHE_LOCAL_CPU)
    os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = str(
        LM_CACHE_MAX_LOCAL_CPU_SIZE
    )

    if LM_CACHE_LOCAL_DISK:
        os.makedirs(f"{LM_CACHE_DISK_PATH}", exist_ok=True)
        print("XYZ created lmcache disk path")
        os.environ["LMCACHE_LOCAL_DISK"] = f"file://{LM_CACHE_DISK_PATH}"
        os.environ["LMCACHE_MAX_LOCAL_DISK_SIZE"] = str(
            LM_CACHE_MAX_LOCAL_DISK_SIZE
        )
        os.environ["DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER"] = str(
            DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER
        )
    else:
        os.environ.pop("LMCACHE_LOCAL_DISK", None)
        os.environ.pop("LMCACHE_MAX_LOCAL_DISK_SIZE", None)
        os.environ.pop("DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER", None)

    os.environ["LMCACHE_INTERNAL_API_SERVER_ENABLED"] = str(
        LM_CACHE_INTERNAL_API_SERVER_ENABLED
    )
    os.environ["PROMETHEUS_MULTIPROC_DIR"] = os.environ.get(
        "PROMETHEUS_MULTIPROC_DIR",
        "/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/prometheus_vllm",
    )
    os.makedirs(os.environ["PROMETHEUS_MULTIPROC_DIR"], exist_ok=True)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_root", default=str(HKV_ROOT / "LinearRAG/dataset"))
    parser.add_argument("--linearrag_import_dir", default=str(HKV_ROOT / "LinearRAG/import"))
    parser.add_argument("--retrieval_top_k", type=int, default=5)

    parser.add_argument(
        "--question",
        type=str,
    )
    parser.add_argument("--questions_json", type=str)
    parser.add_argument("--max_questions", type=int, default=None)
    parser.add_argument(
        "--submission_batch_size",
        type=int,
        default=8,
        help=(
            "Number of prompts passed to each blocking llm.generate() call. "
            "Use 8 with max_num_seqs=4 for bounded async lookahead."
        ),
    )

    parser.add_argument(
        "--gnn_ckpt",
        default=str(HKV_ROOT / "hierarchical-kv-gnn-3tier-compression/model.pt"),
    )
    parser.add_argument(
        "--gnn_data",
        default=str(HKV_ROOT / "data/all_multigraph_gnn_train.pt"),
    )
    parser.add_argument("--max_seq_len", type=int, default=8192)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument(
        "--embedding_model",
        default=str(HKV_ROOT / "LinearRAG/model/all-mpnet-base-v2"),
    )
    parser.add_argument("--dataset_name", default="hotpotqa")
    parser.add_argument("--llm_model", default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument(
        "--request_order",
        choices=(
            "legacy_prefix_hash",
            "hierarchical_prefix",
            "seeded_shuffle",
        ),
        default="legacy_prefix_hash",
        help=(
            "Cold request ordering. 'legacy_prefix_hash' preserves the "
            "existing combined-prefix hash behavior; 'hierarchical_prefix' "
            "sorts by P1, then P2, and so on; 'seeded_shuffle' creates a "
            "repeatable shuffled order."
        ),
    )
    parser.add_argument(
        "--prefix_sort_depth",
        type=int,
        default=2,
        help=(
            "Number of leading retrieved passages used by the legacy and "
            "hierarchical prefix-ordering modes."
        ),
    )
    parser.add_argument(
        "--request_order_seed",
        type=int,
        default=0,
        help=(
            "Fixed seed used by --request_order seeded_shuffle. Identical "
            "inputs and the same seed produce the same order across runs."
        ),
    )
    parser.add_argument(
        "--warm_order",
        choices=("same", "reverse"),
        default="reverse",
        help=(
            "Warm replay order. 'reverse' starts with the requests most "
            "recently executed in the cold pass; 'same' exactly replays the "
            "cold order."
        ),
    )

    return parser.parse_args()


@contextlib.contextmanager
def build_llm_with_lmcache(lmcache_connector: str, model: str):
    ktc = KVTransferConfig(
        kv_connector=lmcache_connector,
        kv_role="kv_both",
    )

    from vllm.config import KVEventsConfig

    kv_events_config = KVEventsConfig(enable_kv_cache_events=True)

    # llm_args = EngineArgs(
    #     model=model,
    #     kv_transfer_config=ktc,
    #     # kv_events_config=kv_events_config,
    #     max_model_len=8000,
    #     gpu_memory_utilization=0.65,
    #     dtype="bfloat16",
    #     max_num_seqs=20,
    #     tensor_parallel_size=1,
    #     enforce_eager=False,
    #     enable_chunked_prefill=True,
    #     disable_log_stats=False,
    #     distributed_executor_backend=None,
    #     quantization=None,
    #     enable_prefix_caching=True,
    # )

    llm_args = EngineArgs(
        model=model,
        kv_transfer_config=ktc,
        kv_events_config=kv_events_config,
        max_model_len=8000,
        gpu_memory_utilization=float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.65")),
        dtype="bfloat16",
        max_num_seqs=int(os.environ.get("VLLM_MAX_NUM_SEQS", "4")),
        tensor_parallel_size=int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "2")),
        enforce_eager=False,
        enable_chunked_prefill=True,
        disable_log_stats=False,
        distributed_executor_backend="mp",
        quantization="fp8",
        enable_prefix_caching=True,
    )

    if _sc_io_trace_enabled():
        print(
            "[SC_BEFORE_LLM] CUDA_VISIBLE_DEVICES=",
            os.environ.get("CUDA_VISIBLE_DEVICES"),
            flush=True,
        )
        print(
            "[SC_BEFORE_LLM] CUDA_DEVICE_ORDER=",
            os.environ.get("CUDA_DEVICE_ORDER"),
            flush=True,
        )

    print(
        "[LMCache configuration] disk path:",
        LM_CACHE_DISK_PATH,
        flush=True,
    )
    llm = LLM(**asdict(llm_args))

    try:
        yield llm
    finally:
        LMCacheEngineBuilder.destroy(ENGINE_NAME)

def _sc_io_trace_enabled() -> bool:
    return _sc_env_flag("SC_LMCACHE_IO_TRACE_ENABLE", False)


def _sc_io_proc_snapshot() -> dict[str, int]:
    values: dict[str, int] = {}
    if not _sc_io_trace_enabled():
        return values
    try:
        with open("/proc/self/io", "r", encoding="utf-8") as f:
            for line in f:
                key, value = line.split(":", 1)
                values[key.strip()] = int(value.strip())
    except Exception as exc:
        print(f"[SC_IO_DRIVER_PROC_IO_ERROR] error={exc!r}", flush=True)
    return values


def _sc_request_output_metric(phase_name, source_index, output):
    completion = output.outputs[0]
    prompt_token_ids = getattr(output, "prompt_token_ids", None)
    output_token_ids = getattr(completion, "token_ids", None)
    generated_text = getattr(completion, "text", "") or ""
    finish_reason = getattr(completion, "finish_reason", None)
    metric = {
        "phase": phase_name,
        "source_index": source_index,
        "request_id": str(getattr(output, "request_id", "")),
        "prompt_tokens": (
            len(prompt_token_ids) if prompt_token_ids is not None else None
        ),
        "output_tokens": (
            len(output_token_ids) if output_token_ids is not None else None
        ),
        "finish_reason": (
            str(finish_reason) if finish_reason is not None else None
        ),
        "output_sha256": hashlib.sha256(
            generated_text.encode("utf-8")
        ).hexdigest(),
    }
    print(
        "[SC_DRIVER_REQUEST_METRIC] "
        + json.dumps(metric, sort_keys=True, separators=(",", ":")),
        flush=True,
    )
    return metric


def _sc_emit_phase_metrics(phase_name, outputs, elapsed_seconds):
    request_metrics = [
        _sc_request_output_metric(phase_name, source_index, output)
        for source_index, output in enumerate(outputs)
    ]
    prompt_counts = [
        metric["prompt_tokens"]
        for metric in request_metrics
        if metric["prompt_tokens"] is not None
    ]
    output_counts = [
        metric["output_tokens"]
        for metric in request_metrics
        if metric["output_tokens"] is not None
    ]
    finish_reasons = {}
    for metric in request_metrics:
        reason = metric["finish_reason"] or "unknown"
        finish_reasons[reason] = finish_reasons.get(reason, 0) + 1

    requests = len(request_metrics)
    prompt_tokens = sum(prompt_counts) if len(prompt_counts) == requests else None
    output_tokens = sum(output_counts) if len(output_counts) == requests else None
    elapsed = float(elapsed_seconds)
    summary = {
        "phase": phase_name,
        "requests": requests,
        "elapsed_seconds": elapsed,
        "requests_per_second": requests / elapsed if elapsed > 0 else None,
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
        "finish_reasons": finish_reasons,
    }
    print(
        "[SC_DRIVER_PHASE_SUMMARY] "
        + json.dumps(summary, sort_keys=True, separators=(",", ":")),
        flush=True,
    )
    return request_metrics, summary


def _sc_emit_output_consistency(cold_metrics, warm_metrics):
    if len(cold_metrics) != len(warm_metrics):
        raise RuntimeError(
            "Cold/warm request metric lengths differ: "
            f"{len(cold_metrics)} != {len(warm_metrics)}"
        )
    mismatch_indices = [
        index
        for index, (cold, warm) in enumerate(zip(cold_metrics, warm_metrics))
        if cold["output_sha256"] != warm["output_sha256"]
        or cold["output_tokens"] != warm["output_tokens"]
        or cold["finish_reason"] != warm["finish_reason"]
    ]
    summary = {
        "requests": len(cold_metrics),
        "mismatch_count": len(mismatch_indices),
        "mismatch_source_indices": mismatch_indices[:100],
        "mismatch_indices_truncated": len(mismatch_indices) > 100,
    }
    print(
        "[SC_DRIVER_OUTPUT_CONSISTENCY] "
        + json.dumps(summary, sort_keys=True, separators=(",", ":")),
        flush=True,
    )
    return summary


def generate_in_submission_batches(
    llm,
    llm_inputs,
    sampling_params,
    submission_batch_size: int,
    phase_name: str,
):
    """
    Submit only a bounded wave of requests to vLLM at once.

    llm.generate() is blocking, so the next wave is submitted only after
    every request in the current wave has completed.
    """
    if submission_batch_size <= 0:
        raise ValueError(
            "submission_batch_size must be greater than zero, "
            f"got {submission_batch_size}"
        )

    total_requests = len(llm_inputs)
    all_outputs = []

    for start_idx in range(0, total_requests, submission_batch_size):
        end_idx = min(start_idx + submission_batch_size, total_requests)
        batch_inputs = llm_inputs[start_idx:end_idx]

        batch_number = start_idx // submission_batch_size + 1
        total_batches = (
            total_requests + submission_batch_size - 1
        ) // submission_batch_size

        print(
            f"[{phase_name}] Submitting batch "
            f"{batch_number}/{total_batches}: "
            f"requests {start_idx}:{end_idx} "
            f"({len(batch_inputs)} requests)",
            flush=True,
        )

        batch_start = time.time()
        batch_start_mono = time.monotonic()
        batch_io_before = _sc_io_proc_snapshot()
        if _sc_io_trace_enabled():
            print(
                "[SC_IO_DRIVER_BATCH_START] "
                f"phase={phase_name} batch={batch_number}/{total_batches} "
                f"requests={start_idx}:{end_idx} wall={batch_start:.6f} "
                f"mono={batch_start_mono:.6f} proc_io={batch_io_before}",
                flush=True,
            )

        batch_outputs = llm.generate(
            batch_inputs,
            sampling_params,
        )

        batch_end = time.time()
        batch_end_mono = time.monotonic()
        batch_elapsed = batch_end - batch_start
        batch_io_after = _sc_io_proc_snapshot()
        if _sc_io_trace_enabled():
            io_delta = {
                key: batch_io_after.get(key, 0) - batch_io_before.get(key, 0)
                for key in set(batch_io_before) | set(batch_io_after)
            }
            print(
                "[SC_IO_DRIVER_BATCH_DONE] "
                f"phase={phase_name} batch={batch_number}/{total_batches} "
                f"requests={start_idx}:{end_idx} wall={batch_end:.6f} "
                f"mono={batch_end_mono:.6f} elapsed={batch_end_mono - batch_start_mono:.6f} "
                f"proc_io_delta={io_delta}",
                flush=True,
            )

        print(
            f"[{phase_name}] Finished batch "
            f"{batch_number}/{total_batches} in "
            f"{batch_elapsed:.2f} seconds",
            flush=True,
        )

        all_outputs.extend(batch_outputs)

    if len(all_outputs) != total_requests:
        raise RuntimeError(
            "Generated output count does not match input count: "
            f"{len(all_outputs)} outputs for {total_requests} inputs"
        )

    return all_outputs

def build_prompt_text(passages, question):
    parts = [p for p in passages if p]
    if question:
        parts.append(question)
    return "\n".join(parts)


def build_vllm_prompts_from_retrieval_results(retrieval_results, tokenizer):
    system_prompt = (
        "As an advanced reading comprehension assistant, your task is to analyze "
        "text passages and corresponding questions meticulously. Your response "
        'start after "Thought: ", where you will methodically break down the '
        'reasoning process, illustrating how you arrive at conclusions. Conclude '
        'with "Answer: " to present a concise, definitive response, devoid of '
        "additional elaborations."
    )

    llm_inputs = []
    prompt_records = []

    for retrieval_result in retrieval_results:
        question = retrieval_result["question"]
        sorted_passage = retrieval_result["sorted_passage"]

        gnn_prompt = build_prompt_text(sorted_passage, question)

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
        prompt_records.append({
            "question": question,
            "sorted_passage": sorted_passage,
            "gnn_prompt": gnn_prompt,
            "llm_prompt": llm_prompt,
        })

    return llm_inputs, prompt_records


def build_gnn_predictor(args, node_dim):
    gnn_data = torch.load(args.gnn_data, map_location="cpu")
    ckpt = torch.load(args.gnn_ckpt, map_location="cpu")
    cfg = ckpt["config"]

    block_size = int(cfg.get("vllm_block_size", gnn_data.get("vllm_block_size", 16)))

    model = GraphConditionedTokenRanker(
        vocab_size=int(cfg["vocab_size"]),
        node_dim=node_dim,
        hidden_dim=int(cfg["hidden_dim"]),
        gnn_layers=int(cfg["gnn_layers"]),
        tfm_layers=int(cfg["tfm_layers"]),
        num_heads=int(cfg["num_heads"]),
        dropout=float(cfg["dropout"]),
        importance_levels=int(cfg.get("importance_levels", 2)),
        block_size=block_size,
        prediction_unit=str(cfg.get("prediction_unit", "block")),
    ).to(args.device)

    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    max_retrieved = int(gnn_data["retrieved_node_ids"].size(1))
    train_seq_len = int(gnn_data["token_ids"].size(1))
    max_seq_len = min(args.max_seq_len, train_seq_len)

    return model, gnn_data, cfg, block_size, max_retrieved, max_seq_len


def build_gnn_tokenizer(gnn_data):
    tokenizer_name = gnn_data.get("tokenizer_name_or_path", "meta-llama/Meta-Llama-3-8B")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    if tokenizer.pad_token_id is None:
        pad_token_id = gnn_data.get("pad_token_id")
        if pad_token_id is not None:
            tokenizer.pad_token_id = int(pad_token_id)
        elif tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError("GNN tokenizer must define pad_token_id or eos_token_id")

    return tokenizer


def get_model_vocab_size(model):
    if hasattr(model, "token_emb"):
        return model.token_emb.num_embeddings
    if hasattr(model, "token_embedding"):
        return model.token_embedding.num_embeddings
    return None


def predict_prompt_blocks(
    prompt_text,
    sorted_passage,
    model,
    tokenizer,
    node_features,
    edge_index,
    edge_weight,
    passage_text_to_hash,
    node_name_to_idx,
    max_retrieved,
    max_seq_len,
    block_size,
    device,
):
    token_ids, attn, prompt_tokens = encode_prompt_text(
        prompt_text,
        tokenizer,
        max_seq_len,
    )

    passage_hash_ids, r_nodes = derive_retrieved_nodes(
        sorted_passage,
        passage_text_to_hash,
        node_name_to_idx,
        max_retrieved,
    )

    if min(r_nodes) < 0 or max(r_nodes) >= node_features.size(0):
        raise ValueError(
            f"Invalid retrieved node ids: {r_nodes}, "
            f"num_nodes={node_features.size(0)}. "
            "This means sorted_passage does not match passage_embedding.parquet text."
        )

    vocab_size = get_model_vocab_size(model)
    if vocab_size is not None:
        max_token_id = int(token_ids.max().item())
        if max_token_id >= vocab_size:
            raise ValueError(
                f"Invalid token id: max_token_id={max_token_id}, "
                f"vocab_size={vocab_size}. Use the GNN training tokenizer."
            )

    retrieved = torch.tensor(r_nodes, dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(
            token_ids.to(device).unsqueeze(0),
            attn.to(device).unsqueeze(0),
            node_features,
            edge_index,
            edge_weight,
            retrieved.unsqueeze(0),
        ).squeeze(0)

    return {
        "passage_hash_ids": passage_hash_ids,
        "retrieved_node_ids": r_nodes,
        "logits": logits.detach().cpu(),
        "num_valid_tokens": int(attn.sum().item()),
        "tokens": prompt_tokens,
        "block_size": block_size,
    }
    


def save_kv_monitor_results(dataset_name: str, run_ts: float):
    tag = f"{dataset_name}_{int(run_ts)}"

    print("\n" + "=" * 60)
    print(f"  KV CACHE MONITOR RESULTS  [{tag}]")
    print("=" * 60)
    kv_report(top_n=15)

    df = kv_to_dataframe()
    if df.empty:
        print("kv_cache_monitor: no events recorded")
        return

    csv_path = f"kv_cache_events_{tag}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Raw events saved -> {csv_path} ({len(df):,} rows)")

    fig_path = f"kv_cache_analysis_{tag}.png"
    try:
        kv_visualize(save_path=fig_path, top_n_blocks=20, time_bucket_s=5.0)
        print(f"Analysis figure -> {fig_path}")
    except Exception as e:
        print(f"Visualisation failed: {e}")

def gnn_pred_to_block_tiers(pred):
    logits = pred["logits"]
    block_size = pred["block_size"]

    probs = torch.softmax(logits, dim=-1)
    confidence, levels = probs.max(dim=-1)

    tiers = {}

    for block_idx, level in enumerate(levels.tolist()):
        if level == 2:
            tier = "gpu"
        elif level == 1:
            tier = "cpu"
        else:
            tier = "disk"

        tiers[block_idx] = {
            "tier": tier,
            "importance_level": int(level),
            "confidence": float(confidence[block_idx]),
            "token_start": block_idx * block_size,
            "token_end": (block_idx + 1) * block_size,
        }

    return tiers

def main():
    args = parse_arguments()

    if not args.question and not args.questions_json:
        raise ValueError("provide either --question or --questions_json")
    
    _sc_start_monitor()
    setup_environment_variables()

    questions = load_questions(args)
    if args.max_questions is not None:
        questions = questions[:args.max_questions]
        print(f"Using first {len(questions)} questions", flush=True)

    vllm_tokenizer = AutoTokenizer.from_pretrained(args.llm_model, use_fast=True)
    if vllm_tokenizer.pad_token_id is None and vllm_tokenizer.eos_token_id is not None:
        vllm_tokenizer.pad_token = vllm_tokenizer.eos_token

    embedding_model = load_embedding_model(args.embedding_model)

    retrieval_results = dense_retrieve_from_import(
        import_root=Path(args.linearrag_import_dir),
        dataset_name=args.dataset_name,
        questions=questions,
        embedding_model=embedding_model,
        retrieval_top_k=args.retrieval_top_k,
    )

    llm_inputs, prompt_records = build_vllm_prompts_from_retrieval_results(
        retrieval_results,
        vllm_tokenizer,
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
        warm_source_indices = list(range(num_requests - 1, -1, -1))
    else:
        warm_source_indices = list(range(num_requests))

    warm_llm_inputs = [
        llm_inputs[source_index]
        for source_index in warm_source_indices
    ]
    warm_prompt_records = [
        prompt_records[source_index]
        for source_index in warm_source_indices
    ]

    print(
        "Request ordering configuration: "
        f"request_order={args.request_order}, "
        f"prefix_sort_depth={args.prefix_sort_depth}, "
        f"request_order_seed={args.request_order_seed}, "
        f"warm_order={args.warm_order}, "
        f"requests={num_requests}",
        flush=True,
    )
    print("retrieval and vLLM prompt construction finished", flush=True)

    node_features, edge_index, edge_weight, node_name_to_idx, passage_text_to_hash = (
        load_graph_from_linearrag_import(
            Path(args.linearrag_import_dir),
            args.dataset_name,
            args.device,
        )
    )

    gnn_model, gnn_data, gnn_cfg, block_size, max_retrieved, max_seq_len = (
        build_gnn_predictor(args, node_features.size(1))
    )

    gnn_tokenizer = build_gnn_tokenizer(gnn_data)

    gnn_predictions = []
    request_importance = []
    for rec in prompt_records:
        pred = predict_prompt_blocks(
            prompt_text=rec["gnn_prompt"],
            sorted_passage=rec["sorted_passage"],
            model=gnn_model,
            tokenizer=gnn_tokenizer,
            node_features=node_features,
            edge_index=edge_index,
            edge_weight=edge_weight,
            passage_text_to_hash=passage_text_to_hash,
            node_name_to_idx=node_name_to_idx,
            max_retrieved=max_retrieved,
            max_seq_len=max_seq_len,
            block_size=block_size,
            device=args.device,
        )
        
        print("gnn_prediction has finished", flush=True)
        gnn_predictions.append(pred)

    tiers_by_source_index = [
        gnn_pred_to_block_tiers(pred)
        for pred in gnn_predictions
    ]

    importance_sidecar = {}

    # Cold runtime request IDs are 0 through N-1.
    for source_index, tiers in enumerate(tiers_by_source_index):
        importance_sidecar[str(source_index)] = tiers

    # Warm runtime request IDs are N through 2N-1. Map each runtime position
    # back to the source prompt whose GNN prediction belongs at that position.
    for warm_position, source_index in enumerate(warm_source_indices):
        runtime_request_id = num_requests + warm_position
        importance_sidecar[str(runtime_request_id)] = (
            tiers_by_source_index[source_index]
        )
    importance_path = os.environ.get(
        "VLLM_KV_IMPORTANCE_TIERS",
        f"/tmp/kv_importance_tiers_{os.environ.get('USER', 'user')}_{os.environ.get('SLURM_JOB_ID', 'local')}.json",
    )

    with open(importance_path, "w", encoding="utf-8") as f:
        json.dump(importance_sidecar, f, indent=2)

    print(f"all gnn predictions saved -> {importance_path}", flush=True)

    sampling_params = SamplingParams(
        temperature=0,
        top_p=0.95,
        min_tokens=128,
        max_tokens=512,
    )

    # kv_install()

    # import lmcache_hit_hook as hook

    # hook.reset_logs()
    # hook.install()

    lmcache_connector = "LMCacheConnectorV1"
    os.environ["VLLM_KV_IMPORTANCE_TIERS"] = importance_path
    os.environ["GNN_KV_BLOCK_SIZE"] = str(GNN_KV_BLOCK_SIZE)
    
    os.environ["VLLM_KV_IMPORTANCE_ENABLE"] = os.environ.get(
        "VLLM_KV_IMPORTANCE_ENABLE", "0"
    )
   
    # llm_inputs = [llm_inputs[0]] * 1000

    with build_llm_with_lmcache(lmcache_connector, args.llm_model) as llm:
        # kv_reset()

        print(
            "Bounded submission configuration: "
            f"total_requests={len(llm_inputs)}, "
            f"submission_batch_size={args.submission_batch_size}, "
            f"max_num_seqs={os.environ.get('VLLM_MAX_NUM_SEQS', '4')}, "
            f"request_order={args.request_order}, "
            f"warm_order={args.warm_order}",
            flush=True,
        )

        print("Cold run starting...", flush=True)
        cold_start = time.time()

        cold_outputs = generate_in_submission_batches(
            llm=llm,
            llm_inputs=llm_inputs,
            sampling_params=sampling_params,
            submission_batch_size=args.submission_batch_size,
            phase_name="cold",
        )

        cold_time_taken = time.time() - cold_start
        print(
            f"first generation took {cold_time_taken:.2f} seconds.",
            flush=True,
        )
        cold_request_metrics = None
        if _sc_io_trace_enabled():
            cold_request_metrics, _ = _sc_emit_phase_metrics(
                "cold", cold_outputs, cold_time_taken
            )

        if _sc_io_trace_enabled():
            print(
                "[SC_IO_PHASE_BOUNDARY] "
                f"phase=cold_done wall={time.time():.6f} "
                f"mono={time.monotonic():.6f} proc_io={_sc_io_proc_snapshot()}",
                flush=True,
            )

        print("Warm run starting...", flush=True)
        if _sc_io_trace_enabled():
            print(
                "[SC_IO_PHASE_BOUNDARY] "
                f"phase=warm_start wall={time.time():.6f} "
                f"mono={time.monotonic():.6f} proc_io={_sc_io_proc_snapshot()}",
                flush=True,
            )
        warm_start = time.time()

        warm_outputs_in_execution_order = generate_in_submission_batches(
            llm=llm,
            llm_inputs=warm_llm_inputs,
            sampling_params=sampling_params,
            submission_batch_size=args.submission_batch_size,
            phase_name="warm",
        )

        warm_time_taken = time.time() - warm_start
        print(
            f"Second generation took {warm_time_taken:.2f} seconds.",
            flush=True,
        )

        warm_outputs = [None] * num_requests
        for warm_position, source_index in enumerate(warm_source_indices):
            warm_outputs[source_index] = (
                warm_outputs_in_execution_order[warm_position]
            )

        if any(output is None for output in warm_outputs):
            raise RuntimeError(
                "Failed to restore warm outputs to cold/source request order"
            )

        if _sc_io_trace_enabled():
            warm_request_metrics, _ = _sc_emit_phase_metrics(
                "warm", warm_outputs, warm_time_taken
            )
            if cold_request_metrics is None:
                raise RuntimeError(
                    "Cold request metrics missing while IO tracing is enabled"
                )
            _sc_emit_output_consistency(
                cold_request_metrics, warm_request_metrics
            )
            print("[SC_DRIVER_METRICS_DONE]", flush=True)

        outputs = warm_outputs
        # start = warm_start

        # save_kv_monitor_results(args.dataset_name, start)

        for output in outputs:
            generated_text = output.outputs[0].text
            print(f"Output: {generated_text!r}")


        # summary_path = hook.dump_summary()
        # print(f"LMCache hook summary written to: {summary_path}")

        # with open(summary_path, "r", encoding="utf-8") as f:
        #     print(json.dumps(json.load(f), indent=2))


if __name__ == "__main__":
    main()
