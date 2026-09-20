# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import contextlib
import json
import os
import re
import sys
import time
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
def passage_prefix_signature(sorted_passage, n=2):
    prefix = "\n".join(sorted_passage[:n])
    return hashlib.sha1(prefix.encode("utf-8")).hexdigest()


def reorder_by_shared_passage_prefix(llm_inputs, prompt_records, n=2):
    pairs = list(zip(llm_inputs, prompt_records))

    pairs.sort(
        key=lambda x: (
            passage_prefix_signature(x[1]["sorted_passage"], n=n),
            x[1]["question"],
        )
    )

    new_llm_inputs = [p[0] for p in pairs]
    new_prompt_records = [p[1] for p in pairs]

    return new_llm_inputs, new_prompt_records


HKV_ROOT = Path("/mnt/shared/gpfs/home/seliny2/vllm/Hierarchical_KV").resolve()
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

from kvcache_monitor import (
    install as kv_install,
    report as kv_report,
    reset as kv_reset,
    to_dataframe as kv_to_dataframe,
)
from kvcache_visualize import visualize as kv_visualize


def setup_environment_variables():
    def write_lmcache_config(path: str):
        import textwrap

        config = textwrap.dedent("""
        chunk_size: 512
        local_cpu: true
        max_local_cpu_size: 100.0
        local_disk: "file:///scratch/seliny2_cache/vllm/"
        max_local_disk_size: 500.0
        enable_kv_events: true
        pre_caching_hash_algorithm: builtin
        """).strip()
        Path(path).write_text(config + "\n", encoding="utf-8")

    cfg_path = "./lmcache_config.yaml"
    write_lmcache_config(cfg_path)

    os.environ["LMCACHE_HOOK_ENABLE"] = "1"
    os.environ["LMCACHE_HOOK_LOG_DIR"] = "/tmp/lmcache_hit_hook"
    os.environ["LMCACHE_CONFIG_FILE"] = os.path.abspath(cfg_path)
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["PYTHONHASHSEED"] = "0"
    os.environ["VLLM_DISTRIBUTED_BACKEND"] = "nccl"
    os.environ["VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM"] = "1"
    os.environ["VLLM_RPC_TIMEOUT"] = "1200000"
    os.environ["LMCACHE_CPU_READER_THREADS"] = "4"
    os.environ["VLLM_ENGINE_ITERATION_TIMEOUT_S"] = "1200"
    os.environ["VLLM_SAMPLED_TOKEN_ID_BUFFER_SIZE"] = "10"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["LMCACHE_ENABLE_ASYNC_LOADING"] = "True"
    os.environ["LMCACHE_USE_EXPERIMENTAL"] = "True"
    os.environ["LMCACHE_CHUNK_SIZE"] = "512"
    os.environ["LMCACHE_LOCAL_CPU"] = "True"
    os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "100"

    os.makedirs("/scratch/seliny2_cache/vllm", exist_ok=True)
    os.environ["LMCACHE_LOCAL_DISK"] = "file:///scratch/seliny2_cache/vllm/"
    os.environ["LMCACHE_INTERNAL_API_SERVER_ENABLED"] = "True"
    os.environ["LMCACHE_MAX_LOCAL_DISK_SIZE"] = "500"
    os.environ["DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER"] = "False"
    os.environ["PROMETHEUS_MULTIPROC_DIR"] = "/mnt/shared/gpfs/home/seliny2/.cache/vllm"


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

    return parser.parse_args()


@contextlib.contextmanager
def build_llm_with_lmcache(lmcache_connector: str, model: str):
    ktc = KVTransferConfig(
        kv_connector=lmcache_connector,
        kv_role="kv_both",
    )

    from vllm.config import KVEventsConfig

    kv_events_config = KVEventsConfig(enable_kv_cache_events=True)

    llm_args = EngineArgs(
        model=model,
        kv_transfer_config=ktc,
        kv_events_config=kv_events_config,
        max_model_len=8000,
        gpu_memory_utilization=0.65,
        dtype="bfloat16",
        max_num_seqs=20,
        tensor_parallel_size=2,
        enforce_eager=False,
        enable_chunked_prefill=True,
        disable_log_stats=False,
        distributed_executor_backend="mp",
        quantization="fp8",
        enable_prefix_caching=True,
    )

    llm = LLM(**asdict(llm_args))

    try:
        yield llm
    finally:
        LMCacheEngineBuilder.destroy(ENGINE_NAME)


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
    
    setup_environment_variables()

    questions = load_questions(args)

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
    llm_inputs, prompt_records = reorder_by_shared_passage_prefix(
        llm_inputs,
        prompt_records,
        n=2,
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

    importance_sidecar = {
        str(i): gnn_pred_to_block_tiers(pred)
        for i, pred in enumerate(gnn_predictions)
    } 
    with open("/tmp/kv_importance_tiers.json", "w", encoding="utf-8") as f:
        json.dump(importance_sidecar, f, indent=2)
    print("all gnn predictions saved -> gnn_prompt_predictions.pt", flush=True)

    sampling_params = SamplingParams(
        temperature=0,
        top_p=0.95,
        min_tokens=128,
        max_tokens=512,
    )

    kv_install()

    import lmcache_hit_hook as hook

    hook.reset_logs()
    hook.install()

    lmcache_connector = "LMCacheConnectorV1"
    os.environ["VLLM_KV_IMPORTANCE_TIERS"] = "/tmp/kv_importance_tiers.json"
    os.environ["GNN_KV_BLOCK_SIZE"] = str(16)
    
    os.environ["VLLM_KV_IMPORTANCE_ENABLE"] = "0"
   
    # llm_inputs = [llm_inputs[0]] * 1000

    with build_llm_with_lmcache(lmcache_connector, args.llm_model) as llm:
        kv_reset()
        
        print(f"Cold run starting...")
        start = time.time()
        outputs = llm.generate(llm_inputs, sampling_params)
        time_taken = time.time() - start
        print(f"first generation took {time_taken:.2f} seconds.")
        
        print(f"Warm run starting...")
        start = time.time()
        outputs = llm.generate(llm_inputs, sampling_params)
        time_taken = time.time() - start
        print(f"Second generation took {time_taken:.2f} seconds.")


        save_kv_monitor_results(args.dataset_name, start)

        for output in outputs:
            generated_text = output.outputs[0].text
            print(f"Output: {generated_text!r}")


        summary_path = hook.dump_summary()
        print(f"LMCache hook summary written to: {summary_path}")

        with open(summary_path, "r", encoding="utf-8") as f:
            print(json.dumps(json.load(f), indent=2))


if __name__ == "__main__":
    main()
