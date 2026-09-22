# SPDX-License-Identifier: Apache-2.0
import argparse
import json

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig


def read_prompts():
    with open("prefill_output.txt", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--storage", default="local_storage")
    parser.add_argument("--simulate-failure", action="store_true")
    parser.add_argument("--async-load", action="store_true")
    parser.add_argument("--async-scheduling", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--no-connector", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument(
        "--flash-attn-version",
        type=int,
        choices=(2, 3, 4),
        default=None,
        help="Force a FlashAttention version for diagnostic runs.",
    )
    args = parser.parse_args()

    if args.no_connector and args.simulate_failure:
        raise SystemExit("--no-connector and --simulate-failure are mutually exclusive")

    ktc = None
    if args.simulate_failure:
        ktc = KVTransferConfig(
            kv_connector="LoadRecoveryExampleConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "shared_storage_path": args.storage,
                "async_load": args.async_load,
            },
            kv_connector_module_path="load_recovery_example_connector",
            kv_load_failure_policy="recompute",
        )
    elif not args.no_connector:
        ktc = KVTransferConfig(
            kv_connector="ExampleConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"shared_storage_path": args.storage},
        )

    llm = LLM(
        model=args.model,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        attention_config=(
            {"flash_attn_version": args.flash_attn_version}
            if args.flash_attn_version is not None
            else None
        ),
        max_num_batched_tokens=64,
        max_num_seqs=16,
        async_scheduling=args.async_scheduling,
        kv_transfer_config=ktc,
    )
    outputs = llm.generate(
        read_prompts(), SamplingParams(temperature=0, max_tokens=10)
    )
    records = []
    for output in outputs:
        generated = output.outputs[0]
        records.append(
            {
                "prompt": output.prompt,
                "token_ids": list(generated.token_ids),
                "text": generated.text,
            }
        )
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(json.dumps({"output": args.output, "token_ids": [r["token_ids"] for r in records]}))


if __name__ == "__main__":
    main()
