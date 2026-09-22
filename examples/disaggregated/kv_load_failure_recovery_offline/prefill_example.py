# SPDX-License-Identifier: Apache-2.0
import argparse

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig


def read_prompts():
    context = "Hi " * 1000
    context2 = "Hey " * 500
    return [
        context + "Hello, my name is",
        context + "The capital of France is",
        context2 + "Your name is",
        context2 + "The capital of China is",
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--storage", default="local_storage")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument(
        "--flash-attn-version",
        type=int,
        choices=(2, 3, 4),
        default=None,
        help="Force a FlashAttention version for diagnostic runs.",
    )
    args = parser.parse_args()

    prompts = read_prompts()
    sampling_params = SamplingParams(temperature=0, max_tokens=1)
    llm = LLM(
        model=args.model,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        attention_config=(
            {"flash_attn_version": args.flash_attn_version}
            if args.flash_attn_version is not None
            else None
        ),
        kv_transfer_config=KVTransferConfig(
            kv_connector="ExampleConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"shared_storage_path": args.storage},
        ),
    )
    outputs = llm.generate(prompts, sampling_params)
    new_prompts = [o.prompt + o.outputs[0].text for o in outputs]
    with open("prefill_output.txt", "w", encoding="utf-8") as f:
        for prompt in new_prompts:
            f.write(prompt + "\n")
    print(f"Saved {len(new_prompts)} prompts to prefill_output.txt")


if __name__ == "__main__":
    main()
