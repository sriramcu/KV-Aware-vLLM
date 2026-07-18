from dataclasses import asdict

from vllm import EngineArgs

print("import EngineArgs OK")

args = EngineArgs(
    model="HuggingFaceTB/SmolLM2-135M-Instruct",
    dtype="bfloat16",
    max_model_len=2048,
    gpu_memory_utilization=0.40,
    max_num_seqs=4,
    enable_prefix_caching=True,
    enable_chunked_prefill=True,
    # Important smoke-test simplifications:
    tensor_parallel_size=1,
    quantization=None,
    distributed_executor_backend=None,
)

print("EngineArgs constructed")
print(asdict(args))

cfg = args.create_engine_config()
print("create_engine_config OK")
print("device:", cfg.device_config.device)
print("model:", cfg.model_config.model)