from dataclasses import asdict
from vllm import LLM, EngineArgs
from vllm.config import KVTransferConfig

print("import OK")

kv_transfer_config = KVTransferConfig(
    kv_connector="LMCacheConnectorV1",
    kv_role="kv_both",
    kv_buffer_device="cuda",
    kv_buffer_size=1e9,
)

args = EngineArgs(
    model="HuggingFaceTB/SmolLM2-135M-Instruct",
    dtype="bfloat16",
    max_model_len=2048,
    gpu_memory_utilization=0.40,
    max_num_seqs=4,
    enable_prefix_caching=True,
    enable_chunked_prefill=True,
    tensor_parallel_size=1,
    quantization=None,
    distributed_executor_backend=None,
    kv_transfer_config=kv_transfer_config,
)

print("EngineArgs constructed")
print(asdict(args))

llm = LLM(**asdict(args))
print("LLM constructed OK")