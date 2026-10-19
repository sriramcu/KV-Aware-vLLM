# Offline passage and prefix-KV reuse profile

**Interpretation:** passage-level overlap is not chunk-prefix reuse. Warm/cold simulations below are causal, use exact token-sequence equality fingerprints, and assume deterministic persistence of all full chunks admitted to the simulated cache. They do not simulate CPU/disk GET timing, vLLM 16-token VPC, cache-tier weights, or answer quality.

## Inputs

- Requests: N/A; top-k: 5
- Request order: seeded_shuffle; cold seed: 0
- Tokenizer: not available; full chunk size: 512
- Retrieval provenance: None


## Validity boundaries

- `shared_anywhere` includes future requests; `shared_earlier` respects the specified replay order.
- `greedy` optimizes each request over currently simulated available hashes, not a globally optimal sequence.
- Both cold and warm phases update the same simulated cache; no store latency or stage overlap is modeled.
- Token fingerprints are prefix-dependent. Hash serialization differs from LMCache ObjectKey; verify live exact prompt token IDs on the cluster.
- Whole-passage reordering changes evidence order and may affect quality; measure answers separately.
- The raw `dataset/medical/questions.json` does **not** contain sorted passages. Matching the historical 90.4%/49.6% requires the original retrieval method, top-k, subset, and sampling order.
