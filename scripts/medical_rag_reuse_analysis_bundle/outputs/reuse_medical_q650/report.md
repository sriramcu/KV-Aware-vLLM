# Offline passage and prefix-KV reuse profile

**Interpretation:** passage-level overlap is not chunk-prefix reuse. Warm/cold simulations below are causal, use exact token-sequence equality fingerprints, and assume deterministic persistence of all full chunks admitted to the simulated cache. They do not simulate CPU/disk GET timing, vLLM 16-token VPC, cache-tier weights, or answer quality.

## Inputs

- Requests: 650; top-k: 5
- Request order: source; cold seed: 0
- Tokenizer: meta-llama/Llama-3.3-70B-Instruct; full chunk size: 512
- Retrieval provenance: /mnt/shared/gpfs/home/sriramc2/runs/kvaware/h100_shortq_gnn_dynamic_vpc_q650_disk_u025_mns16_vote_threshold_8_2_dropplurality_fixed128_rand0_rand1_29265/results/prompt_records.jsonl

## Ordered passage-prefix repetition

| First k passages | Share with any question | Share with earlier request | Distinct ordered prefixes |
|---|---:|---:|---:|
| 1 | 94.62% | 80.15% | 129 |
| 2 | 55.69% | 39.23% | 395 |
| 3 | 32.15% | 19.23% | 525 |
| 4 | 15.08% | 8.00% | 598 |
| 5 | 7.54% | 4.00% | 624 |

Distinct passage texts: **195**

## Exact-token chunk conversion

- Full chunk occurrences / unique hashes: **6,829 / 4,855**
- Repeated full chunk occurrence share (global, retrospective): **28.91%**
- Causal usable contiguous-prefix share (cold, unlimited cache): **28.91%**
- Physically present but stranded full-chunk hashes: **0**
- Sampled requests with mixed-passage 512-token chunks: **100.00%**
- First passage shared earlier but zero full-chunk hits: **0**
- Recorded prompts failing reconstruction audit: **0 / 650**
- Padding estimates are ideal token-count overheads only, not a live-text padding experiment.

## Causal simulation (unlimited or fixed LRU)

| Strategy | Capacity (full chunks) | Phase | Useful hit chunks | Token reuse | Reordered | Evictions cumulative |
|---|---:|---|---:|---:|---:|---:|
| original | unlimited | cold | 1,974 | 27.58% | 0.00% | 0 |
| original | unlimited | warm | 6,829 | 95.40% | 0.00% | 0 |
| canonical | unlimited | cold | 3,046 | 42.55% | 99.54% | 0 |
| canonical | unlimited | warm | 6,829 | 95.40% | 99.54% | 0 |
| greedy | unlimited | cold | 3,811 | 53.24% | 77.85% | 0 |
| greedy | unlimited | warm | 6,829 | 95.40% | 77.85% | 0 |
| greedy_keep_top1 | unlimited | cold | 2,941 | 41.09% | 42.31% | 0 |
| greedy_keep_top1 | unlimited | warm | 6,829 | 95.40% | 42.31% | 0 |

## Validity boundaries

- `shared_anywhere` includes future requests; `shared_earlier` respects the specified replay order.
- `greedy` optimizes each request over currently simulated available hashes, not a globally optimal sequence.
- Both cold and warm phases update the same simulated cache; no store latency or stage overlap is modeled.
- Token fingerprints are prefix-dependent. Hash serialization differs from LMCache ObjectKey; verify live exact prompt token IDs on the cluster.
- Whole-passage reordering changes evidence order and may affect quality; measure answers separately.
- The raw `dataset/medical/questions.json` does **not** contain sorted passages. Matching the historical 90.4%/49.6% requires the original retrieval method, top-k, subset, and sampling order.
