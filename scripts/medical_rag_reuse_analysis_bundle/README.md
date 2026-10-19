# Medical RAG passage-to-KV-chunk reuse profiler

Standalone analysis for the **KV-Aware-vLLM / LinearRAG medical dataset**. The script mirrors the exact prompt **text construction** in `experiments/mp_nognn_project.py`: the specified system prompt, each retrieved passage followed by `\n`, `Question: {question}\n Thought: `, and the Hugging Face chat template with `add_generation_prompt=True`.

The output is an **offline potential / causal simulation**, **not** a measured LMCache/vLLM execution. For complete 512-token chunks, it uses a cryptographic *rolling token-prefix equality fingerprint*. This preserves exact-match equivalence, but the byte-string keys are not the production LMCache `ObjectKey` hashes. Confirm tokenizer version and token IDs on the inference cluster.

## Inputs and provenance

The repository ZIP has `dataset/medical/questions.json` (**2,062** questions) and `dataset/medical/chunks.json` (**225** source chunks). **It does not contain the top-5 ranked retrieved passages per question.** Therefore those files alone cannot confirm the previously reported 90.4% / 49.6% overlap from a 250-question retrieval profile.

**Strongly recommended:** load `prompt_records.jsonl` from the actual run, which the uploaded driver writes to `RESULT_DIR/prompt_records.jsonl`. It has `sorted_passage`, `sorted_passage_scores`, `question`, and the already-rendered `llm_prompt`. The script checks reconstruction against `llm_prompt` byte-for-byte and reports any mismatches.

**Important request-order detail:** `prompt_records.jsonl` is **already saved in the cold execution order** determined by `--request_order` in the experiment driver. Analyze it with `--request-order source` to preserve that execution order. `--request-order seeded_shuffle --request-order-seed 0` is appropriate only for a **retrieval-only JSON in original question/source order**, not for driver-saved reordered prompt records.

Alternatively, reconstruct the driver's **dense-import fallback** retrieval with `--build-dense-from-import`, assuming `passage_embedding.parquet` and its exact embedding model are available. This is not the graph-search LinearRAG retrieval path, and changing the retrieval algorithm/model invalidates direct overlap comparisons.

## Dependencies

```bash
python -m pip install 'transformers>=4.46' tokenizers matplotlib
```

For optional dense-import reconstruction:

```bash
python -m pip install numpy pandas pyarrow sentence-transformers
```

You must have authorized access to the corresponding Llama tokenizer, locally or through Hugging Face. The script **does not run the LLM**.

## A. Confirm historical Q250 passage-prefix percentages

Run from the project root, using an original-order retrieval list:

```bash
python analyze_medical_reuse.py \
  --retrieval-json /path/to/retrieval_first_250.json \
  --questions-json dataset/medical/questions.json \
  --chunks-json dataset/medical/chunks.json \
  --max-questions 250 --topk 5 \
  --request-order source \
  --output-dir reuse_q250_passages
```

This first command intentionally omits `--tokenizer`, so it reports passage-level reuse **without claiming any 512-token results**. The top-k ordered-prefix table in `report.md` includes “shared anywhere” (the historical-style retrospective 90.4% / 49.6% metric) and “shared with an earlier request” (causal). These denominators differ from pairwise overlap and global individual-passage reuse.

## B. Profile your actual Q650 H100 run

```bash
python analyze_medical_reuse.py \
  --retrieval-json /path/to/results/prompt_records.jsonl \
  --dataset-root dataset --dataset medical \
  --max-questions 650 --topk 5 \
  --request-order source --warm-order random --warm-seed 1 \
  --tokenizer meta-llama/Llama-3.3-70B-Instruct \
  --chunk-size 512 --batch-size 50 \
  --strategies original canonical greedy greedy_keep_top1 \
  --capacities unlimited \
  --output-dir reuse_medical_q650
```

**Runtime note:** The exhaustive greedy strategies evaluate up to **120 full tokenizations per query per phase**. Start with a pilot (e.g., `--max-questions 50`) or omit greedy strategies for the first Q650 profiling pass. Candidate prefix-hash summaries are cached separately to avoid re-tokenizing unchanged candidates during warm replay and subsequent scenario experiments. `--candidate-key-cache` controls this memory/time tradeoff.

For a quick original-order Q650 baseline:

```bash
python analyze_medical_reuse.py \
  --retrieval-json /path/to/results/prompt_records.jsonl \
  --request-order source \
  --tokenizer meta-llama/Llama-3.3-70B-Instruct \
  --strategies original --capacities unlimited \
  --max-questions 650 --output-dir reuse_baseline
```

## C. Finite-LRU sensitivity / admission-independent eviction

```bash
python analyze_medical_reuse.py \
  --retrieval-json /path/to/results/prompt_records.jsonl \
  --request-order source \
  --tokenizer meta-llama/Llama-3.3-70B-Instruct \
  --strategies original canonical greedy \
  --capacities unlimited 205 1280 1485 \
  --output-dir reuse_lru_sensitivity
```

Capacities are **counts of 512-token chunks in a single abstract LRU**, not physical GPU+CPU+disk tier budgets. `205`, `1280`, and `1485` are illustrative sensitivity points only; they are not claimed to match dynamically available GPU VPC slots. There is no tier-load latency, PUT pacing, GNN DROP, stage overlap, or prefix-closure implementation in this study.

## D. Explicit offline future-aware benchmark (optional)

```bash
python analyze_medical_reuse.py \
  --retrieval-json /path/to/results/prompt_records.jsonl \
  --request-order source \
  --tokenizer meta-llama/Llama-3.3-70B-Instruct \
  --strategies original --capacities unlimited \
  --hindsight-oracle --hindsight-limit 50 \
  --output-dir reuse_hindsight_pilot
```

This maximizes each request's reuse against the union of the **other** requests' original-order cached hashes, including requests from the future. It is **not deployable** and not a global optimum over jointly reordered workloads. It answers a specific “what if every other original prompt was in an infinite cache?” question.

## E. Regenerate retrieval from the exact dense-import embedding matrix

```bash
python analyze_medical_reuse.py \
  --build-dense-from-import \
  --import-root Hierarchical_KV/LinearRAG/import \
  --embedding-model /path/to/original/embedding_model \
  --dataset-root dataset --dataset medical \
  --max-questions 650 --topk 5 \
  --request-order seeded_shuffle --request-order-seed 0 \
  --tokenizer meta-llama/Llama-3.3-70B-Instruct \
  --strategies original canonical \
  --output-dir reuse_reconstructed
```

This exports `reconstructed_dense_retrieval.json` for reproducibility. If the original embedding model/weights or passage embedding export differ, percentages may differ too.

## Outputs and research questions

| Output | What it measures |
|---|---|
| `report.md`, `report.json` | Overall statistics and caveats |
| `request_metrics.csv` | Per-request previous/anywhere shared ordered top-1…top-5, earlier max passage-prefix length, number of shared passages |
| `passage_frequency.csv` | Unique passage IDs, reuse popularity, rank concentration, text lengths |
| `pairwise_overlap.csv` | Pairwise Jaccard, common passages, top-prefix length, same sets/order; sampled for larger datasets |
| `question_type_breakdown.csv` | Passage reuse by medical question category |
| `baseline_chunk_requests.csv` | Prompt-token lengths, complete chunk counts, causal prefix hits, physically present later chunks |
| `passage_to_chunk_conversion.csv` | **Passage overlap surviving/failing 512-token conversion** for each request |
| `chunk_frequency.csv` | Prefix-dependent chunk identities, frequencies, chunk positions |
| `boundaries.csv` | Mixed-passage full chunks, passage boundaries, mixed question chunks, ideal token-level padding budget |
| `chunk_size_sensitivity.csv` | Reuse at 128, 256, 512, and 1024 token granularity |
| `scenario_summary.csv` | Unlimited/fixed-LRU cold+warm reuse, original, canonical, greedy, etc. |
| `scenario_requests.csv` | Per-request prefix reuse, late stranded hash matches, chosen reorder, number of permutations evaluated |
| `batch_summary.csv` | Reuse over B50 submission waves, not scheduler slots |
| `hindsight_oracle_requests.csv` | Future-aware cross-request hindsight reference (optional) |
| `graphs/*.png` | Ordered prefix sharing, policy comparison, warm-wave reuse |

The repeated-prefix table includes **Wilson 95% intervals** for the observed proportions. Those intervals express sampling uncertainty conditional on the particular dataset/profile; they do not correct for correlated questions or multiple testing.

## Interpretation and pitfalls

1. **Global passage sharing != deployable reuse.** “Shared anywhere” sees future questions. “Shared earlier” depends on the replay order; neither is a chunk-hit measurement.
2. **Exact token-prefix identity matters.** A chunk with a particular passage in a different preceding context generally has a different hash. The script rebuilds each complete candidate prompt and tokenizes it before hashing; it never permutes previously chunked KV blocks.
3. **Use only contiguous-prefix hits.** Chunks physically present beyond the first miss are counted separately as stranded and never credited as useful for the existing LMCache/vLLM prefix-restore path.
4. **Greedy is locally optimal, not globally optimal.** For up to five passages, exhaustive greedy finds the permutation with the most currently available contiguous full chunks. It does not optimize the entire future trace, real tier latencies, or answer quality.
5. **Unlimited cache is a capacity-free upper bound only under the given order/admission policy.** The finite-LRU configurations are abstract; neither models separate GPU and CPU tiers nor true-DROP admission.
6. **Padding analysis is intentionally conservative about validity.** `boundaries.csv` estimates the **ideal token count** to align passages under hypothetical token-level padding. It does not insert textual spaces, reproduce real tokenizer boundary changes, or claim reuse benefits with altered prompts.
7. **Prompt audit matters.** Inspect `token_chunks.recorded_prompt_mismatch_count` in `report.json`. It should be zero when analyzing the driver's `prompt_records.jsonl` with the correct tokenizer/chat-template version. Even then, compare token IDs against the live vLLM request pipeline before calling hashes production-identical.
8. **Response quality isn't assessed.** Passage permutations may change output quality, and this dataset-level script cannot establish EM/F1 preservation or E2E latency improvements.

## Validation

```bash
python -m unittest test_medical_reuse -v
python -m py_compile analyze_medical_reuse.py
```

`test_medical_reuse.py` exercises prefix-dependent chunk identity, greedy retrieval opportunity, passage statistics, mixed-boundary processing, hindsight estimates, and granularity sweeps with a synthetic byte-like tokenizer. These tests do not require Hugging Face model access.
