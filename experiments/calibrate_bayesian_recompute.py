"""One-off model-backed prefill prior; no test-set prompts.

Uses actual serving model and scheduler engine-step wall intervals. This is
an effective congestion-aware recompute reference per 512 prompt tokens,
not exclusive GPU compute/marginal end-to-end savings. Calibration is
purposefully small; do not use final Q650 requests as calibration inputs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import tempfile


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--prior-in', type=Path, required=True)
    p.add_argument('--prior-out', type=Path, required=True)
    p.add_argument('--trace-dir', type=Path, required=True)
    p.add_argument('--model', default='meta-llama/Llama-3.3-70B-Instruct')
    p.add_argument('--max-model-len', type=int, default=8000)
    p.add_argument('--gpu-utilization', type=float, default=0.65)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--min-samples', type=int, default=3)
    args = p.parse_args()
    if args.rounds < 2 or args.min_samples < 2:
        p.error('need >=2 rounds and >=2 valid samples')
    # Device timing is not required: engine events need neither CUDA
    # synchronization nor cross-rank clock alignment.
    os.environ['VLLM_PHASE_TRACE'] = '1'
    os.environ['VLLM_PHASE_TRACE_DEVICE_TIMING'] = '0'
    os.environ['VLLM_PHASE_TRACE_DIR'] = str(args.trace_dir)
    args.trace_dir.mkdir(parents=True, exist_ok=True)
    if any(args.trace_dir.glob('phase_engine.pid*.jsonl')):
        raise RuntimeError('Refusing to reuse nonempty engine-trace directory')

    from vllm import LLM, SamplingParams
    from Hierarchical_KV.shortq_placement.bayesian_mapper import PhaseRecomputeCostReader

    print('[BAYES_CALIB_RECOMPUTE] loading actual serving model', flush=True)
    llm = LLM(model=args.model, tensor_parallel_size=2, quantization='fp8',
              gpu_memory_utilization=args.gpu_utilization,
              max_model_len=args.max_model_len, max_num_seqs=16,
              enable_prefix_caching=False)
    tokenizer = llm.get_tokenizer()
    tokens = tokenizer.encode('Medical background and retrieved evidence. ',
                              add_special_tokens=False)
    if len(tokens) < 2:
        raise RuntimeError('calibration tokenization failed')
    bos = getattr(tokenizer, 'bos_token_id', None)
    lengths = (1024, 2048, 3072, 4096)
    prompts = []
    for size in lengths:
        content = (tokens * ((size + len(tokens) - 1) // len(tokens)))[:size]
        prompts.append({'prompt_token_ids': ([bos] if bos is not None else []) + content})
    sampling = SamplingParams(temperature=0.0, min_tokens=1, max_tokens=1)
    print('[BAYES_CALIB_RECOMPUTE] warmup', flush=True)
    llm.generate(prompts[:2], sampling, use_tqdm=False)
    # Discard warm-up trace without perturbing mapper or GPU.
    class Accumulator:
        def __init__(self):
            self.samples: list[tuple[float, float]] = []
        def observe_cost(self, tier, milliseconds_per_chunk, count=1):
            if tier != 'recompute' or not math.isfinite(milliseconds_per_chunk):
                raise ValueError('invalid recompute sample')
            self.samples.append((float(milliseconds_per_chunk), float(count)))

    reader = PhaseRecomputeCostReader(args.trace_dir)
    sink = Accumulator()
    reader.collect(sink)
    for i in range(args.rounds):
        print(f'[BAYES_CALIB_RECOMPUTE] probe round={i+1}/{args.rounds}', flush=True)
        llm.generate(prompts, sampling, use_tqdm=False)
    stats = reader.collect(sink)
    if len(sink.samples) < args.min_samples:
        raise RuntimeError(f'Only {len(sink.samples)} prefill samples; need {args.min_samples}. '
                           f'Diagnostics: {stats}')
    total_chunks = sum(w for _, w in sink.samples)
    prior_ms = sum(v * w for v, w in sink.samples) / total_chunks
    if not math.isfinite(prior_ms) or prior_ms <= 0:
        raise RuntimeError('Nonpositive or nonfinite measured recompute cost')
    prior = json.loads(args.prior_in.read_text(encoding='utf-8'))
    prior['cost_prior_ms']['recompute'] = round(prior_ms, 5)
    prior['comment'] = ('Score histogram unchanged; CPU/disk standalone transfer calibration; '
                        'recompute reference is a short actual-model prefill probe.')
    meta = prior.setdefault('cost_calibration', {})
    meta.update({
        'recompute_prior_measured': True,
        'recompute_method': 'actual-model prefill engine completion-throughput proxy (async) or executor interval (sync); chunk-equivalent weighted mean',
        'recompute_model': args.model,
        'recompute_samples': len(sink.samples),
        'recompute_async_steps': stats['recompute_async_steps'],
        'recompute_sync_steps': stats['recompute_sync_steps'],
        'recompute_chunk_equivalents': total_chunks,
        'recompute_step_median_ms_per_chunk': statistics.median(v for v, _ in sink.samples),
        'recompute_step_min_ms_per_chunk': min(v for v, _ in sink.samples),
        'recompute_step_max_ms_per_chunk': max(v for v, _ in sink.samples),
        'recompute_probe_prompt_lengths': list(lengths),
        'recompute_probe_rounds': args.rounds,
        'recompute_warning': 'Async samples are consecutive completion-throughput intervals, not individual forward latency; sync samples include executor waiting. Neither is marginal E2E; short calibration may not recreate serving congestion.',
    })
    args.prior_out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=args.prior_out.parent,
                                     prefix='.bayes_prior_', delete=False) as f:
        tmp = Path(f.name)
        json.dump(prior, f, sort_keys=True, indent=2)
        f.write('\n')
    os.replace(tmp, args.prior_out)
    print(f'[BAYES_CALIB_RECOMPUTE] prior={prior_ms:.5f}ms samples={len(sink.samples)} '
          f'chunks={total_chunks:.2f} async={stats["recompute_async_steps"]} '
          f'sync={stats["recompute_sync_steps"]} written={args.prior_out}', flush=True)


if __name__ == '__main__':
    main()
