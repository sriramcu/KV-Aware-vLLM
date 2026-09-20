#!/usr/bin/env python3
"""Extract prompt-token supervision from a causal LM.

For each sample, this script builds:
  retrieved passages + question + answer
and computes prompt-token importance scores using either:
  - last-layer attention from answer tokens back to prompt tokens
  - gradient-based logit attribution for answer token probabilities
The resulting normalized scores can be consumed by the GNN data builders.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_attention_supervision import normalize_attention_scores


def build_prompt(passages: List[str], question: str) -> str:
    parts = [p for p in passages if p]
    parts.append(question)
    return "\n".join(parts)


def token_positions_from_offsets(offsets, start_char: int, end_char: int) -> List[int]:
    positions = []
    for i, (s, e) in enumerate(offsets):
        if e <= start_char or s >= end_char:
            continue
        if e > s:
            positions.append(i)
    return positions


def compute_attention_scores(model, enc, prompt_positions, prompt_len, answer_positions):
    with torch.no_grad():
        outputs = model(**enc, output_attentions=True)
    if not outputs.attentions:
        raise RuntimeError(
            "Model did not return attentions. "
            "Try a transformers/model combo that supports "
            "output_attentions=True with eager attention."
        )
    last_attn = outputs.attentions[-1][0]  # [heads, seq, seq]
    if not prompt_positions:
        return []
    if not answer_positions:
        scores = last_attn[:, prompt_len - 1, prompt_positions].mean(dim=0).tolist()
        return normalize_attention_scores(scores)
    attn_slice = last_attn[:, answer_positions][:, :, prompt_positions]
    scores = attn_slice.mean(dim=(0, 1)).tolist()
    return normalize_attention_scores(scores)


def compute_logit_grad_scores(model, enc, prompt_positions, prompt_len, answer_positions):
    if not prompt_positions:
        return []
    if prompt_len < 1:
        return []

    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask")
    embed_layer = model.get_input_embeddings()
    inputs_embeds = embed_layer(input_ids).detach().clone()
    inputs_embeds.requires_grad_(True)

    model.zero_grad(set_to_none=True)
    outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    logits = outputs.logits  # [1, T, V]

    total_len = int(input_ids.size(1))
    if total_len < 2:
        return [0.0 for _ in prompt_positions]

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # [1, T-1]

    target_pred_positions = []
    for pos in answer_positions:
        pred_pos = pos - 1
        if 0 <= pred_pos < token_log_probs.size(1):
            target_pred_positions.append(pred_pos)
    if not target_pred_positions and prompt_len - 1 < token_log_probs.size(1):
        target_pred_positions = [prompt_len - 1]
    if not target_pred_positions:
        return [0.0 for _ in prompt_positions]

    objective = token_log_probs[0, target_pred_positions].sum()
    objective.backward()

    grads = inputs_embeds.grad[0]  # [T, H]
    embeds = inputs_embeds.detach()[0]
    saliency = (grads * embeds).abs().sum(dim=-1)
    scores = saliency[prompt_positions].tolist()
    return normalize_attention_scores(scores)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="HF model path/id that supports output_attentions=True")
    p.add_argument("--predictions-json", required=True)
    p.add_argument("--use-gold-answer", action="store_true",
                   help="Use gold_answer instead of pred_answer when available")
    p.add_argument("--method",
                   choices=["attention", "logit_grad"],
                   default="attention",
                   help="Supervision method for prompt token scores")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    with open(args.predictions_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if args.max_samples is not None:
        data = data[:args.max_samples]

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {}
    if args.method == "attention":
        model_kwargs["attn_implementation"] = "eager"
    if args.device.startswith("cuda"):
        model_kwargs["dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.to(args.device)
    model.eval()

    out: List[Dict] = []
    for item in data:
        question = item["question"]
        passages = item.get("sorted_passage", [])
        answer = item.get("gold_answer" if args.use_gold_answer else "pred_answer")
        if not answer:
            answer = item.get("gold_answer", "") or item.get("pred_answer", "")

        prompt = build_prompt(passages, question)
        full_text = f"{prompt}\nAnswer: {answer}".rstrip()

        enc = tokenizer(full_text,
                        return_tensors="pt",
                        return_offsets_mapping=True,
                        truncation=True)
        offsets = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(args.device) for k, v in enc.items()}
        prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True)["input_ids"][0]
        prompt_len = int(prompt_ids.numel())
        total_len = int(enc["input_ids"].size(1))
        answer_positions = list(range(prompt_len, total_len))

        prompt_positions = token_positions_from_offsets(offsets, 0, len(prompt))

        if not prompt_positions:
            prompt_tokens = []
            prompt_scores = []
        else:
            prompt_tokens = tokenizer.convert_ids_to_tokens(
                enc["input_ids"][0, prompt_positions].tolist())
            if args.method == "attention":
                prompt_scores = compute_attention_scores(
                    model=model,
                    enc=enc,
                    prompt_positions=prompt_positions,
                    prompt_len=prompt_len,
                    answer_positions=answer_positions,
                )
            else:
                prompt_scores = compute_logit_grad_scores(
                    model=model,
                    enc=enc,
                    prompt_positions=prompt_positions,
                    prompt_len=prompt_len,
                    answer_positions=answer_positions,
                )

        out.append({
            "question": question,
            "prompt_tokens": prompt_tokens,
            "prompt_token_scores": prompt_scores,
            "score_method": args.method,
        })

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
