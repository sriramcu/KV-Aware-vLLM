#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

DATASETS="${DATASETS:-hotpotqa,2wikimultihop,musique,medical}"
ATTN_MODEL="${ATTN_MODEL:-}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT_DIR}/LinearRAG/results}"
IMPORT_ROOT="${IMPORT_ROOT:-${ROOT_DIR}/LinearRAG/import}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
MAX_RETRIEVED="${MAX_RETRIEVED:-5}"
VLLM_BLOCK_SIZE="${VLLM_BLOCK_SIZE:-16}"
IMPORTANCE_LEVELS="${IMPORTANCE_LEVELS:-3}"
HIDDEN_DIM="${HIDDEN_DIM:-384}"
GNN_LAYERS="${GNN_LAYERS:-2}"
TFM_LAYERS="${TFM_LAYERS:-3}"
NUM_HEADS="${NUM_HEADS:-8}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-20}"
OUTPUT_DATA="${OUTPUT_DATA:-${ROOT_DIR}/data/all_multigraph_gnn_train.pt}"
OUTPUT_MODEL="${OUTPUT_MODEL:-${ROOT_DIR}/ckpts/all_multigraph_token_ranker.pt}"
OUTPUT_METRICS="${OUTPUT_METRICS:-${ROOT_DIR}/ckpts/all_multigraph_token_ranker_metrics.json}"
USE_GOLD_ANSWER="${USE_GOLD_ANSWER:-0}"
SUPERVISION_METHOD="${SUPERVISION_METHOD:-logit_grad}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-${ATTN_MODEL}}"

usage() {
  cat <<EOF
Usage:
  ATTN_MODEL=<hf_model_or_local_path> ./run_multidataset_prompt_gnn_pipeline.sh

Optional environment variables:
  DATASETS            Comma-separated datasets. Default: ${DATASETS}
  ATTN_MODEL          Hugging Face model/path used by dump_last_layer_query_attention.py
  RESULTS_ROOT        LinearRAG results root. Default: ${RESULTS_ROOT}
  IMPORT_ROOT         LinearRAG import root. Default: ${IMPORT_ROOT}
  MAX_SEQ_LEN         Prompt token sequence length. Default: ${MAX_SEQ_LEN}
  MAX_RETRIEVED       Max retrieved passages per sample. Default: ${MAX_RETRIEVED}
  VLLM_BLOCK_SIZE     Logical vLLM KV block size in tokens. Default: ${VLLM_BLOCK_SIZE}
  IMPORTANCE_LEVELS   2 or 3. Default: ${IMPORTANCE_LEVELS}
  HIDDEN_DIM          GNN hidden dim. Default: ${HIDDEN_DIM}
  GNN_LAYERS          Graph encoder layers. Default: ${GNN_LAYERS}
  TFM_LAYERS          Transformer layers. Default: ${TFM_LAYERS}
  NUM_HEADS           Transformer heads. Default: ${NUM_HEADS}
  BATCH_SIZE          Training batch size. Default: ${BATCH_SIZE}
  EPOCHS              Training epochs. Default: ${EPOCHS}
  OUTPUT_DATA         Combined training data path. Default: ${OUTPUT_DATA}
  OUTPUT_MODEL        Model checkpoint path. Default: ${OUTPUT_MODEL}
  OUTPUT_METRICS      Metrics JSON path. Default: ${OUTPUT_METRICS}
  USE_GOLD_ANSWER     1 to supervise with gold answers instead of predictions. Default: ${USE_GOLD_ANSWER}
  SUPERVISION_METHOD  attention or logit_grad. Default: ${SUPERVISION_METHOD}
  TOKENIZER_MODEL     HF tokenizer used for token IDs. Default: ${TOKENIZER_MODEL}

This script expects that for each dataset:
  1. LinearRAG has already been run
  2. The latest predictions.json exists under RESULTS_ROOT/<dataset>/<timestamp>/predictions.json
  3. The graph/import artifacts exist under IMPORT_ROOT/<dataset>/
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -z "${ATTN_MODEL}" ]]; then
  echo "ERROR: set ATTN_MODEL to the HF model/path used for prompt attention dumping." >&2
  usage >&2
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT_DATA}")" "$(dirname "${OUTPUT_MODEL}")" "$(dirname "${OUTPUT_METRICS}")"

IFS=',' read -r -a DATASET_ARRAY <<< "${DATASETS}"

echo "==> Dumping prompt attention supervision"
for ds in "${DATASET_ARRAY[@]}"; do
  ds="$(echo "${ds}" | xargs)"
  if [[ -z "${ds}" ]]; then
    continue
  fi

  pred_path="$(find "${RESULTS_ROOT}/${ds}" -path '*/predictions.json' -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
  if [[ -z "${pred_path}" ]]; then
    echo "ERROR: no predictions.json found for dataset '${ds}' under ${RESULTS_ROOT}/${ds}" >&2
    exit 1
  fi

  run_dir="$(dirname "${pred_path}")"
  out_path="${run_dir}/prompt_attention_scores.json"

  echo "  - ${ds}"
  echo "    predictions: ${pred_path}"
  echo "    output:      ${out_path}"

  if [[ "${USE_GOLD_ANSWER}" == "1" ]]; then
    python dump_last_layer_query_attention.py \
      --model "${ATTN_MODEL}" \
      --predictions-json "${pred_path}" \
      --method "${SUPERVISION_METHOD}" \
      --output "${out_path}" \
      --use-gold-answer
  else
    python dump_last_layer_query_attention.py \
      --model "${ATTN_MODEL}" \
      --predictions-json "${pred_path}" \
      --method "${SUPERVISION_METHOD}" \
      --output "${out_path}"
  fi
done

echo "==> Building multigraph training data"
python build_multigraph_gnn_data_from_linearrag.py \
  --datasets "${DATASETS}" \
  --results-root "${RESULTS_ROOT}" \
  --linearrag-import-dir "${IMPORT_ROOT}" \
  --tokenizer "${TOKENIZER_MODEL}" \
  --block-size "${VLLM_BLOCK_SIZE}" \
  --attention-filename prompt_attention_scores.json \
  --require-attention-supervision \
  --max-seq-len "${MAX_SEQ_LEN}" \
  --max-retrieved "${MAX_RETRIEVED}" \
  --output "${OUTPUT_DATA}"

echo "==> Training graph-conditioned block ranker"
python rag_query_gnn_predictor.py \
  --data-path "${OUTPUT_DATA}" \
  --importance-levels "${IMPORTANCE_LEVELS}" \
  --three-level-label-mode adaptive \
  --adaptive-medium-std 0.5 \
  --adaptive-high-std 1.0 \
  --adaptive-min-high-ratio 0.05 \
  --adaptive-max-high-ratio 0.10 \
  --three-level-class-weighting balanced \
  --hidden-dim "${HIDDEN_DIM}" \
  --gnn-layers "${GNN_LAYERS}" \
  --tfm-layers "${TFM_LAYERS}" \
  --num-heads "${NUM_HEADS}" \
  --batch-size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --output-model "${OUTPUT_MODEL}" \
  --output-metrics "${OUTPUT_METRICS}"

echo "==> Done"
echo "Training data: ${OUTPUT_DATA}"
echo "Model:         ${OUTPUT_MODEL}"
echo "Metrics:       ${OUTPUT_METRICS}"
