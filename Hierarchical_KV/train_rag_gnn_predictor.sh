#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/shared/gpfs/home/yh21/Hierarchical_KV"
cd "$ROOT"

source /mnt/shared/gpfs/home/yh21/miniconda3/etc/profile.d/conda.sh
conda activate vllm

DATASETS="${DATASETS:-hotpotqa,2wikimultihop,musique,medical}"
ATTN_MODEL="${ATTN_MODEL:-}"
ATTN_ARGS="${ATTN_ARGS:---use-gold-answer}"
BUILD_FILE="${BUILD_FILE:-data/all_multigraph_gnn_train.pt}"
CKPT_FILE="${CKPT_FILE:-outputs/all_multigraph_token_ranker_3tier.pt}"
METRICS_FILE="${METRICS_FILE:-outputs/all_multigraph_token_ranker_3tier_metrics.json}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-64}"
MAX_RETRIEVED="${MAX_RETRIEVED:-5}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"
DEVICE="${DEVICE:-cuda}"
SPACY_MODEL="${SPACY_MODEL:-en_core_web_trf}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-sentence-transformers/all-mpnet-base-v2}"
LLM_MODEL="${LLM_MODEL:-gpt-4o-mini}"
OPENAI_API_KEY="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"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
FORCE_RERUN="${FORCE_RERUN:-0}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"

if [[ "$LLM_MODEL" == gpt-* || "$LLM_MODEL" == o1* || "$LLM_MODEL" == o3* || "$LLM_MODEL" == o4* ]]; then
  if [[ -z "$OPENAI_API_KEY" ]]; then
    echo "OPENAI_API_KEY is required when LLM_MODEL=$LLM_MODEL" >&2
    echo "Set it before running, or edit train_rag_gnn_predictor.sh to assign it." >&2
    exit 1
  fi
  export OPENAI_API_KEY
fi

echo "=== Ensuring LinearRAG dataset path ==="
ln -sfn "$ROOT/dataset_linear-rag" "$ROOT/LinearRAG/dataset"

echo "=== Running LinearRAG for datasets ==="
cd "$ROOT/LinearRAG"
IFS=',' read -r -a DATASET_LIST <<< "$DATASETS"
for DS in "${DATASET_LIST[@]}"; do
  LATEST_PRED=$(find "results/$DS" -mindepth 2 -maxdepth 2 -name predictions.json 2>/dev/null | sort | tail -n 1 || true)
  GRAPH_FILE="import/$DS/LinearRAG.graphml"
  if [[ "$SKIP_EXISTING" == "1" && "$FORCE_RERUN" != "1" && -n "$LATEST_PRED" && -f "$LATEST_PRED" && -f "$GRAPH_FILE" ]]; then
    echo "Skipping LinearRAG for $DS; found $LATEST_PRED and $GRAPH_FILE"
    continue
  fi
  python run.py \
    --dataset_name "$DS" \
    --spacy_model "$SPACY_MODEL" \
    --embedding_model "$EMBEDDING_MODEL" \
    --llm_model "$LLM_MODEL"
done
cd "$ROOT"

echo "=== Building attention supervision (optional) ==="
if [[ -n "$ATTN_MODEL" ]]; then
  for DS in "${DATASET_LIST[@]}"; do
    PRED=$(ls -t "$ROOT/LinearRAG/results/$DS"/*/predictions.json | head -n 1)
    RUN_DIR=$(dirname "$PRED")
    if [[ "$SKIP_EXISTING" == "1" && "$FORCE_RERUN" != "1" && -f "$RUN_DIR/query_attention_scores.json" ]]; then
      echo "Skipping attention supervision for $DS; found $RUN_DIR/query_attention_scores.json"
      continue
    fi
    python dump_last_layer_query_attention.py \
      --model "$ATTN_MODEL" \
      --predictions-json "$PRED" \
      $ATTN_ARGS \
      --output "$RUN_DIR/query_attention_scores.json"
  done
fi

echo "=== Building unified multi-graph GNN training file ==="
if [[ "$SKIP_EXISTING" == "1" && "$FORCE_REBUILD" != "1" && -f "$BUILD_FILE" ]]; then
  echo "Skipping build; found $BUILD_FILE"
else
  BUILD_CMD=(
    python build_multigraph_gnn_data_from_linearrag.py
    --datasets "$DATASETS"
    --results-root LinearRAG/results
    --linearrag-import-dir LinearRAG/import
    --max-seq-len "$MAX_SEQ_LEN"
    --max-retrieved "$MAX_RETRIEVED"
    --output "$BUILD_FILE"
  )

  if [[ -n "$ATTN_MODEL" ]]; then
    BUILD_CMD+=(--require-attention-supervision)
  else
    BUILD_CMD+=(--skip-missing)
  fi

  "${BUILD_CMD[@]}"
fi

echo "=== Training 3-tier multi-graph predictor ==="
# if [[ "$SKIP_EXISTING" == "1" && "$FORCE_TRAIN" != "1" && -f "$CKPT_FILE" && -f "$METRICS_FILE" ]]; then
#   echo "Skipping training; found $CKPT_FILE and $METRICS_FILE"

# else
  python rag_query_gnn_predictor.py \
    --data-path "$BUILD_FILE" \
    --device "$DEVICE" \
    --importance-levels 3 \
    --balanced-sampling \
    --lr-scheduler plateau \
    --early-stop-patience 6 \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --output-model "$CKPT_FILE" \
    --output-metrics "$METRICS_FILE"
# fi

echo "checkpoint -> $CKPT_FILE"
echo "metrics -> $METRICS_FILE"
