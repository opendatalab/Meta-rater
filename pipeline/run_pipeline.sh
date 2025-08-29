#!/bin/bash
set -euo pipefail

# Default arguments
NUM_RATERS=${NUM_RATERS:-25}
NUM_SAMPLES=${NUM_SAMPLES:-256}
DIRICHLET_ALPHA=${DIRICHLET_ALPHA:-1.0}
PERCENTILE=${PERCENTILE:-85}
KEEP_LAST=${KEEP_LAST:-128}
TRAIN_RATIO=${TRAIN_RATIO:-0.8}
SEARCH_SAMPLES=${SEARCH_SAMPLES:-1000000}
SEARCH_ALPHA=${SEARCH_ALPHA:-1.0}
TOPK=${TOPK:-128}

OUT_DIR=${OUT_DIR:-pipeline_out}
mkdir -p "$OUT_DIR"

# Step 1: Generate random weight combinations for selecting data and pre-training small proxy models.
echo "[Step 1] Generate weights"
python step1_generate_weights.py \
    --num_raters "$NUM_RATERS" \
    --num_samples "$NUM_SAMPLES" \
    --strength "$DIRICHLET_ALPHA" \
    --output "$OUT_DIR/weights.json"

# Step 2: Select M data subsets according to each weight combination
echo "[Step 2] Select data for pre-training proxy models"
python step2_select_data.py \
    --inputs "../example_data/toy.jsonl" \
    --output "$OUT_DIR/filtered.jsonl" \
    --percentile 85 \
    --weights_json "$OUT_DIR/weights.json" \
    --weights_index 0

# After Step 2, pre-train proxy models, we recommend using Llama-Factory (https://github.com/hiyouga/LLaMA-Factory), and collect validation loss on validation set of Slimpajama

echo "[Step 3] Train a simple regression model"
python step3_train_regressor.py \
  --weights_file "$OUT_DIR/weights.json" \
  --losses_file "$OUT_DIR/losses.json" \
  --keep_last "$KEEP_LAST" \
  --train_ratio "$TRAIN_RATIO" \
  --model_out "$OUT_DIR/regressor_avg_loss.joblib" \
  --metrics_out "$OUT_DIR/regressor_metrics.json"

echo "[Step 4] Search optimal weights"
python step4_search_optimal.py \
  --model "$OUT_DIR/regressor_avg_loss.joblib" \
  --num_raters "$NUM_RATERS" \
  --search_samples "$SEARCH_SAMPLES" \
  --alpha "$SEARCH_ALPHA" \
  --topk "$TOPK" \
  --output "$OUT_DIR/optimal_weights.json"

echo "Done. Results saved in $OUT_DIR"

# Finally, use this optimal weight combination to select data and pre-train LLM