#!/bin/bash
# Run all models with 3 classes (exclude Both Feet) and report accuracy.
# Usage: ./scripts/run_all_models_3class.sh [DATA_PATH]
# Default DATA_PATH from Makefile.

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

DATA_PATH="${1:-/Users/anushkaarjun/Desktop/Outside of School/Prosethic Research Data/files 2}"
if [[ ! -d "$DATA_PATH" ]]; then
  echo "ERROR: Data path not found: $DATA_PATH"
  exit 1
fi

echo "=============================================="
echo "Running ALL models with 3 classes (no Both Feet)"
echo "Data path: $DATA_PATH"
echo "=============================================="

RESULTS_FILE="$REPO_ROOT/results/accuracy_3class_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p "$REPO_ROOT/results"
echo "Results will be appended to: $RESULTS_FILE"
echo ""

# 1) CSP+SVM and EEGNet (train_on_validation_data.py)
echo "========== 1/3: CSP+SVM and EEGNet (3 classes) =========="
python3 scripts/train_on_validation_data.py \
  --data-path "$DATA_PATH" \
  --max-subjects 10 \
  --eegnet --csp-svm \
  --exclude-both-feet 2>&1 | tee -a "$RESULTS_FILE" || true

# 2) CNN-LSTM (3 classes: Both Fists, Left Hand, Right Hand)
echo ""
echo "========== 2/3: CNN-LSTM (3 classes) =========="
python3 scripts/train_cnn_lstm.py \
  --data-path "$DATA_PATH" \
  --max-subjects 10 \
  --exclude-both-feet 2>&1 | tee -a "$RESULTS_FILE" || true

# 3) ImprovedEEGNet, SimpleEEGNet, DeepEEGNet (test-all, 3 classes)
echo ""
echo "========== 3/3: ImprovedEEGNet, SimpleEEGNet, DeepEEGNet (3 classes) =========="
python3 scripts/train_improved_model.py \
  --data-path "$DATA_PATH" \
  --max-subjects 10 \
  --epochs 80 \
  --test-all \
  --exclude-both-feet 2>&1 | tee -a "$RESULTS_FILE" || true

echo ""
echo "=============================================="
echo "Done. Check output above and $RESULTS_FILE for accuracy."
echo "=============================================="
