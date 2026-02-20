#!/bin/bash
# Train CNN-LSTM (3 classes) with settings aimed at 60% accuracy.
# Run from repo root: bash scripts/train_cnn_lstm_60pct.sh
# Or with custom data path: DATA_PATH="/path/to/edf" bash scripts/train_cnn_lstm_60pct.sh

set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

DATA_PATH="${DATA_PATH:-/Users/anushkaarjun/Desktop/Outside of School/Prosethic Research Data/files 2}"
MAX_SUBJECTS="${MAX_SUBJECTS:-15}"

echo "Training CNN-LSTM (3 classes) for 60% target..."
echo "  Data: $DATA_PATH"
echo "  Max subjects: $MAX_SUBJECTS"
echo "  Epochs: 50, augmentation: on (mild), multi-seed: 3"
echo ""

python3 scripts/train_cnn_lstm.py \
  --data-path "$DATA_PATH" \
  --max-subjects "$MAX_SUBJECTS" \
  --exclude-both-feet \
  --multi-seed 3

echo ""
echo "Done. Check validation accuracy above."
