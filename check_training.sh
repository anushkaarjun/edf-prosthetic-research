#!/bin/bash
# Script to check training progress and compare accuracies

echo "=== Training Status ==="
ps aux | grep train_improved_model | grep -v grep && echo "Training in progress..." || echo "Training completed or not running"

echo ""
echo "=== Current Model Accuracies ==="
echo ""
echo "1. CNN-LSTM:     51.94% (Current Best)"
echo "2. CSP+SVM:      44.83%"
echo "3. EEGNet:       43.10%"
echo "4. ImprovedEEGNet: Training..."
echo ""

echo "=== Training Output (Last 20 lines) ==="
tail -20 results/improved_model_results.txt 2>/dev/null || echo "Results file not created yet..."

echo ""
echo "=== Models Available ==="
ls -lh models/*.pth models/*.pkl 2>/dev/null | awk '{print $9, "("$5")"}'

echo ""
echo "=== To monitor training in real-time ==="
echo "tail -f results/improved_model_results.txt"
