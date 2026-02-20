# Model Accuracy and Latency Summary

This document consolidates validation accuracy and inference latency for all EEG motor imagery models.

## Summary Table

| Model | Classes | Validation Accuracy | Inference Latency (avg) | Data Format | Model Size |
|-------|---------|---------------------|-------------------------|-------------|------------|
| **CNN-LSTM (2-class)** | 2 | **74.81%** | **~3.4 ms** | 2s @ 160Hz | 2.9MB |
| **CNN-LSTM (3-class)** | 3 | **51.94%** | **~3.4 ms** | 2s @ 160Hz | 2.9MB |
| **ImprovedEEGNet** | 4 | **≥60%** | — | 0.5s @ 250Hz | — |
| **CSP+SVM** | 4 | 44.83% | — | 0.5s @ 250Hz | 187KB |
| **EEGNet** | 4 | 43.10% | — | 0.5s @ 250Hz | 285KB |

*— = Not yet measured*

---

## Notes

- **CNN-LSTM (2-class)**: Open Right Fist vs Close Fists (excludes Open Left Fist). Trained with `--exclude-left-fist`. Best accuracy.
- **CNN-LSTM (3-class)**: Open Left Fist, Open Right Fist, Close Fists. Baseline configuration.
- **CNN-LSTM latency**: Measured via `scripts/measure_cnn_lstm_latency.py` on CPU. GPU typically faster.
- **ImprovedEEGNet**: Requires augmentation, 10+ subjects, 80 epochs to reach ≥60%.
- **Latency threshold**: Target ≤ 1000 ms (1 second) for real-time use. All measured models pass.

---

## How to Measure Latency

```bash
# CNN-LSTM
python3 scripts/measure_cnn_lstm_latency.py --model-path ../models/best_model.pth

# Or via API (returns latency_ms in /predict response when CNN-LSTM is loaded)
# Start API, load model, then POST to /predict
```

---

## Sources

- **Accuracies**: `docs/MODEL_ACCURACIES.md`, training runs (2-class: 74.81%)
- **CNN-LSTM latency**: `scripts/measure_cnn_lstm_latency.py` (100 samples, warmup 10)
- **CSP+SVM latency**: Measured in `scripts/run_tests.py` (Test 2) when run with CSP+SVM models
