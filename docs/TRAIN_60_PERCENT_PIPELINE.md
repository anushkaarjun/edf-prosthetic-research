# Pipeline that reached 60%+ validation accuracy

The results in `results/nn_results.txt` (60.71% and 62.50% validation accuracy) came from this setup:

## Setup

- **Script:** `scripts/train_model.py`
- **Model:** EEGMotorImageryNet (from `src/edf_ml_model/model.py`)
- **Data:** 4 classes (Both Feet, Both Fists, Left Hand, Right Hand), 0.5s epochs, 5 subjects
- **Split:** Per-subject train/val/test (~64% / 16% / 20%)
- **Training:**
  1. Phase 1: Train all weights for 30 epochs
  2. Phase 2: Freeze backbone (only classifier trainable)
  3. Phase 3: Hyperparameter tuning on validation (LR × weight decay grid)
  4. Phase 4: Fine-tune classifier 20 epochs with best hyperparameters

## How to run

From the repo root:

```bash
# Using Make (set DATA_PATH in Makefile if needed)
make train-eegnet-freeze

# Or directly
python3 scripts/train_model.py \
  --data-path "/path/to/your/EDF/data" \
  --max-subjects 5 \
  --freeze-after 30 \
  --epochs 50
```

Results are printed per subject (validation and test accuracy). The pipeline reports average validation accuracy across subjects; individual subjects in the older run reached **60.71%** and **62.50%**.
