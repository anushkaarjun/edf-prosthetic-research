#!/usr/bin/env python3
"""
Test the visualization pipeline on the trained model and test set.
Loads the same test data and model as the UI, runs predictions, and reports accuracy.
Usage (from backend/): python3 test_visualization_data.py
"""
import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
os.chdir(BACKEND_DIR)
sys.path.insert(0, str(BACKEND_DIR))

# Reuse app's DATA_DIR resolution
def _resolve_data_dir():
    if os.environ.get("DATA_DIR"):
        p = Path(os.environ["DATA_DIR"])
        if p.is_dir():
            return str(p.resolve())
    fallbacks = [
        Path.home() / "Desktop" / "Outside of School" / "Prosethic Research Data" / "files 2",
        Path.home() / "Desktop" / "Outside of School" / "Prosthetic Research Data" / "files 2",
        BACKEND_DIR / "dataset",
    ]
    for p in fallbacks:
        if p.is_dir():
            return str(p.resolve())
    return ""

DATA_DIR = _resolve_data_dir()
if not DATA_DIR:
    print("DATA_DIR not set and no fallback path found. Set it to your EDF folder.")
    sys.exit(1)

CLASS_NAMES = ["Rest", "Left Hand", "Right Hand"]

def _zscore_trial(x):
    import numpy as np
    out = np.array(x, dtype=np.float32)
    if out.ndim == 2:
        for c in range(out.shape[0]):
            seg = out[c, :]
            std = np.std(seg)
            if std > 1e-8:
                out[c, :] = (seg - np.mean(seg)) / std
    return out

def main():
    import numpy as np

    # Load model
    from model_cnn_lstm import balanced_accuracy_3class
    import tensorflow as tf
    model_path = BACKEND_DIR / "saved_model_3class.keras"
    if not model_path.exists():
        print(f"Model not found: {model_path}. Train first with python3 train_model_3class.py")
        sys.exit(1)
    model = tf.keras.models.load_model(
        str(model_path),
        custom_objects={"balanced_accuracy_3class": balanced_accuracy_3class},
    )
    print(f"Loaded model from {model_path}")

    # Load test set (same as UI)
    from data_loader_edf import load_test_from_edf_dir
    random_state = int(os.environ.get("RANDOM_STATE", "42"))
    X_test, y_test = load_test_from_edf_dir(
        DATA_DIR, random_state=random_state, fs=160, n_channels=64, classes_to_load=(0, 1, 4)
    )
    if X_test is None or len(X_test) == 0:
        print("No test data loaded. Check DATA_DIR.")
        sys.exit(1)
    print(f"Loaded TEST set: {len(X_test)} trials from {DATA_DIR}")

    # Preprocess: z-score per trial (same as app)
    X_z = np.array([_zscore_trial(X_test[i]) for i in range(len(X_test))], dtype=np.float32)
    if not np.isfinite(X_z).all():
        X_z = np.nan_to_num(X_z, nan=0.0, posinf=0.0, neginf=0.0)

    def sanitize_probs(p):
        """Replace NaN/Inf with 0 and renormalize so probs sum to 1."""
        p = np.asarray(p, dtype=np.float64)
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        s = p.sum()
        if s <= 0:
            return np.ones(3) / 3.0
        return p / s

    # Batch predict for full accuracy (faster)
    batch_size = 32
    n = len(X_test)
    all_preds = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        probs = model.predict(X_z[start:end], verbose=0)
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        for k in range(probs.shape[0]):
            row = sanitize_probs(probs[k])
            all_preds.append(np.argmax(row))
    all_preds = np.array(all_preds, dtype=int)

    correct = int(np.sum(all_preds == y_test))
    accuracy = correct / n

    # Show first 10 samples with probabilities
    n_show = min(10, n)
    print(f"\nFirst {n_show} samples (actual -> predicted, probabilities):")
    print("-" * 60)
    for i in range(n_show):
        x_batch = X_z[i : i + 1]
        probs = model.predict(x_batch, verbose=0)[0]
        probs = sanitize_probs(probs)
        pred = int(np.argmax(probs))
        actual = int(y_test[i])
        probs_str = ", ".join(f"{CLASS_NAMES[j]}: {probs[j]:.2f}" for j in range(3))
        match = "✓" if pred == actual else "✗"
        print(f"  {i+1}. Actual: {CLASS_NAMES[actual]:12} -> Pred: {CLASS_NAMES[pred]:12} {match}  [{probs_str}]")

    print("-" * 60)
    print(f"Test accuracy on visualization data: {correct}/{n} = {accuracy:.2%}")
    print("\nThis is the same data and model the UI uses when you click 'Next sample'.")

if __name__ == "__main__":
    main()
