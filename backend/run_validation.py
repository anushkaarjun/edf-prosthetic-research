#!/usr/bin/env python3
"""
Run validation (and optionally test) on the saved 3-class model.
Uses the same 60/20/20 subject split and preprocessing as train_model_3class.py.

Usage:
  cd backend
  export DATA_DIR="/path/to/Prosethic Research Data/files 2"   # optional if fallback exists
  python3 run_validation.py
"""
import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
os.chdir(BACKEND_DIR)
sys.path.insert(0, str(BACKEND_DIR))

def _resolve_data_dir():
    if os.environ.get("DATA_DIR"):
        p = Path(os.environ["DATA_DIR"])
        if p.is_dir():
            return str(p.resolve())
    for d in [
        Path.home() / "Desktop" / "Outside of School" / "Prosethic Research Data" / "files 2",
        Path.home() / "Desktop" / "Outside of School" / "Prosthetic Research Data" / "files 2",
        BACKEND_DIR / "dataset",
    ]:
        if d.is_dir():
            return str(d.resolve())
    return ""

DATA_DIR = _resolve_data_dir()
MODEL_PATH = Path(os.environ.get("MODEL_PATH", str(BACKEND_DIR / "saved_model_3class.keras")))
CLASS_NAMES = ["Rest", "Left Hand", "Right Hand"]
N_CLASSES = 3

if not DATA_DIR:
    print("Set DATA_DIR to your EDF folder.")
    sys.exit(1)
if not MODEL_PATH.exists():
    print(f"Model not found: {MODEL_PATH}. Train first: python3 train_model_3class.py")
    sys.exit(1)


def main():
    import numpy as np
    from data_loader_edf import load_validation_from_edf_dir
    from model_cnn_lstm import to_three_class, balanced_accuracy_3class
    from sklearn.model_selection import train_test_split
    import tensorflow as tf

    random_state = int(os.environ.get("RANDOM_STATE", "42"))

    # Load model
    model = tf.keras.models.load_model(
        str(MODEL_PATH),
        custom_objects={"balanced_accuracy_3class": balanced_accuracy_3class},
    )
    print(f"Loaded: {MODEL_PATH}")

    # Load data (same as training)
    X, y_5, subject_ids = load_validation_from_edf_dir(
        DATA_DIR, max_subjects=None, max_trials_per_run=None, max_total_trials=None,
        fs=160, n_channels=64, classes_to_load=[0, 1, 4],
    )
    if X is None or len(X) < 20:
        print("Not enough data.")
        sys.exit(1)

    # Z-score (same as training)
    for i in range(len(X)):
        for c in range(X.shape[1]):
            seg = X[i, c, :]
            std = np.std(seg)
            if std > 1e-8:
                X[i, c, :] = (seg - np.mean(seg)) / std

    y = to_three_class(y_5)
    unique_subjects = np.unique(subject_ids)

    # 60/20/20 split (same as train_model_3class.py)
    train_val_subjects, test_subjects = train_test_split(
        unique_subjects, test_size=0.2, random_state=random_state
    )
    if len(test_subjects) == 0:
        test_subjects = unique_subjects[-1:]
        train_val_subjects = unique_subjects[:-1]
    train_subjects, val_subjects = train_test_split(
        train_val_subjects, test_size=0.25, random_state=random_state
    )
    if len(val_subjects) == 0:
        val_subjects = train_val_subjects[-1:]
        train_subjects = train_val_subjects[:-1]

    mask_val = np.isin(subject_ids, val_subjects)
    mask_test = np.isin(subject_ids, test_subjects)
    X_val = X[mask_val]
    y_val = y[mask_val]
    X_test = X[mask_test]
    y_test = y[mask_test]

    print(f"Validation set: {len(y_val)} trials ({len(val_subjects)} subjects)")
    print(f"Test set:       {len(y_test)} trials ({len(test_subjects)} subjects)")

    # Validation
    probs_val = model.predict(X_val, verbose=0)
    probs_val = np.nan_to_num(probs_val, nan=0.0, posinf=0.0, neginf=0.0)
    y_val_pred = np.argmax(probs_val, axis=1)
    val_acc = np.mean(y_val_pred == y_val)
    val_recalls = [np.sum((y_val == c) & (y_val_pred == c)) / max(1, np.sum(y_val == c)) for c in range(N_CLASSES)]
    val_balanced = np.mean(val_recalls)
    val_cm = [[int(np.sum((y_val == a) & (y_val_pred == p))) for p in range(N_CLASSES)] for a in range(N_CLASSES)]

    print("\n--- Validation ---")
    print(f"Accuracy:         {val_acc:.2%}")
    print(f"Balanced accuracy: {val_balanced:.2%}")
    for c in range(N_CLASSES):
        print(f"  Recall {CLASS_NAMES[c]}: {val_recalls[c]:.2%}")
    print("Confusion matrix (rows=actual, cols=predicted):")
    for a in range(N_CLASSES):
        print(f"  {CLASS_NAMES[a]}: {val_cm[a]}")

    # Test
    probs_test = model.predict(X_test, verbose=0)
    probs_test = np.nan_to_num(probs_test, nan=0.0, posinf=0.0, neginf=0.0)
    y_test_pred = np.argmax(probs_test, axis=1)
    test_acc = np.mean(y_test_pred == y_test)
    test_balanced = np.mean([np.sum((y_test == c) & (y_test_pred == c)) / max(1, np.sum(y_test == c)) for c in range(N_CLASSES)])

    print("\n--- Test ---")
    print(f"Accuracy:         {test_acc:.2%}")
    print(f"Balanced accuracy: {test_balanced:.2%}")


if __name__ == "__main__":
    main()
