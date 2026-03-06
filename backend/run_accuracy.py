#!/usr/bin/env python3
"""
Load a saved model and report accuracy on held-out test subjects.
Uses the same data pipeline as training (subject split, z-score) so metrics are comparable.

Usage:
  export DATA_DIR="/path/to/files 2"
  export MODEL_PATH="backend/saved_model_3class.keras"   # optional; default below
  python3 run_accuracy.py

Prints: test accuracy, balanced accuracy, confusion matrix.
"""
import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
os.chdir(BACKEND_DIR)
sys.path.insert(0, str(BACKEND_DIR))

_DEFAULT_DATA_DIR = "/Users/anushkaarjun/Desktop/Outside of School/Prosethic Research Data/files 2"
DATA_DIR = os.environ.get("DATA_DIR", _DEFAULT_DATA_DIR)
MODEL_PATH = os.environ.get("MODEL_PATH", str(BACKEND_DIR / "saved_model_3class.keras"))
if not DATA_DIR or not Path(DATA_DIR).is_dir():
    print("Set DATA_DIR to your EDF folder, e.g.:")
    print(f'  export DATA_DIR="{_DEFAULT_DATA_DIR}"')
    sys.exit(1)
if not Path(MODEL_PATH).exists():
    print(f"Model not found: {MODEL_PATH}")
    print("Train first: python3 train_model_3class.py  (or train_model.py for 2-class)")
    sys.exit(1)

try:
    import numpy as np
    import tensorflow as tf
except ImportError:
    print("Need numpy and tensorflow. pip install tensorflow")
    sys.exit(1)


def main():
    from data_loader_edf import load_validation_from_edf_dir
    from model_cnn_lstm import to_three_class, to_two_class
    from sklearn.model_selection import train_test_split

    random_state = int(os.environ.get("RANDOM_STATE", "42"))
    np.random.seed(random_state)

    # Load model and infer number of classes from output shape
    from model_cnn_lstm import balanced_accuracy_3class
    model = tf.keras.models.load_model(
        str(MODEL_PATH),
        custom_objects={"balanced_accuracy_3class": balanced_accuracy_3class},
    )
    n_classes = int(model.output.shape[-1])
    if n_classes == 2:
        classes_to_load = [1, 4]
        class_names = ["Rest", "Right Hand"]
        to_labels = to_two_class
    else:
        n_classes = 3
        classes_to_load = [0, 1, 4]
        class_names = ["Rest", "Left Hand", "Right Hand"]
        to_labels = to_three_class

    print(f"Loaded model: {MODEL_PATH} ({n_classes}-class)")
    print("Loading EDF data from", DATA_DIR)

    X, y_5, subject_ids = load_validation_from_edf_dir(
        DATA_DIR, max_subjects=None, max_trials_per_run=None, max_total_trials=None, fs=160, n_channels=64,
        classes_to_load=classes_to_load,
    )
    if X is None or len(X) < 10:
        print("Not enough data.")
        sys.exit(1)

    # Same preprocessing as training: per-trial per-channel z-score
    for i in range(len(X)):
        for c in range(X.shape[1]):
            seg = X[i, c, :]
            std = np.std(seg)
            if std > 1e-8:
                X[i, c, :] = (seg - np.mean(seg)) / std

    y = to_labels(y_5)
    unique_subjects = np.unique(subject_ids)
    # 80/20 split (same as training): 80% train subjects, 20% test subjects
    train_subjects, test_subjects = train_test_split(
        unique_subjects, test_size=0.2, random_state=random_state
    )
    if len(test_subjects) == 0:
        test_subjects = unique_subjects[-1:]
    mask_test = np.isin(subject_ids, test_subjects)
    X_test = X[mask_test]
    y_test = y[mask_test]

    print(f"Test set: {len(y_test)} trials from {len(test_subjects)} held-out subjects")

    # Predict and compute metrics
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    accuracy = np.mean(y_pred == y_test)
    recalls = [
        np.sum((y_test == c) & (y_pred == c)) / max(1, np.sum(y_test == c))
        for c in range(n_classes)
    ]
    balanced_accuracy = np.mean(recalls)
    confusion = [[int(np.sum((y_test == a) & (y_pred == p))) for p in range(n_classes)] for a in range(n_classes)]

    print("\n--- Accuracy ---")
    print(f"Test accuracy:       {accuracy:.2%}")
    print(f"Balanced accuracy:   {balanced_accuracy:.2%}")
    print("Per-class recall:")
    for c in range(n_classes):
        print(f"  {class_names[c]}: {recalls[c]:.2%}")
    print("Confusion matrix (rows=actual, cols=predicted):")
    for a in range(n_classes):
        print(f"  {class_names[a]}: {confusion[a]}")


if __name__ == "__main__":
    main()
