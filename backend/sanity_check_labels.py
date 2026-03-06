#!/usr/bin/env python3
"""
Sanity check: train with SHUFFLED labels. Test accuracy should drop to ~33% (random guess for 3-class).
If test accuracy stays high (e.g. > 50%) → likely bug (e.g. leakage or wrong labels).
Usage:
  export DATA_DIR="/path/to/files 2"
  python3 sanity_check_labels.py
"""
import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
os.chdir(BACKEND_DIR)
sys.path.insert(0, str(BACKEND_DIR))

DATA_DIR = os.environ.get("DATA_DIR", "")
if not DATA_DIR or not Path(DATA_DIR).is_dir():
    print("Set DATA_DIR to your EDF folder.")
    sys.exit(1)

import numpy as np
from sklearn.model_selection import train_test_split

np.random.seed(42)

from data_loader_edf import load_validation_from_edf_dir
from model_cnn_lstm import build_model_3class, to_three_class

def main():
    print("Loading data (same as train_model_3class)...")
    X, y_5, subject_ids = load_validation_from_edf_dir(
        DATA_DIR, max_subjects=20, max_trials_per_run=80, max_total_trials=2500, fs=160, n_channels=64,
        classes_to_load=[0, 1, 4],
    )
    if X is None or len(X) < 20:
        print("Not enough data.")
        sys.exit(1)

    for i in range(len(X)):
        for c in range(X.shape[1]):
            seg = X[i, c, :].astype(np.float64)
            std = np.std(seg)
            if std > 1e-8:
                X[i, c, :] = (seg - np.mean(seg)) / std

    y = to_three_class(y_5)
    unique_subjects = np.unique(subject_ids)
    train_subjects, test_subjects = train_test_split(unique_subjects, test_size=0.2, random_state=42)
    mask_train = np.isin(subject_ids, train_subjects)
    mask_test = np.isin(subject_ids, test_subjects)
    X_train = X[mask_train]
    y_train = y[mask_train]
    X_test = X[mask_test]
    y_test = y[mask_test]

    # Shuffle labels so X and y are uncorrelated
    y_train_shuffled = y_train.copy()
    np.random.shuffle(y_train_shuffled)
    print("Labels SHUFFLED for training (sanity check). Expect test acc ~33%.")

    import tensorflow as tf
    tf.random.set_seed(42)
    model = build_model_3class(n_channels=64, n_times=X.shape[2])
    model.fit(
        X_train, y_train_shuffled,
        validation_data=(X_test, y_test),
        epochs=10,
        batch_size=32,
        verbose=1,
    )
    y_test_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    test_acc = np.mean(y_test_pred == y_test)
    print(f"\nTest accuracy (with SHUFFLED train labels): {test_acc:.2%}")
    if test_acc > 0.5:
        print("*** SANITY CHECK FAILED: accuracy should be ~33% with shuffled labels. Check for leakage or bugs.")
        sys.exit(1)
    print("Sanity check passed: test accuracy near random (~33%).")
    sys.exit(0)

if __name__ == "__main__":
    main()
