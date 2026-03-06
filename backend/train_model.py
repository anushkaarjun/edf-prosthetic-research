#!/usr/bin/env python3
"""
Train the 2-class CNN-LSTM on your EDF data and save it so the app can load it.
Usage:
  export DATA_DIR="/path/to/files 2"
  python3 train_model.py
Saves: saved_model.keras in the backend folder. Then start app.py (it will load this if MODEL_PATH is not set).
"""
import json
import os
import sys
from pathlib import Path

# Run from backend directory
BACKEND_DIR = Path(__file__).resolve().parent
os.chdir(BACKEND_DIR)
sys.path.insert(0, str(BACKEND_DIR))

DATA_DIR = os.environ.get("DATA_DIR", "")
if not DATA_DIR or not Path(DATA_DIR).is_dir():
    print("Set DATA_DIR to your EDF folder, e.g.:")
    print('  export DATA_DIR="/Users/anushkaarjun/Desktop/Outside of School/Prosethic Research Data/files 2"')
    sys.exit(1)

try:
    import tensorflow as tf
except ImportError:
    print("TensorFlow is required for training. Install it with:")
    print("  pip3 install tensorflow")
    sys.exit(1)

def main():
    import numpy as np
    from data_loader_edf import load_validation_from_edf_dir
    from model_cnn_lstm import build_model, to_two_class

    random_state = int(os.environ.get("RANDOM_STATE", "42"))
    np.random.seed(random_state)
    tf.random.set_seed(random_state)

    print("Loading EDF data from", DATA_DIR)
    # Rest (4) vs Right Hand (1) only for a clear 2-class problem
    X, y_5, subject_ids = load_validation_from_edf_dir(
        DATA_DIR, max_subjects=20, max_trials_per_run=80, max_total_trials=2500, fs=160, n_channels=64,
        classes_to_load=[1, 4],
    )
    if X is None or len(X) < 20:
        print("Not enough data. Need at least 20 trials from EDF folder.")
        sys.exit(1)

    # Per-trial per-channel z-score (improves generalization)
    for i in range(len(X)):
        for c in range(X.shape[1]):
            seg = X[i, c, :]
            std = np.std(seg)
            if std > 1e-8:
                X[i, c, :] = (seg - np.mean(seg)) / std

    y = to_two_class(y_5)
    n_classes = 2
    n_rest, n_motor = int(np.sum(y == 0)), int(np.sum(y == 1))
    unique_subjects = np.unique(subject_ids)
    n_subjects = len(unique_subjects)
    print(f"Loaded {len(X)} trials, shape {X.shape}, {n_subjects} subjects, 2-class labels: Rest={n_rest}, Right Hand={n_motor}")

    if n_subjects < 2:
        print("Subject-independent evaluation requires at least 2 subjects. Add more subject folders (S001, S002, ...) to DATA_DIR.")
        sys.exit(1)

    # 80/20 split: 80% subjects for training, 20% held out for test (subject-independent)
    from sklearn.model_selection import train_test_split
    train_subjects, test_subjects = train_test_split(
        unique_subjects, test_size=0.2, random_state=random_state
    )
    if len(test_subjects) == 0:
        test_subjects = unique_subjects[-1:]
        train_subjects = unique_subjects[:-1]
    mask_train = np.isin(subject_ids, train_subjects)
    mask_test = np.isin(subject_ids, test_subjects)
    X_train_pool = X[mask_train]
    y_train_pool = y[mask_train]
    X_test = X[mask_test]
    y_test = y[mask_test]
    print(f"Train subjects: {len(train_subjects)}, test subjects (held out): {len(test_subjects)}")
    print(f"Train pool: {len(y_train_pool)} trials, test set: {len(y_test)} trials (unseen subjects)")
    if len(test_subjects) < 5:
        print("\n*** WARNING: Few test subjects ({}) → test accuracy has high variance.".format(len(test_subjects)))

    # Oversample Right Hand within training pool only
    rest_idx = np.where(y_train_pool == 0)[0]
    right_idx = np.where(y_train_pool == 1)[0]
    n_right = len(right_idx)
    n_rest = len(rest_idx)
    if n_right > 0 and n_rest > n_right:
        repeat = (n_rest + n_right - 1) // n_right
        right_oversampled = np.tile(right_idx, repeat)[:n_rest]
        balance_idx = np.concatenate([rest_idx, right_oversampled])
        np.random.shuffle(balance_idx)
        X_train_pool = X_train_pool[balance_idx]
        y_train_pool = y_train_pool[balance_idx]
        print(f"Oversampled Right Hand in train pool: {len(y_train_pool)} (Rest={np.sum(y_train_pool==0)}, Right Hand={np.sum(y_train_pool==1)})")

    # 80/20 split within training pool: 80% train, 20% val (for early stopping)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_pool, y_train_pool, test_size=0.2, stratify=y_train_pool, random_state=random_state
    )
    print(f"Split: train={len(y_train)}, val={len(y_val)}, test={len(y_test)} (test = held-out subjects only)")

    # Class weights from training set
    total = len(y_train)
    n_rest, n_motor = int(np.sum(y_train == 0)), int(np.sum(y_train == 1))
    class_weight = {
        0: total / (2 * n_rest) if n_rest > 0 else 1.0,
        1: total / (2 * n_motor) if n_motor > 0 else 1.0,
    }
    print(f"Class weights: Rest={class_weight[0]:.2f}, Right Hand={class_weight[1]:.2f}")

    model = build_model(n_channels=64, n_times=X.shape[2], n_classes=n_classes)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=12,
            restore_best_weights=True,
            mode="max",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            mode="max",
        ),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=60,
        batch_size=32,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    # Validation metrics (all from predicted vs actual labels on X_val)
    y_val_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)

    # Accuracy = (TP + TN) / total = fraction of correct predictions
    val_accuracy = np.mean(y_val_pred == y_val)

    # Per-class counts for Rest (0) and Right Hand (1)
    # TP_c = predicted c and actual c; FP_c = predicted c but actual other; FN_c = actual c but predicted other
    def tp_fp_fn(c):
        tp = np.sum((y_val == c) & (y_val_pred == c))
        fp = np.sum((y_val != c) & (y_val_pred == c))
        fn = np.sum((y_val == c) & (y_val_pred != c))
        return tp, fp, fn

    # Precision = TP / (TP+FP), Recall = TP / (TP+FN)
    # F1 = 2 * (precision * recall) / (precision + recall); use 0 if denominator is 0
    def f1(tp, fp, fn):
        if tp + fp == 0 or tp + fn == 0:
            return 0.0
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        if prec + rec == 0:
            return 0.0
        return 2 * prec * rec / (prec + rec)

    f1_rest = f1(*tp_fp_fn(0))
    f1_right = f1(*tp_fp_fn(1))
    macro_f1 = (f1_rest + f1_right) / 2

    recall_rest = np.sum((y_val == 0) & (y_val_pred == 0)) / max(1, np.sum(y_val == 0))
    recall_right = np.sum((y_val == 1) & (y_val_pred == 1)) / max(1, np.sum(y_val == 1))
    balanced_accuracy = (recall_rest + recall_right) / 2

    print("\n--- Validation metrics ---")
    print("Accuracy:          (TP+TN)/total")
    print(f"  {val_accuracy:.2%}")
    print("Balanced accuracy: mean of per-class recall")
    print(f"  {balanced_accuracy:.2%}")
    print("F1 (harmonic mean of precision & recall):")
    print(f"  Rest:       {f1_rest:.4f}")
    print(f"  Right Hand: {f1_right:.4f}")
    print(f"  Macro F1:   {macro_f1:.4f}")
    print("Confusion matrix (rows=actual, cols=predicted):")
    for actual in range(n_classes):
        row = [np.sum((y_val == actual) & (y_val_pred == p)) for p in range(n_classes)]
        print(f"  {['Rest', 'Right Hand'][actual]}: {row}")

    # Test set evaluation (held-out, never used for training or early stopping)
    y_test_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    test_accuracy = np.mean(y_test_pred == y_test)
    print("\n--- Test set (held-out) ---")
    print(f"Test accuracy: {test_accuracy:.2%}")
    for actual in range(n_classes):
        row = [np.sum((y_test == actual) & (y_test_pred == p)) for p in range(n_classes)]
        print(f"  {['Rest', 'Right Hand'][actual]}: {row}")

    # Inference latency (ms per sample, average over 50 runs after warmup)
    import time
    warmup = X_val[:3]
    model.predict(warmup, verbose=0)
    n_runs = 50
    t0 = time.perf_counter()
    for _ in range(n_runs):
        model.predict(X_val[:1], verbose=0)
    latency_ms = (time.perf_counter() - t0) / n_runs * 1000
    print(f"\nInference latency: {latency_ms:.2f} ms per sample")

    # Save metrics + latency for chart (merge with existing entries by model name)
    eval_path = BACKEND_DIR / "eval_results.json"
    existing = []
    if eval_path.exists():
        try:
            with open(eval_path) as f:
                existing = json.load(f)
        except Exception:
            existing = []
    entry = {
        "model": "2-class CNN-LSTM",
        "accuracy": float(val_accuracy),
        "test_accuracy": float(test_accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "f1_rest": float(f1_rest),
        "f1_right": float(f1_right),
        "macro_f1": float(macro_f1),
        "latency_ms": round(latency_ms, 2),
    }
    other = [e for e in existing if e.get("model") != "2-class CNN-LSTM"]
    with open(eval_path, "w") as f:
        json.dump(other + [entry], f, indent=2)
    print(f"Wrote metrics to {eval_path}")

    out_path = BACKEND_DIR / "saved_model.keras"
    model.save(out_path)
    print(f"\nSaved model to {out_path}")
    print("Start the app with: export DATA_DIR=... ; python3 app.py")
    print("The app will load saved_model.keras by default if MODEL_PATH is not set.")

if __name__ == "__main__":
    main()
