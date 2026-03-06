#!/usr/bin/env python3
"""
Train the 3-class CNN-LSTM on your EDF data: Rest, Left Hand, Right Hand.

Preprocessing (in data loader):
  - Filter the signals to keep only 8–30 Hz (the range that matters for imagining movement).
  - Ensure every trial has exactly 64 channels and the same length (4.2 s = 672 samples at 160 Hz):
    pad short trials and trim long ones.
  - Keep only Rest, Left hand, and Right hand trials (three classes).
  - In the end you have one array: (number of trials, 64 channels, 672 time points).

Training pipeline (3-class):
  - For each trial and each channel, subtract the mean and divide by the standard deviation
    (per-trial per-channel z-score).
  - Convert labels to three classes: Rest (0), Left hand (1), Right hand (2).
  - Because there are usually more Rest trials, undersample Rest and duplicate Left hand and
    Right hand trials so the three classes are about equal (1:1:1).
  - Set class weights so the model treats all three classes as equally important
    (with a slight boost for motor classes to avoid defaulting to Rest).

Training:
  - Train on the training set in small batches (e.g. 32 trials at a time), for up to 60 epochs.
  - After each epoch, check accuracy (and balanced accuracy) on the validation set.
  - If validation balanced accuracy doesn't improve for 12 epochs, stop and keep the best model.
  - If it's stuck for 5 epochs, reduce the learning rate by half.
  - Save that best model as saved_model_3class.keras.

Also: 60/20/20 subject split, augmentation.
Usage:
  export DATA_DIR="/path/to/files 2"
  python3 train_model_3class.py
Saves: saved_model_3class.keras and appends to eval_results.json.
"""
import json
import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
os.chdir(BACKEND_DIR)
sys.path.insert(0, str(BACKEND_DIR))

_CANDIDATE_DATA_DIRS = [
    "/Users/anushkaarjun/Desktop/Outside of School/Prosethic Research Data/files 2",
    "/Users/anushkaarjun/Desktop/Outside of School/Prosethic Research Data",  # S001, S002 directly here
    "/Users/anushkaarjun/Desktop/Outside of School/Prosthetic Research Data/files 2",
    "/Users/anushkaarjun/Desktop/Outside of School/Prosthetic Research Data",
    "/Users/anushkaarjun/Desktop/Outside of School/edf-prosthetic-research",
    "/Users/anushkaarjun/Desktop/Outside of School/synopsys/Prosethic Research Data/files 2",
    "/Users/anushkaarjun/Desktop/Outside of School/synopsys/Prosethic Research Data",
]
DATA_DIR = os.environ.get("DATA_DIR", "").strip()
if not DATA_DIR:
    for d in _CANDIDATE_DATA_DIRS:
        if Path(d).is_dir():
            DATA_DIR = d
            break
if not DATA_DIR or not Path(DATA_DIR).is_dir():
    print("DATA_DIR must point to the folder containing S001, S002, etc. None of the usual paths exist.")
    print("Find your EDF folder, then run:")
    print('  export DATA_DIR="/full/path/to/your/files 2"')
    print("  python3 train_model_3class.py")
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
    from model_cnn_lstm import build_model_3class, to_three_class, balanced_accuracy_3class

    random_state = int(os.environ.get("RANDOM_STATE", "42"))
    np.random.seed(random_state)
    tf.random.set_seed(random_state)

    CLASS_NAMES = ["Rest", "Left Hand", "Right Hand"]
    n_classes = 3

    print("Loading EDF data from", DATA_DIR, "(all subjects, all trials — no max)")
    # Rest (4), Left Hand (0), Right Hand (1); no caps → use all available data
    X, y_5, subject_ids = load_validation_from_edf_dir(
        DATA_DIR, max_subjects=None, max_trials_per_run=None, max_total_trials=None, fs=160, n_channels=64,
        classes_to_load=[0, 1, 4],
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

    y = to_three_class(y_5)
    counts = {c: int(np.sum(y == c)) for c in range(n_classes)}
    unique_subjects = np.unique(subject_ids)
    n_subjects = len(unique_subjects)
    print(f"Loaded {len(X)} trials, shape {X.shape}, {n_subjects} subjects, 3-class: {counts}")

    if n_subjects < 3:
        print("Subject-independent evaluation requires at least 3 subjects (train/val/test). Add more subject folders (S001, S002, ...) to DATA_DIR.")
        sys.exit(1)

    # 60/20/20 split by SUBJECTS (all data used; 60% train, 20% val, 20% test — no subject in two sets)
    from sklearn.model_selection import train_test_split
    train_val_subjects, test_subjects = train_test_split(
        unique_subjects, test_size=0.2, random_state=random_state  # 20% test
    )
    if len(test_subjects) == 0:
        test_subjects = unique_subjects[-1:]
        train_val_subjects = unique_subjects[:-1]
    train_subjects, val_subjects = train_test_split(
        train_val_subjects, test_size=0.25, random_state=random_state  # 25% of remaining = 20% val, 60% train
    )
    if len(val_subjects) == 0:
        val_subjects = train_val_subjects[-1:]
        train_subjects = train_val_subjects[:-1]
    mask_train = np.isin(subject_ids, train_subjects)
    mask_val = np.isin(subject_ids, val_subjects)
    mask_test = np.isin(subject_ids, test_subjects)
    X_train_pool = X[mask_train]
    y_train_pool = y[mask_train]
    X_val = X[mask_val]
    y_val = y[mask_val]
    X_test = X[mask_test]
    y_test = y[mask_test]
    print(f"Train subjects: {len(train_subjects)}, Val subjects (held out): {len(val_subjects)}, Test subjects (held out): {len(test_subjects)}")
    print(f"Train: {len(y_train_pool)} trials, Val: {len(y_val)} trials, Test: {len(y_test)} trials")
    print("(Val & test = unseen subjects → val acc is a realistic proxy for test; early stopping uses val.)")

    counts_pool = {c: int(np.sum(y_train_pool == c)) for c in range(n_classes)}
    print(f"Train pool per class: Rest={counts_pool[0]}, Left Hand={counts_pool[1]}, Right Hand={counts_pool[2]}")
    if counts_pool[1] == 0 or counts_pool[2] == 0:
        print("\n*** ERROR: Training subjects have no Left Hand and/or no Right Hand trials.")
        print("    The 60/20/20 subject split put all motor-imagery data in val/test.")
        print("    Fix: try a different split so train gets motor trials, e.g.:")
        print("      export RANDOM_STATE=0   (or 1, 2, 3, ... then run again)")
        print("      python3 train_model_3class.py")
        print("    Or add more EDF runs/subjects that contain T1 (Left) and T2 (Right) events.")
        sys.exit(1)

    if len(test_subjects) < 5:
        print("\n*** WARNING: Few test subjects ({}) → test accuracy has high variance. Consider more subjects or LOSO.".format(len(test_subjects)))

    # Balanced 1:1:1 — undersample Rest, oversample Left/Right so all three classes have equal count.
    # This stops the model from collapsing to "always Rest".
    motor_max = max(counts_pool[1], counts_pool[2])
    target_per_class = motor_max  # each class gets this many trials
    rs = np.random.RandomState(random_state)
    balance_idx = []
    for c in range(n_classes):
        idx = np.where(y_train_pool == c)[0]
        n_c = len(idx)
        if n_c >= target_per_class:
            # Undersample (e.g. Rest): random sample without replacement
            balance_idx.append(rs.choice(idx, size=target_per_class, replace=False))
        else:
            # Oversample (e.g. Left/Right): sample with replacement to reach target
            balance_idx.append(rs.choice(idx, size=target_per_class, replace=True))
    balance_idx = np.concatenate(balance_idx)
    rs.shuffle(balance_idx)
    X_train_pool = X_train_pool[balance_idx]
    y_train_pool = y_train_pool[balance_idx]
    counts_after = {c: int(np.sum(y_train_pool == c)) for c in range(n_classes)}
    print(f"After 1:1:1 balance (Rest undersampled, motor oversampled): {counts_after}")

    # Class weights: slight downweight Rest, boost motor so model doesn't default to Rest
    total = len(y_train_pool)
    class_weight = {
        c: total / (n_classes * max(1, int(np.sum(y_train_pool == c)))) for c in range(n_classes)
    }
    class_weight[0] *= 0.6   # downweight Rest
    class_weight[1] *= 1.5  # boost Left
    class_weight[2] *= 1.5  # boost Right
    for c in range(n_classes):
        class_weight[c] = min(max(float(class_weight[c]), 0.5), 5.0)
    print(f"Class weights (Rest down, motor up): {class_weight}")

    # Train = train_pool (no trial split); Val and Test = held-out subjects
    X_train, y_train = X_train_pool, y_train_pool
    print(f"Train: {len(y_train)} trials, Val: {len(y_val)} trials, Test: {len(y_test)} trials (val & test = unseen subjects)")

    # Balanced batches: each batch has ~equal samples per class → higher balanced accuracy
    class BalancedBatchSequence(tf.keras.utils.Sequence):
        def __init__(self, X, y, batch_size, n_classes=3, shuffle=True, seed=42):
            self.X, self.y = X, y
            self.batch_size = batch_size
            self.n_classes = n_classes
            self.shuffle = shuffle
            self.seed = seed
            self.indices_per_class = [np.where(y == c)[0] for c in range(n_classes)]
            self.n_per_class = [len(idx) for idx in self.indices_per_class]
            self.steps_per_epoch = max(1, sum(self.n_per_class) // batch_size)
            self.epoch = 0
            self.epoch_indices = self._build_epoch_indices()

        def _build_epoch_indices(self):
            n_total = self.steps_per_epoch * self.batch_size
            per_class = n_total // self.n_classes
            extra = n_total - per_class * self.n_classes
            rs = np.random.RandomState(self.seed + self.epoch * 997)
            batch_idx = []
            for c in range(self.n_classes):
                n_c = self.n_per_class[c]
                take = per_class + (1 if c < extra else 0)
                if n_c > 0 and take > 0:
                    batch_idx.append(rs.choice(self.indices_per_class[c], size=take, replace=True))
            batch_idx = np.concatenate(batch_idx) if batch_idx else np.arange(min(n_total, len(self.y)))
            rs.shuffle(batch_idx)
            return batch_idx

        def __len__(self):
            return self.steps_per_epoch

        def on_epoch_end(self):
            if self.shuffle:
                self.epoch += 1
                self.epoch_indices = self._build_epoch_indices()

        def __getitem__(self, idx):
            start = idx * self.batch_size
            end = min(start + self.batch_size, len(self.epoch_indices))
            batch_idx = self.epoch_indices[start:end]
            X_batch = self.X[batch_idx].copy()
            # Stronger augmentation to reduce overfitting: time shift ±100 samples, moderate noise
            rs = np.random.RandomState(self.seed + self.epoch * 997 + idx)
            shift = rs.randint(-100, 101)
            if shift != 0:
                for i in range(X_batch.shape[0]):
                    X_batch[i] = np.roll(X_batch[i], shift, axis=1)
            X_batch += rs.randn(*X_batch.shape).astype(X_batch.dtype) * 0.02
            return X_batch, self.y[batch_idx]

    batch_size = 32
    train_seq = BalancedBatchSequence(X_train, y_train, batch_size, n_classes=3, shuffle=True, seed=random_state)

    model = build_model_3class(n_channels=64, n_times=X.shape[2])
    model.summary()

    # Moderate LR with clip to avoid NaN; ReduceLROnPlateau will lower if val balanced acc plateaus
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", balanced_accuracy_3class],
    )

    # Early stopping: stop if val balanced accuracy doesn't improve for 12 epochs; reduce LR if stuck 5
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_balanced_accuracy_3class",
            patience=12,
            restore_best_weights=True,
            mode="max",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_balanced_accuracy_3class",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            mode="max",
        ),
    ]
    history = model.fit(
        train_seq,
        validation_data=(X_val, y_val),
        epochs=60,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    # Train accuracy (monitor for overfitting: large train-test gap is suspicious)
    y_train_pred = np.argmax(model.predict(X_train, verbose=0), axis=1)
    train_accuracy = np.mean(y_train_pred == y_train)

    # Validation metrics
    y_val_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    val_accuracy = np.mean(y_val_pred == y_val)

    def tp_fp_fn(c):
        tp = np.sum((y_val == c) & (y_val_pred == c))
        fp = np.sum((y_val != c) & (y_val_pred == c))
        fn = np.sum((y_val == c) & (y_val_pred != c))
        return tp, fp, fn

    def f1(tp, fp, fn):
        if tp + fp == 0 or tp + fn == 0:
            return 0.0
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        if prec + rec == 0:
            return 0.0
        return 2 * prec * rec / (prec + rec)

    f1_scores = [f1(*tp_fp_fn(c)) for c in range(n_classes)]
    macro_f1 = np.mean(f1_scores)
    recalls = [
        np.sum((y_val == c) & (y_val_pred == c)) / max(1, np.sum(y_val == c))
        for c in range(n_classes)
    ]
    balanced_accuracy = np.mean(recalls)

    print("\n--- Train accuracy (same subjects as train pool) ---")
    print(f"Train accuracy: {train_accuracy:.2%}")
    print("\n--- Validation metrics ---")
    print(f"Accuracy: {val_accuracy:.2%}")
    print(f"Balanced accuracy: {balanced_accuracy:.2%}")
    print("F1 per class:")
    for c in range(n_classes):
        print(f"  {CLASS_NAMES[c]}: {f1_scores[c]:.4f}")
    print(f"  Macro F1: {macro_f1:.4f}")
    print("Confusion matrix (rows=actual, cols=predicted):")
    for actual in range(n_classes):
        row = [int(np.sum((y_val == actual) & (y_val_pred == p))) for p in range(n_classes)]
        print(f"  {CLASS_NAMES[actual]}: {row}")

    # Test set evaluation
    y_test_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    test_accuracy = np.mean(y_test_pred == y_test)
    test_confusion = [[int(np.sum((y_test == a) & (y_test_pred == p))) for p in range(n_classes)] for a in range(n_classes)]
    print("\n--- Test set (held-out subjects) ---")
    print(f"Test accuracy: {test_accuracy:.2%}")
    for actual in range(n_classes):
        print(f"  {CLASS_NAMES[actual]}: {test_confusion[actual]}")

    # Inference latency
    import time
    warmup = X_val[:3]
    model.predict(warmup, verbose=0)
    t0 = time.perf_counter()
    for _ in range(50):
        model.predict(X_val[:1], verbose=0)
    latency_ms = (time.perf_counter() - t0) / 50 * 1000
    print(f"\nInference latency: {latency_ms:.2f} ms per sample")

    # Save to eval_results.json
    eval_path = BACKEND_DIR / "eval_results.json"
    existing = []
    if eval_path.exists():
        try:
            with open(eval_path) as f:
                existing = json.load(f)
        except Exception:
            existing = []
    entry = {
        "model": "3-class CNN-LSTM",
        "train_accuracy": float(train_accuracy),
        "accuracy": float(val_accuracy),
        "test_accuracy": float(test_accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "macro_f1": float(macro_f1),
        "f1_rest": float(f1_scores[0]),
        "f1_left": float(f1_scores[1]),
        "f1_right": float(f1_scores[2]),
        "latency_ms": round(latency_ms, 2),
        "test_confusion_matrix": [[int(x) for x in row] for row in test_confusion],
        "n_test_subjects": len(test_subjects),
        "n_test_trials": int(len(y_test)),
    }
    other = [e for e in existing if e.get("model") != "3-class CNN-LSTM"]
    with open(eval_path, "w") as f:
        json.dump(other + [entry], f, indent=2)
    print(f"Wrote metrics to {eval_path}")

    out_path = BACKEND_DIR / "saved_model_3class.keras"
    model.save(out_path)
    print(f"\nSaved model to {out_path}")
    print("To use: export MODEL_PATH=backend/saved_model_3class.keras ; python3 app.py")


if __name__ == "__main__":
    main()
