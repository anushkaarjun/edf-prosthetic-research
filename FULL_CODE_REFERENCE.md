# EEG Simulator UI – Full Code Reference

All main project files in one place. Paths are relative to `eeg-simulator-ui-2/`.

---

## backend/train_model_3class.py

```python
#!/usr/bin/env python3
"""
Train the 3-class CNN-LSTM on your EDF data: Rest, Left Hand, Right Hand.
Usage: export DATA_DIR="/path/to/files 2" && python3 train_model_3class.py
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
    "/Users/anushkaarjun/Desktop/Outside of School/Prosethic Research Data",
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
    print("DATA_DIR must point to the folder containing S001, S002, etc.")
    print('  export DATA_DIR="/full/path/to/your/files 2"')
    print("  python3 train_model_3class.py")
    sys.exit(1)

try:
    import tensorflow as tf
except ImportError:
    print("pip3 install tensorflow")
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

    print("Loading EDF data from", DATA_DIR)
    X, y_5, subject_ids = load_validation_from_edf_dir(
        DATA_DIR, max_subjects=None, max_trials_per_run=None, max_total_trials=None, fs=160, n_channels=64,
        classes_to_load=[0, 1, 4],
    )
    if X is None or len(X) < 20:
        print("Not enough data.")
        sys.exit(1)

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
        print("Need at least 3 subjects.")
        sys.exit(1)

    from sklearn.model_selection import train_test_split
    train_val_subjects, test_subjects = train_test_split(unique_subjects, test_size=0.2, random_state=random_state)
    if len(test_subjects) == 0:
        test_subjects = unique_subjects[-1:]
        train_val_subjects = unique_subjects[:-1]
    train_subjects, val_subjects = train_test_split(train_val_subjects, test_size=0.25, random_state=random_state)
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
    print(f"Train: {len(y_train_pool)}, Val: {len(y_val)}, Test: {len(y_test)}")

    counts_pool = {c: int(np.sum(y_train_pool == c)) for c in range(n_classes)}
    if counts_pool[1] == 0 or counts_pool[2] == 0:
        print("Try RANDOM_STATE=0 or 1 and run again.")
        sys.exit(1)

    max_count = max(counts_pool.values())
    balance_idx = []
    for c in range(n_classes):
        idx = np.where(y_train_pool == c)[0]
        repeat = (max_count + len(idx) - 1) // len(idx) if len(idx) > 0 else 0
        oversampled = np.tile(idx, repeat)[:max_count] if repeat > 0 else idx
        balance_idx.append(oversampled)
    rs = np.random.RandomState(random_state)
    for _ in range(2):
        for c in [1, 2]:
            idx = np.where(y_train_pool == c)[0]
            if len(idx) > 0:
                balance_idx.append(rs.choice(idx, size=max_count, replace=True))
    balance_idx = np.concatenate(balance_idx)
    rs.shuffle(balance_idx)
    X_train_pool = X_train_pool[balance_idx]
    y_train_pool = y_train_pool[balance_idx]

    total = len(y_train_pool)
    class_weight = {c: total / (n_classes * max(1, int(np.sum(y_train_pool == c)))) for c in range(n_classes)}
    motor_boost = 5.0
    class_weight[1] = class_weight.get(1, 1.0) * motor_boost
    class_weight[2] = class_weight.get(2, 1.0) * motor_boost
    for c in range(n_classes):
        class_weight[c] = min(float(class_weight[c]), 10.0)

    X_train, y_train = X_train_pool, y_train_pool

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
            rs = np.random.RandomState(self.seed + self.epoch * 997 + idx)
            shift = rs.randint(-80, 81)
            if shift != 0:
                for i in range(X_batch.shape[0]):
                    X_batch[i] = np.roll(X_batch[i], shift, axis=1)
            X_batch += rs.randn(*X_batch.shape).astype(X_batch.dtype) * 0.01
            return X_batch, self.y[batch_idx]

    batch_size = 32
    train_seq = BalancedBatchSequence(X_train, y_train, batch_size, n_classes=3, shuffle=True, seed=random_state)

    model = build_model_3class(n_channels=64, n_times=X.shape[2])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4, clipnorm=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", balanced_accuracy_3class],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_balanced_accuracy_3class", patience=22, restore_best_weights=True, mode="max"),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_balanced_accuracy_3class", factor=0.5, patience=6, min_lr=1e-6, mode="max"),
    ]
    model.fit(train_seq, validation_data=(X_val, y_val), epochs=80, class_weight=class_weight, callbacks=callbacks, verbose=1)

    y_train_pred = np.argmax(model.predict(X_train, verbose=0), axis=1)
    train_accuracy = np.mean(y_train_pred == y_train)
    y_val_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    val_accuracy = np.mean(y_val_pred == y_val)
    recalls = [np.sum((y_val == c) & (y_val_pred == c)) / max(1, np.sum(y_val == c)) for c in range(n_classes)]
    balanced_accuracy = np.mean(recalls)

    y_test_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    test_accuracy = np.mean(y_test_pred == y_test)
    test_confusion = [[int(np.sum((y_test == a) & (y_test_pred == p))) for p in range(n_classes)] for a in range(n_classes)]

    print(f"Train: {train_accuracy:.2%}, Val: {val_accuracy:.2%}, Test: {test_accuracy:.2%}, Balanced: {balanced_accuracy:.2%}")

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
        "test_confusion_matrix": [[int(x) for x in row] for row in test_confusion],
        "n_test_subjects": len(test_subjects),
        "n_test_trials": int(len(y_test)),
    }
    other = [e for e in existing if e.get("model") != "3-class CNN-LSTM"]
    with open(eval_path, "w") as f:
        json.dump(other + [entry], f, indent=2)

    out_path = BACKEND_DIR / "saved_model_3class.keras"
    model.save(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
```

---

## backend/model_cnn_lstm.py

```python
"""3-class CNN-LSTM: input (batch, 64, 672), output 3 probs (Rest, Left Hand, Right Hand)."""
import numpy as np


def to_two_class(labels_5):
    return np.where(np.asarray(labels_5) == 4, 0, 1)


def to_three_class(labels_5):
    labels = np.asarray(labels_5)
    out = np.zeros_like(labels, dtype=np.int64)
    out[labels == 4] = 0
    out[labels == 0] = 1
    out[labels == 1] = 2
    return out


def balanced_accuracy_3class(y_true, y_pred):
    import tensorflow as tf
    n_classes = 3
    y_pred_class = tf.argmax(y_pred, axis=-1)
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int64)
    y_pred_class = tf.reshape(y_pred_class, [-1])
    recalls = []
    for c in range(n_classes):
        in_c = tf.equal(y_true, c)
        n_c = tf.maximum(tf.reduce_sum(tf.cast(in_c, tf.float32)), 1.0)
        correct_c = tf.reduce_sum(tf.cast(tf.logical_and(in_c, tf.equal(y_pred_class, c)), tf.float32))
        recalls.append(correct_c / n_c)
    return tf.reduce_mean(recalls)


def build_model_3class(n_channels=64, n_times=672):
    import tensorflow as tf
    reg = tf.keras.regularizers.L2(4e-5)
    inp = tf.keras.layers.Input(shape=(n_channels, n_times))
    x = tf.keras.layers.Permute((2, 1))(inp)
    x = tf.keras.layers.Conv1D(64, 12, padding="same", activation="relu", kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool1D(2)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv1D(128, 6, padding="same", activation="relu", kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool1D(2)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv1D(128, 4, padding="same", activation="relu", kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool1D(2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.2)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=reg)(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    out = tf.keras.layers.Dense(3, activation="softmax")(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy", balanced_accuracy_3class])
    return model
```

---

## backend/data_loader_edf.py

See the repo file; it contains:
- `load_validation_from_edf_dir()` – load all EDF trials (N, 64, 672), labels 0–4, subject_ids
- `load_test_from_edf_dir()` – same 60/20/20 subject split as training, returns (X_test, y_test) 3-class

---

## backend/app.py

See the repo file; it contains:
- `load_model()` – load saved_model_3class.keras
- `load_validation_data()` – prefer test set from EDF (load_test_from_edf_dir), fallback to all EDF or NPY
- `get_next_validation_sample()` – return one (64, 672) sample + actual label
- `_predict_for_sample(eeg_list)` – run model on that sample, return probs + pred
- `GET /api/validation/sample` – returns eeg, actualClass, probabilities, predictedClass, etc.
- `POST /api/predict` – optional; same model inference on body.eeg
- `GET /api/status`, `GET /api/metrics`

---

## src/constants.js

```javascript
export const MOTOR_CLASSES = ['Rest', 'Left Hand', 'Right Hand']
export const TOTAL_CHANNELS = 64
export const CHANNELS_DISPLAYED = 4
export const SAMPLE_RATE = 160
export const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:5001'
```

---

## src/api/client.js

```javascript
import { API_BASE } from '../constants'

export async function getStatus() {
  const res = await fetch(`${API_BASE}/api/status`)
  if (!res.ok) throw new Error('API not reachable')
  return res.json()
}

export async function getValidationSample() {
  const res = await fetch(`${API_BASE}/api/validation/sample?t=${Date.now()}`, {
    cache: 'no-store',
    headers: { Pragma: 'no-cache', 'Cache-Control': 'no-cache' },
  })
  if (!res.ok) throw new Error('Failed to get validation sample')
  return res.json()
}

export async function predict(eegSegment) {
  const res = await fetch(`${API_BASE}/api/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ eeg: eegSegment }),
    cache: 'no-store',
  })
  if (!res.ok) throw new Error('Predict failed')
  return res.json()
}

export async function getMetrics() {
  const res = await fetch(`${API_BASE}/api/metrics`)
  if (!res.ok) return []
  return res.json()
}
```

---

## src/App.jsx

```jsx
import EEGSimulator from './components/EEGSimulator'
import './App.css'

function App() {
  return (
    <div className="app">
      <EEGSimulator />
    </div>
  )
}

export default App
```

---

## src/components/EEGSimulator.jsx

- Calls `getValidationSample()` on “Next sample” / “Run validation”.
- Uses `sample.eeg`, `sample.actualClass`, `sample.probabilities` / `sample.probs`, `sample.predictedClass` from the API (no separate predict call).
- Renders: status bar, EEG line chart (4 channels), Motor Function Probability bars, Actual/Predicted class, Confidence Radar (Recharts).
- Buttons: “Run validation” (auto-advance), “Next sample”.

---

## How to run

```bash
# Terminal 1 – backend (from project root)
cd eeg-simulator-ui-2/backend && python3 app.py

# Terminal 2 – frontend
cd eeg-simulator-ui-2 && npm install && npm run dev
```

Optional: `export DATA_DIR="/path/to/Prosethic Research Data/files 2"` so the app loads test-set EDF data.

Open **http://localhost:5173**.

To train first:

```bash
cd eeg-simulator-ui-2/backend
export DATA_DIR="/path/to/files 2"
python3 train_model_3class.py
```

Then run `app.py` and the frontend as above.
