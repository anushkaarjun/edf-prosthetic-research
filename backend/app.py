"""
Flask API for EEG Motor Imagery CNN-LSTM.
- Serves validation data and runs inference with your saved model.
- Set MODEL_PATH and VALIDATION_DATA_PATH (see below) to use your files.
"""
import os
import json
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Paths: set in env or place files in backend/
# Visualization uses the 3-class CNN-LSTM (Rest, Left Hand, Right Hand). Default model:
# saved_model_3class.keras — same model trained by train_model_3class.py
_BACKEND_DIR = Path(__file__).resolve().parent
MODEL_PATH = os.environ.get("MODEL_PATH", "")
_default_model_3class = _BACKEND_DIR / "saved_model_3class.keras"
if not MODEL_PATH:
    MODEL_PATH = str(_default_model_3class.resolve())
VISUALIZATION_MODEL_PATH = MODEL_PATH  # Model used for /api/predict and UI visualization
VALIDATION_DATA_PATH = os.environ.get("VALIDATION_DATA_PATH", "")
VALIDATION_LABELS_PATH = os.environ.get("VALIDATION_LABELS_PATH", "")

# DATA_DIR: try env var first, then fallback to known dataset paths
def _resolve_data_dir():
    if os.environ.get("DATA_DIR"):
        p = Path(os.environ["DATA_DIR"])
        if p.is_dir():
            return str(p.resolve())
    # Fallback: Prosthetic Research Data "files 2" (BCI-style EDF)
    fallbacks = [
        Path.home() / "Desktop" / "Outside of School" / "Prosethic Research Data" / "files 2",
        Path.home() / "Desktop" / "Outside of School" / "Prosthetic Research Data" / "files 2",
        _BACKEND_DIR / "dataset",
        _BACKEND_DIR.parent / "dataset",
    ]
    for p in fallbacks:
        if p.is_dir():
            return str(p.resolve())
    return ""

DATA_DIR = _resolve_data_dir()
if DATA_DIR:
    print(f"[Data] Using dataset: {DATA_DIR}")

MOTOR_CLASSES_2 = ["Rest", "Right Hand"]
MOTOR_CLASSES_3 = ["Rest", "Left Hand", "Right Hand"]
# API uses 3-class CNN-LSTM: probabilities always Rest, Left Hand, Right Hand
n_classes = 3
MOTOR_CLASSES = MOTOR_CLASSES_3
model = None
val_data = None
val_labels = None
val_labels_are_3class = False  # True when loaded via load_test_from_edf_dir (test set)
val_index = [0]


def load_model():
    global model, n_classes, MOTOR_CLASSES
    path = Path(MODEL_PATH).resolve() if MODEL_PATH else None
    if not path or not path.exists():
        return False
    try:
        import tensorflow as tf
        from model_cnn_lstm import balanced_accuracy_3class
        model = tf.keras.models.load_model(
            str(path),
            custom_objects={"balanced_accuracy_3class": balanced_accuracy_3class},
        )
        # Infer 2 vs 3 class from output shape
        out_shape = model.output.shape
        n_out = int(out_shape[-1]) if out_shape else 2
        if n_out == 3:
            n_classes = 3
            MOTOR_CLASSES = MOTOR_CLASSES_3
            print(f"[Model] Loaded 3-class CNN-LSTM from {path} (used for visualization)")
        else:
            n_classes = 2
            MOTOR_CLASSES = MOTOR_CLASSES_2
            print(f"[Model] Loaded 2-class CNN-LSTM from {path} (API will still return 3-class probability format)")
        load_validation_data()
        return True
    except Exception as e:
        print(f"[Model] Keras load failed: {e}")
    try:
        import torch
        model = torch.load(str(path), map_location="cpu")
        if hasattr(model, "eval"):
            model.eval()
        # Assume 2-class for PyTorch unless path suggests 3-class
        if "3class" in str(path).lower():
            n_classes = 3
            MOTOR_CLASSES = MOTOR_CLASSES_3
        else:
            n_classes = 2
            MOTOR_CLASSES = MOTOR_CLASSES_2
        load_validation_data()
        print(f"[Model] Loaded PyTorch model from {path}")
        return True
    except Exception as e:
        print(f"[Model] PyTorch load failed: {e}")
    return False


def load_validation_data():
    global val_data, val_labels, val_labels_are_3class
    val_labels_are_3class = False
    try:
        import numpy as np
    except ImportError:
        return
    # 1) EDF folder: load TEST set (same 60/20/20 split as training) so UI shows probabilities on held-out test data
    if DATA_DIR and Path(DATA_DIR).is_dir() and n_classes == 3:
        try:
            from data_loader_edf import load_test_from_edf_dir
            random_state = int(os.environ.get("RANDOM_STATE", "42"))
            X_test, y_test = load_test_from_edf_dir(DATA_DIR, random_state=random_state, fs=160, n_channels=64, classes_to_load=(0, 1, 4))
            if X_test is not None and len(X_test) > 0:
                val_data = X_test
                val_labels = y_test
                val_labels_are_3class = True
                print(f"[Data] Loaded TEST set: {len(X_test)} trials (same split as training, RANDOM_STATE={random_state})")
                return
        except Exception as e:
            print("[Data] Test set loader failed:", e)
    # 1b) EDF folder fallback: all data (if test load failed or 2-class)
    if DATA_DIR and Path(DATA_DIR).is_dir():
        try:
            from data_loader_edf import load_validation_from_edf_dir
            kwargs = {"max_subjects": None, "max_trials_per_run": None, "max_total_trials": None, "fs": 160, "n_channels": 64}
            if n_classes == 3:
                kwargs["classes_to_load"] = [0, 1, 4]
            out = load_validation_from_edf_dir(DATA_DIR, **kwargs)
            val_data, val_labels = out[0], out[1]
            if val_data is not None and len(val_data) > 0:
                return
        except Exception as e:
            print("EDF loader failed:", e)
        val_data = None
        val_labels = None
        return
    # 2) NumPy files
    if VALIDATION_DATA_PATH and Path(VALIDATION_DATA_PATH).exists():
        try:
            val_data = np.load(VALIDATION_DATA_PATH)
            if val_data.ndim == 2:
                val_data = val_data[np.newaxis, ...]
        except Exception:
            val_data = None
    if VALIDATION_LABELS_PATH and Path(VALIDATION_LABELS_PATH).exists():
        try:
            val_labels = np.load(VALIDATION_LABELS_PATH)
        except Exception:
            val_labels = None
    if val_data is not None and val_labels is None:
        val_labels = np.zeros(len(val_data), dtype=int)


def _to_two_class(label_5):
    """Map 5-class EDF label to 2-class: 4=Rest -> 0, 0,1,2,3=Motor -> 1."""
    return 0 if label_5 == 4 else 1


def _to_three_class(label_5):
    """Map 5-class to 3-class: 4->0 (Rest), 0->1 (Left Hand), 1->2 (Right Hand)."""
    if label_5 == 4:
        return 0
    if label_5 == 0:
        return 1
    if label_5 == 1:
        return 2
    return -1


def get_next_validation_sample():
    """Return one sample (64, T) and class label index (2-class or 3-class)."""
    import numpy as np
    global val_index
    if val_data is None or len(val_data) == 0:
        n_time = 672  # match model input so /api/predict gets valid shape
        mock = np.random.randn(64, n_time).astype(float) * 20
        return mock.tolist(), 0, "Rest"
    idx = val_index[0] % len(val_data)
    val_index[0] += 1
    sample = val_data[idx]
    if sample.ndim == 2 and sample.shape[0] != 64:
        sample = sample.T
    if val_labels_are_3class and val_labels is not None:
        label_idx = int(val_labels[idx])
        label_idx = min(max(0, label_idx), len(MOTOR_CLASSES) - 1)
        label_name = MOTOR_CLASSES[label_idx]
    else:
        label_5 = int(val_labels[idx]) if val_labels is not None else 4
        if n_classes == 3:
            label_idx = _to_three_class(label_5)
            if label_idx < 0:
                label_idx = 0
            label_name = MOTOR_CLASSES[label_idx]
        else:
            label_idx = _to_two_class(label_5)
            label_name = MOTOR_CLASSES[label_idx]
    return sample.tolist(), label_idx, label_name


def _sanitize_float(x):
    """Replace NaN/Inf with 0.0 so JSON response is always valid."""
    import math
    try:
        v = float(x)
        if not math.isfinite(v):
            return 0.0
        return v
    except (TypeError, ValueError):
        return 0.0


def _zscore_trial(x):
    """Per-channel z-score (same as in training) so inference matches training distribution."""
    import numpy as np
    out = np.array(x, dtype=np.float32)
    if out.ndim == 2:
        for c in range(out.shape[0]):
            seg = out[c, :]
            std = np.std(seg)
            if std > 1e-8:
                out[c, :] = (seg - np.mean(seg)) / std
    return out


def predict_keras(eeg):
    import numpy as np
    x = np.array(eeg, dtype=np.float32)
    if x.ndim == 2:
        x = x[np.newaxis, ...]
    # Ensure (batch, 64, 672): channels=64, time=672
    if x.shape[1] != 64 and x.shape[2] == 64:
        x = np.transpose(x, (0, 2, 1))
    n_ch, n_time = x.shape[1], x.shape[2]
    if n_ch != 64 or n_time != 672:
        # Pad or crop to (1, 64, 672) so model always gets expected shape
        target_t = 672
        target_c = 64
        out = np.zeros((x.shape[0], target_c, target_t), dtype=np.float32)
        out[:, : min(target_c, n_ch), : min(target_t, n_time)] = x[:, : min(target_c, n_ch), : min(target_t, n_time)]
        x = out
    # Match training: per-trial per-channel z-score
    x = np.array([_zscore_trial(x[i]) for i in range(x.shape[0])], dtype=np.float32)
    if not np.isfinite(x).all():
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    # Model expects (batch, 64, 672); log shape once per 100 requests if debugging
    probs = model.predict(x, verbose=0)[0]
    probs = np.asarray(probs, dtype=np.float64)
    if not np.isfinite(probs).all():
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    # If model outputs 5 classes, map to 2: Rest = class 4, Motor = sum(0,1,2,3)
    if len(probs) == 5 and n_classes == 2:
        rest_p = float(probs[4])
        motor_p = float(probs[0] + probs[1] + probs[2] + probs[3])
        probs = np.array([rest_p, motor_p], dtype=np.float32)
    probs = np.asarray(probs).flatten()[:n_classes]
    pred_idx = int(np.argmax(probs))
    return probs.tolist(), pred_idx


def predict_pytorch(eeg):
    import numpy as np
    import torch
    x = np.array(eeg, dtype=np.float32)
    if x.ndim == 2:
        x = x[np.newaxis, ...]
    if x.shape[1] != 64:
        x = np.transpose(x, (0, 2, 1))
    # Match training: per-trial per-channel z-score
    x = np.array([_zscore_trial(x[i]) for i in range(x.shape[0])], dtype=np.float32)
    t = torch.from_numpy(x).float()
    with torch.no_grad():
        logits = model(t)
    if hasattr(logits, "numpy"):
        probs = torch.softmax(logits, dim=-1).numpy()[0]
    else:
        probs = np.asarray(logits.cpu().numpy() if hasattr(logits, "cpu") else logits).flatten()
    if len(probs) == 5 and n_classes == 2:
        rest_p = float(probs[4])
        motor_p = float(probs[0] + probs[1] + probs[2] + probs[3])
        probs = np.array([rest_p, motor_p], dtype=np.float32)
    probs = np.asarray(probs).flatten()[:n_classes]
    pred_idx = int(np.argmax(probs))
    return probs.tolist(), pred_idx


def predict_mock(eeg):
    """No model: alternate classes for demo."""
    import random
    if n_classes == 3:
        i = random.randint(0, 2)
        probs = [0.1, 0.1, 0.1]
        probs[i] = 0.7
        return probs, i
    if random.random() < 0.5:
        return [0.45, 0.55], 1  # Right Hand
    return [0.55, 0.45], 0  # Rest


@app.route("/api/status")
def status():
    return jsonify({
        "connected": True,
        "modelLoaded": model is not None,
        "modelName": "3-class CNN-LSTM" if (model is not None and n_classes == 3) else (None if model is None else "2-class CNN-LSTM"),
        "classes": MOTOR_CLASSES,
        "nClasses": n_classes,
    })


@app.route("/api/metrics")
def metrics():
    """Return evaluation metrics and latency for all models (from eval_results.json)."""
    eval_path = _BACKEND_DIR / "eval_results.json"
    if not eval_path.exists():
        return jsonify([])
    try:
        with open(eval_path) as f:
            data = json.load(f)
        return jsonify(data if isinstance(data, list) else [])
    except Exception:
        return jsonify([])


def _predict_for_sample(eeg_list):
    """Run model on the same eeg we serve (no round-trip). Returns (probs_3, pred_idx_3) for UI."""
    three_class_names = MOTOR_CLASSES_3
    if model is None:
        import random
        i = random.randint(0, 2)
        probs_3 = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        probs_3[i] = 0.7
        return probs_3, i
    try:
        if hasattr(model, "predict"):
            probs, pred_idx = predict_keras(eeg_list)
        else:
            probs, pred_idx = predict_pytorch(eeg_list)
    except Exception:
        probs, pred_idx = predict_mock(eeg_list)
    if n_classes == 3:
        probs_3 = list(probs)[:3]
        pred_idx_3 = pred_idx
    else:
        probs_3 = [float(probs[0]) if len(probs) > 0 else 0, 0, float(probs[1]) if len(probs) > 1 else 0]
        pred_idx_3 = 0 if pred_idx == 0 else 2
    probs_3 = (probs_3 + [0] * 3)[:3]
    probs_3 = [_sanitize_float(p) for p in probs_3]
    s = sum(probs_3)
    if not s or s <= 0:
        probs_3 = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        pred_idx_3 = 0
    else:
        probs_3 = [p / s for p in probs_3]
    pred_idx_3 = min(max(0, pred_idx_3), 2)
    return probs_3, pred_idx_3


@app.route("/api/validation/sample")
def validation_sample():
    eeg, actual_idx, actual_name = get_next_validation_sample()
    # Run prediction on the same data we serve so UI gets real probabilities (no round-trip)
    probs_3, pred_idx_3 = _predict_for_sample(eeg)
    order_for_ui = [1, 2, 0]  # Left Hand, Right Hand, Rest
    probs_for_ui = [probs_3[i] for i in order_for_ui]
    three_class_names = MOTOR_CLASSES_3
    resp = jsonify({
        "eeg": eeg,
        "actualClassIndex": actual_idx,
        "actualClass": actual_name,
        "classes": MOTOR_CLASSES,
        "probabilities": dict(zip(three_class_names, probs_3)),
        "probs": probs_for_ui,
        "predictedClass": pred_idx_3,
        "predictedClassName": three_class_names[pred_idx_3],
        "radarValues": [p * 100 for p in probs_for_ui],
    })
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.route("/api/predict", methods=["POST"])
def predict():
    body = request.get_json(silent=True) or {}
    eeg = list(body.get("eeg") or [])
    if not eeg:
        return jsonify({"error": "Missing eeg"}), 400

    if model is not None:
        try:
            if hasattr(model, "predict"):
                probs, pred_idx = predict_keras(eeg)
            else:
                probs, pred_idx = predict_pytorch(eeg)
        except Exception as e:
            import sys
            print(f"[Predict] Exception: {e}", file=sys.stderr)
            probs, pred_idx = predict_mock(eeg)
    else:
        probs, pred_idx = predict_mock(eeg)

    # Always return 3-class probabilities (Rest, Left Hand, Right Hand) for the UI
    three_class_names = MOTOR_CLASSES_3
    if n_classes == 3:
        probs_3 = list(probs)[:3]
        pred_idx_3 = pred_idx
    else:
        # 2-class model: [Rest, Right Hand] -> pad Left Hand as 0
        probs_3 = [float(probs[0]) if len(probs) > 0 else 0, 0, float(probs[1]) if len(probs) > 1 else 0]
        pred_idx_3 = 0 if pred_idx == 0 else 2  # Rest -> 0, Right Hand -> 2
    probs_3 = (probs_3 + [0] * 3)[:3]
    # Sanitize so NaN/Inf never get into JSON (invalid JSON otherwise)
    raw_probs = list(probs_3)
    probs_3 = [_sanitize_float(p) for p in probs_3]
    s = sum(probs_3)
    if not s or s <= 0:
        import sys
        n_ch = len(eeg) if eeg else 0
        n_time = len(eeg[0]) if (eeg and eeg[0] is not None) else 0
        print(f"[Predict] Model output sum=0; raw probs={raw_probs}; input eeg: {n_ch} ch x {n_time} time (need 64 x 672). Using uniform 33.33%.", file=sys.stderr)
        probs_3 = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        pred_idx_3 = 0
    else:
        probs_3 = [p / s for p in probs_3]
    pred_idx_3 = min(max(0, pred_idx_3), 2)  # clamp to valid index
    # Frontend expects order [Left Hand, Right Hand, Rest]; backend model order is [Rest, Left Hand, Right Hand]
    order_for_ui = [1, 2, 0]  # Left Hand, Right Hand, Rest
    probs_for_ui = [probs_3[i] for i in order_for_ui]
    return jsonify({
        "probabilities": dict(zip(three_class_names, probs_3)),
        "probs": probs_for_ui,
        "predictedClass": pred_idx_3,
        "predictedClassName": three_class_names[pred_idx_3],
        "radarValues": [p * 100 for p in probs_for_ui],
        "classes": three_class_names,
    })


if __name__ == "__main__":
    load_validation_data()
    loaded = load_model()
    if not loaded:
        print("[Model] No model loaded. Run: python3 train_model_3class.py  (with DATA_DIR set), then restart.")
    port = int(os.environ.get("PORT", 5001))
    print(f"[API] Starting on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
