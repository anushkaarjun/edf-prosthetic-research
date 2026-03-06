"""
Load validation data from BCI Competition-style EDF folder.

Preprocessing (applied when loading):
  - Filter the signals to keep only 8–30 Hz (the range that matters for imagining movement).
  - Ensure every trial has exactly 64 channels and the same length (4.2 s at 160 Hz = 672 samples):
    pad short trials and trim long ones.
  - For 2-class: keep only Rest and Right hand trials. For 3-class: keep Rest, Left hand, Right hand.
  - In the end you have one array of trials: (number of trials, 64 channels, 672 time points).

Expects: DATA_DIR with subject folders S001, S002, ... each containing
  SxxxR03.edf .. SxxxR14.edf and SxxxR03.edf.event ... (R01/R02 are calibration).
Event file format: "T0 duration: 4.2" (rest), "T1" (left), "T2" (right), "T3" (feet), "T4" (tongue).
Maps to our 5 classes: 0=Left Hand, 1=Right Hand, 2=Both Feet, 3=Both Fists, 4=Rest.
"""
import os
import re
import warnings
from pathlib import Path

import numpy as np

# Map event codes to our class indices (0-4)
EVENT_TO_CLASS = {"T0": 4, "T1": 0, "T2": 1, "T3": 2, "T4": 3}


def _parse_event_file(path, fs=160):
    """Parse .edf.event file. Returns list of (start_sample, length_samples, class_idx)."""
    events = []
    try:
        with open(path, "rb") as f:
            text = f.read().decode("utf-8", errors="ignore")
    except Exception:
        return events
    # Optional: read "time resolution: 160" from first line
    res = re.search(r"time\s+resolution:\s*(\d+)", text, re.I)
    if res:
        fs = int(res.group(1))
    pattern = re.compile(r"T([0-4])\s+duration:\s*([\d.]+)", re.IGNORECASE)
    pos = 0
    for m in pattern.finditer(text):
        tid = "T" + m.group(1)
        dur_sec = float(m.group(2))
        n_samp = max(50, int(dur_sec * fs))
        class_idx = EVENT_TO_CLASS.get(tid, 4)
        events.append((pos, n_samp, class_idx))
        pos += n_samp
    return events


def _load_raw_edf(edf_path, n_channels=64, bandpass_hz=(8, 30), sfreq=160):
    """Load full EDF into (n_channels, n_times). Optionally bandpass filter for motor imagery (mu/beta). Returns None on failure."""
    try:
        import mne
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # MNE: "annotation(s) expanding outside the data range"
            raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
        if bandpass_hz and sfreq > 2 * bandpass_hz[1]:
            raw.filter(l_freq=bandpass_hz[0], h_freq=bandpass_hz[1], verbose=False)
        data, _ = raw.get_data(return_times=True)
        if data.shape[0] > n_channels:
            data = data[:n_channels]
        elif data.shape[0] < n_channels:
            pad = np.zeros((n_channels - data.shape[0], data.shape[1]), dtype=data.dtype)
            data = np.concatenate([data, pad], axis=0)
        return data.astype(np.float64)
    except Exception:
        return None


# Fixed trial length so all segments stack (same shape). 4.2 sec at 160 Hz = 672
FIXED_TRIAL_LEN = 672


def load_validation_from_edf_dir(data_dir, max_subjects=3, max_trials_per_run=20, max_total_trials=2000, fs=160, n_channels=64, classes_to_load=None):
    """
    Load validation trials from EDF folder. Returns (data, labels, subject_ids).
    data: (N, n_channels, T), labels: (N,) class indices 0-4, subject_ids: (N,) int subject index per trial.
    All segments padded/cropped to same T.
    classes_to_load: if set (e.g. [1, 4]), only load events with these class indices (4=Rest, 1=Right Hand).
    Pass max_subjects, max_trials_per_run, or max_total_trials=None to use all available data (no cap).
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        return None, None, None

    trial_len = FIXED_TRIAL_LEN
    all_data = []
    all_labels = []
    all_subject_ids = []
    subject_folders = sorted([
        f for f in data_dir.iterdir()
        if f.is_dir() and f.name.startswith("S") and f.name[1:].isdigit()
    ], key=lambda x: int(x.name[1:]))
    if max_subjects is not None:
        subject_folders = subject_folders[:max_subjects]

    for subj_idx, subj_dir in enumerate(subject_folders):
        for run in range(3, 15):
            if max_total_trials is not None and len(all_data) >= max_total_trials:
                break
            rname = f"{subj_dir.name}R{run:02d}"
            edf_path = subj_dir / f"{rname}.edf"
            event_path = subj_dir / f"{rname}.edf.event"
            if not edf_path.exists():
                continue

            events = _parse_event_file(str(event_path), fs=fs)
            raw_data = _load_raw_edf(edf_path, n_channels=n_channels, bandpass_hz=(8, 30), sfreq=fs)
            if raw_data is None:
                continue

            if not events:
                n_times = raw_data.shape[1]
                cap = (trial_len * max_trials_per_run) if max_trials_per_run is not None else (n_times - trial_len)
                for start in range(0, min(n_times - trial_len, cap), trial_len):
                    if max_total_trials is not None and len(all_data) >= max_total_trials:
                        break
                    seg = raw_data[:, start : start + trial_len].copy()
                    if seg.shape[1] < trial_len:
                        seg = np.pad(seg, ((0, 0), (0, trial_len - seg.shape[1])), mode="edge")
                    all_data.append(seg)
                    all_labels.append(4)
                    all_subject_ids.append(subj_idx)
                continue

            for start_samp, n_samp, class_idx in events:
                if max_total_trials is not None and len(all_data) >= max_total_trials:
                    break
                if n_samp < 50:
                    continue
                if classes_to_load is not None and class_idx not in classes_to_load:
                    continue
                end = min(start_samp + n_samp, raw_data.shape[1])
                seg = raw_data[:, start_samp:end]
                # Pad or crop to fixed length so all segments have same shape
                if seg.shape[1] >= trial_len:
                    seg = seg[:, :trial_len].copy()
                else:
                    seg = np.pad(seg, ((0, 0), (0, trial_len - seg.shape[1])), mode="edge")
                all_data.append(seg)
                all_labels.append(class_idx)
                all_subject_ids.append(subj_idx)
                if max_trials_per_run is not None and len(all_data) >= max_trials_per_run * 4:
                    break
        if max_total_trials is not None and len(all_data) >= max_total_trials:
            break

    if not all_data:
        return None, None, None
    X = np.stack(all_data, axis=0)
    y = np.array(all_labels, dtype=np.int64)
    subject_ids = np.array(all_subject_ids, dtype=np.int64)
    return X, y, subject_ids


def _to_three_class(y_5):
    """Map 5-class to 3-class: 4->0 (Rest), 0->1 (Left Hand), 1->2 (Right Hand). Others -> -1."""
    y_5 = np.asarray(y_5, dtype=np.int64)
    out = np.full_like(y_5, -1)
    out[y_5 == 4] = 0
    out[y_5 == 0] = 1
    out[y_5 == 1] = 2
    return out


def load_test_from_edf_dir(data_dir, random_state=42, fs=160, n_channels=64, classes_to_load=(0, 1, 4)):
    """
    Load only the TEST set (same 60/20/20 subject split as train_model_3class.py).
    Use this so the app shows probabilities on held-out test data.
    Returns (X_test, y_test) where y_test is 3-class (0=Rest, 1=Left Hand, 2=Right Hand).
    """
    X, y_5, subject_ids = load_validation_from_edf_dir(
        data_dir, max_subjects=None, max_trials_per_run=None, max_total_trials=None,
        fs=fs, n_channels=n_channels, classes_to_load=list(classes_to_load) if classes_to_load else None,
    )
    if X is None or len(X) < 2:
        return None, None
    unique_subjects = np.unique(subject_ids)
    if len(unique_subjects) < 3:
        return None, None
    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        return None, None
    train_val_subjects, test_subjects = train_test_split(
        unique_subjects, test_size=0.2, random_state=random_state
    )
    if len(test_subjects) == 0:
        test_subjects = unique_subjects[-1:]
    mask_test = np.isin(subject_ids, test_subjects)
    X_test = X[mask_test]
    y_5_test = y_5[mask_test]
    y_test = _to_three_class(y_5_test)
    # Drop trials that don't map to 3-class (shouldn't happen if classes_to_load=[0,1,4])
    keep = y_test >= 0
    if not np.any(keep):
        return None, None
    X_test = X_test[keep]
    y_test = y_test[keep]
    return X_test, y_test
