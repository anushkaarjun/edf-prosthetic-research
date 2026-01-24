"""
Unified preprocessing functions for EEG data.
Reusable across training, validation, and inference.
"""
import numpy as np
import mne
from scipy import stats, signal
from typing import Tuple, Optional, Dict, Any

# Configuration constants
TARGET_SFREQ = 250  # Hz
FREQ_LOW, FREQ_HIGH = 8.0, 30.0  # Motor imagery band
EPOCH_WINDOW_MIN = 0.1  # Minimum chunk size: 0.1 seconds
EPOCH_WINDOW_MAX = 0.5  # Maximum chunk size: 0.5 seconds
EPOCH_WINDOW = 0.5  # Default: 0.5 seconds for training (backward compatibility)
BASELINE_WINDOW = (-0.5, 0)  # Baseline period
INFERENCE_BUFFER_SECONDS = 0.5  # Use small chunks (0.5s) instead of 30s buffer
SMOOTHING_WINDOW_SIZE = 5  # Number of samples for moving average smoothing


def smooth_signal(
    X: np.ndarray, 
    window_size: int = SMOOTHING_WINDOW_SIZE,
    method: str = "moving_average"
) -> np.ndarray:
    """
    Apply smoothing/averaging to reduce noise in EEG signals.
    
    Args:
        X: Signal data (n_channels, n_samples) or (n_epochs, n_channels, n_samples)
        window_size: Size of smoothing window in samples
        method: 'moving_average' for simple moving average, 'gaussian' for Gaussian smoothing
    
    Returns:
        Smoothed signal with same shape as input
    """
    X = X.copy()
    original_shape = X.shape
    was_2d = len(X.shape) == 2
    
    if was_2d:
        X = X[np.newaxis, ...]
    
    n_epochs, n_channels, n_samples = X.shape
    X_smooth = np.zeros_like(X)
    
    if method == "moving_average":
        # Simple moving average using convolution
        kernel = np.ones(window_size) / window_size
        for i in range(n_epochs):
            for j in range(n_channels):
                # Pad edges to maintain length
                padded = np.pad(X[i, j, :], (window_size // 2, window_size // 2), mode='edge')
                X_smooth[i, j, :] = np.convolve(padded, kernel, mode='valid')
    
    elif method == "gaussian":
        # Gaussian smoothing using scipy
        from scipy.ndimage import gaussian_filter1d
        for i in range(n_epochs):
            for j in range(n_channels):
                X_smooth[i, j, :] = gaussian_filter1d(
                    X[i, j, :], 
                    sigma=window_size / 3.0  # Convert window size to sigma
                )
    
    if was_2d:
        X_smooth = X_smooth.squeeze(0)
    
    return X_smooth


def normalize_signal(X: np.ndarray, method: str = "minmax") -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Normalize signals to [-1, 1] range for consistent weight scaling.
    
    Args:
        X: Signal data (n_channels, n_samples) or (n_epochs, n_channels, n_samples)
        method: 'minmax' for [-1,1] scaling, 'zscore' for zero-mean unit-variance
    
    Returns:
        Normalized data and normalization parameters (for inverse transform)
    """
    X = X.copy()
    original_shape = X.shape
    
    if len(X.shape) == 2:
        X = X[np.newaxis, ...]
    
    # Compute global min/max across all channels and epochs
    X_flat = X.reshape(-1, X.shape[-1])
    global_min = np.min(X_flat)
    global_max = np.max(X_flat)
    
    if method == "minmax":
        # Scale to [-1, 1]
        if global_max - global_min > 1e-10:  # Avoid division by zero
            X_norm = 2 * (X - global_min) / (global_max - global_min) - 1
        else:
            X_norm = X * 0  # All zeros
        norm_params = {"method": "minmax", "min": global_min, "max": global_max}
    else:  # zscore
        global_mean = np.mean(X_flat)
        global_std = np.std(X_flat)
        if global_std > 1e-10:
            X_norm = (X - global_mean) / global_std
            # Clip to [-1, 1] after zscore
            X_norm = np.clip(X_norm, -1, 1)
        else:
            X_norm = X * 0
        norm_params = {"method": "zscore", "mean": global_mean, "std": global_std}
    
    if len(original_shape) == 2:
        X_norm = X_norm.squeeze(0)
    
    return X_norm, norm_params


def compute_spectral_analysis(
    raw: mne.io.Raw, freq_range: Tuple[float, float] = (1, 50)
) -> Dict[str, Any]:
    """
    Perform spectral analysis to validate bandpass filtering.
    
    Args:
        raw: MNE Raw object
        freq_range: Frequency range for analysis
    
    Returns:
        Dictionary with spectral power information
    """
    # Compute power spectral density using scipy
    data = raw.get_data()
    sfreq = raw.info["sfreq"]
    n_fft = min(2048, int(sfreq * 2))
    
    # Compute PSD for each channel and average
    all_psd = []
    for channel_data in data:
        freqs, psd_ch = signal.welch(
            channel_data,
            fs=sfreq,
            nperseg=n_fft,
            noverlap=n_fft // 4,
            nfft=n_fft
        )
        all_psd.append(psd_ch)
    
    psd = np.array(all_psd)  # (n_channels, n_freqs)
    
    # Filter to frequency range
    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    freqs = freqs[freq_mask]
    psd = psd[:, freq_mask]
    
    # Extract power in motor imagery band
    band_idx = np.where((freqs >= FREQ_LOW) & (freqs <= FREQ_HIGH))[0]
    band_power = np.mean(psd[:, band_idx], axis=1)  # Average across frequencies
    
    # Total power
    total_power = np.mean(psd, axis=1)
    
    # Band power ratio
    band_ratio = band_power / (total_power + 1e-10)
    
    spectral_info = {
        "freqs": freqs,
        "psd": psd,
        "band_power": band_power,
        "total_power": total_power,
        "band_ratio": band_ratio,
        "band_range": (FREQ_LOW, FREQ_HIGH),
    }
    
    return spectral_info


def detect_bad_data(raw: mne.io.Raw, z_threshold: float = 3.0) -> Dict[str, Any]:
    """
    Detect bad channels and epochs using statistical methods.
    
    Args:
        raw: MNE Raw object
        z_threshold: Z-score threshold for outlier detection
    
    Returns:
        Dictionary with bad channel indices and data quality metrics
    """
    # Convert to numpy array
    data = raw.get_data()
    n_channels, n_samples = data.shape
    
    # Check for flat channels (zero variance)
    channel_stds = np.std(data, axis=1)
    flat_channels = np.where(channel_stds < 1e-10)[0].tolist()
    
    # Check for channels with excessive variance (outliers)
    z_scores = np.abs(stats.zscore(channel_stds))
    high_var_channels = np.where(z_scores > z_threshold)[0].tolist()
    
    # Check for NaN or Inf values
    nan_channels = []
    inf_channels = []
    for i in range(n_channels):
        if np.any(np.isnan(data[i, :])):
            nan_channels.append(i)
        if np.any(np.isinf(data[i, :])):
            inf_channels.append(i)
    
    bad_channels = list(
        set(flat_channels + high_var_channels + nan_channels + inf_channels)
    )
    
    # Compute data quality metrics
    quality_metrics = {
        "bad_channels": bad_channels,
        "flat_channels": flat_channels,
        "high_var_channels": high_var_channels,
        "nan_channels": nan_channels,
        "inf_channels": inf_channels,
        "mean_channel_std": float(np.mean(channel_stds)),
        "median_channel_std": float(np.median(channel_stds)),
        "quality_score": 1.0 - (len(bad_channels) / n_channels),  # 1.0 = perfect, 0.0 = all bad
    }
    
    return quality_metrics


def clean_data(raw: mne.io.Raw, bad_channels: Optional[list] = None) -> mne.io.Raw:
    """
    Clean data by removing bad channels and interpolating if needed.
    
    Args:
        raw: MNE Raw object
        bad_channels: List of bad channel indices (if None, auto-detect)
    
    Returns:
        Cleaned Raw object
    """
    raw = raw.copy()
    
    if bad_channels is None:
        quality_info = detect_bad_data(raw)
        bad_channels = quality_info["bad_channels"]
    
    if len(bad_channels) > 0:
        # Mark bad channels
        raw.info["bads"] = [
            raw.ch_names[i] for i in bad_channels if i < len(raw.ch_names)
        ]
        # Interpolate bad channels
        raw.interpolate_bads(reset_bads=True, verbose=False)
    
    return raw


def preprocess_raw(
    raw: mne.io.Raw,
    apply_filter: bool = True,
    clean: bool = True,
    normalize: bool = True,
    smooth: bool = True,
    smoothing_window: int = SMOOTHING_WINDOW_SIZE,
) -> Tuple[mne.io.Raw, Dict[str, Any]]:
    """
    Complete preprocessing pipeline for raw EEG data.
    
    Steps:
    1. Detect and clean bad data
    2. Apply bandpass filter (8-30 Hz)
    3. Resample to target frequency
    4. Apply smoothing to reduce noise
    5. Normalize to [-1, 1]
    6. Spectral analysis for validation
    
    Args:
        raw: MNE Raw object
        apply_filter: Whether to apply bandpass filter
        clean: Whether to clean bad channels
        normalize: Whether to normalize signals
        smooth: Whether to apply smoothing/averaging
        smoothing_window: Size of smoothing window in samples
    
    Returns:
        Preprocessed Raw object and preprocessing metadata
    """
    metadata = {}
    
    # Step 1: Clean bad data
    if clean:
        quality_info = detect_bad_data(raw)
        metadata["quality_info"] = quality_info
        raw = clean_data(raw, quality_info["bad_channels"])
    
    # Step 2: Apply bandpass filter
    if apply_filter:
        raw.filter(FREQ_LOW, FREQ_HIGH, verbose=False, fir_design="firwin")
        metadata["filter_applied"] = True
    else:
        metadata["filter_applied"] = False
    
    # Step 3: Resample if needed
    original_sfreq = raw.info["sfreq"]
    if original_sfreq != TARGET_SFREQ:
        raw.resample(TARGET_SFREQ, npad="auto", verbose=False)
        metadata["resampled"] = True
        metadata["original_sfreq"] = original_sfreq
    else:
        metadata["resampled"] = False
    
    # Step 4: Apply smoothing to reduce noise
    if smooth:
        data = raw.get_data()
        data_smooth = smooth_signal(data, window_size=smoothing_window, method="moving_average")
        raw._data = data_smooth
        metadata["smoothed"] = True
        metadata["smoothing_window"] = smoothing_window
    else:
        metadata["smoothed"] = False
    
    # Step 5: Spectral analysis
    spectral_info = compute_spectral_analysis(raw)
    metadata["spectral_info"] = spectral_info
    
    # Step 6: Normalize (this modifies the raw data)
    if normalize:
        data = raw.get_data()
        data_norm, norm_params = normalize_signal(data, method="minmax")
        raw._data = data_norm
        metadata["normalized"] = True
        metadata["norm_params"] = norm_params
    else:
        metadata["normalized"] = False
    
    return raw, metadata


def create_half_second_epochs(
    raw: mne.io.Raw,
    events: np.ndarray,
    event_id: dict,
    tmin: float = -0.5,
    tmax: float = 0.5,
    chunk_size: float = None,
) -> mne.Epochs:
    """
    Create epochs with configurable chunk size (0.1s to 0.5s) for training.
    
    Args:
        raw: Preprocessed Raw object
        events: Event array
        event_id: Event ID dictionary
        tmin: Start time relative to event (default -0.5 for baseline)
        tmax: End time relative to event (default 0.5 for 0.5s window)
        chunk_size: Size of chunk in seconds (0.1 to 0.5). If None, uses EPOCH_WINDOW.
    
    Returns:
        Epochs object with specified chunk size windows
    """
    # Use provided chunk_size or default
    if chunk_size is None:
        chunk_size = EPOCH_WINDOW
    else:
        # Clamp chunk_size to valid range
        chunk_size = max(EPOCH_WINDOW_MIN, min(EPOCH_WINDOW_MAX, chunk_size))
    
    # Create epochs with baseline period
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=BASELINE_WINDOW,
        preload=True,
        verbose=False,
    )
    
    # Crop to exactly chunk_size seconds (0 to chunk_size after event)
    epochs.crop(tmin=0, tmax=chunk_size)
    
    return epochs


def create_sliding_chunks(
    data: np.ndarray,
    sfreq: int,
    chunk_size: float = 0.5,
    overlap: float = 0.0,
) -> np.ndarray:
    """
    Create sliding window chunks from continuous data.
    
    Args:
        data: Continuous data (n_channels, n_samples)
        sfreq: Sampling frequency in Hz
        chunk_size: Size of each chunk in seconds (0.1 to 0.5)
        overlap: Overlap between chunks as fraction (0.0 to 0.9)
    
    Returns:
        Array of chunks (n_chunks, n_channels, n_samples_per_chunk)
    """
    # Clamp chunk_size to valid range
    chunk_size = max(EPOCH_WINDOW_MIN, min(EPOCH_WINDOW_MAX, chunk_size))
    overlap = max(0.0, min(0.9, overlap))
    
    n_channels, n_samples = data.shape
    samples_per_chunk = int(chunk_size * sfreq)
    step_size = int(samples_per_chunk * (1 - overlap))
    
    chunks = []
    for start in range(0, n_samples - samples_per_chunk + 1, step_size):
        chunk = data[:, start:start + samples_per_chunk]
        chunks.append(chunk)
    
    if len(chunks) == 0:
        # If data is shorter than one chunk, pad it
        if n_samples < samples_per_chunk:
            padded = np.zeros((n_channels, samples_per_chunk))
            padded[:, :n_samples] = data
            chunks = [padded]
        else:
            # Take the last chunk
            chunks = [data[:, -samples_per_chunk:]]
    
    return np.array(chunks)
