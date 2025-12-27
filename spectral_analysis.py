#!/usr/bin/env python3
"""Spectral analysis script to visualize frequency content of EEG data.
Helps determine optimal bandpass filter frequencies for motor imagery classification.
"""

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import mne
import numpy as np

# Import from our modules - reuse existing functions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
try:
    from edf_ml_model.preprocessing import (
        FREQ_HIGH,
        FREQ_LOW,
        TARGET_SFREQ,
        compute_spectral_analysis,
    )

    FUNCTIONS_AVAILABLE = True
except ImportError:
    # Fallback if modules not available (e.g., missing dependencies)
    print(
        "Warning: Could not import preprocessing functions. Using fallback implementations."
    )
    FREQ_LOW, FREQ_HIGH = 8.0, 30.0
    TARGET_SFREQ = 250
    FUNCTIONS_AVAILABLE = False

# Default data path (update as needed)
DEFAULT_BASE_PATH = os.path.expanduser("~/Desktop/Prosthetic Research Data/files 2")


def get_psd_from_spectral_info(raw: mne.io.Raw, freq_range: tuple = (1, 50)) -> tuple:
    """Get PSD using existing compute_spectral_analysis function.

    Args:
        raw: MNE Raw object
        freq_range: Frequency range to analyze (min_freq, max_freq)

    Returns:
        frequencies, psd_array (n_channels, n_freqs)
    """
    if FUNCTIONS_AVAILABLE:
        # Reuse existing function from preprocessing module
        spectral_info = compute_spectral_analysis(raw, freq_range=freq_range)
        freqs = spectral_info["freqs"]
        psd = spectral_info["psd"]  # (n_channels, n_freqs)
        return freqs, psd
    else:
        # Fallback implementation if imports failed
        from scipy import signal

        data = raw.get_data()
        sfreq = raw.info["sfreq"]
        nperseg = min(2048, int(sfreq * 2))

        all_psd = []
        all_freqs = None

        for channel_data in data:
            freqs, psd_ch = signal.welch(
                channel_data,
                fs=sfreq,
                nperseg=nperseg,
                noverlap=nperseg // 4,
                nfft=nperseg * 2,
            )

            freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
            freqs_filtered = freqs[freq_mask]
            psd_filtered = psd_ch[freq_mask]

            all_psd.append(psd_filtered)
            if all_freqs is None:
                all_freqs = freqs_filtered

        psd_array = np.array(all_psd)
        return all_freqs, psd_array


def plot_spectral_comparison(
    raw_before: mne.io.Raw,
    raw_after: mne.io.Raw,
    freq_low: float = FREQ_LOW,
    freq_high: float = FREQ_HIGH,
    subject_id: str = "",
    save_path: str = None,
):
    """Plot spectral analysis comparing before and after bandpass filtering.

    Args:
        raw_before: Raw data before filtering
        raw_after: Raw data after filtering
        freq_low: Lower frequency cutoff
        freq_high: Upper frequency cutoff
        subject_id: Subject identifier for title
        save_path: Path to save figure (optional)
    """
    # Compute PSD for both using existing function
    freqs_before, psd_before = get_psd_from_spectral_info(
        raw_before, freq_range=(1, 50)
    )
    freqs_after, psd_after = get_psd_from_spectral_info(raw_after, freq_range=(1, 50))

    # Compute mean across channels
    psd_before_mean = np.mean(psd_before, axis=0)
    psd_before_std = np.std(psd_before, axis=0)
    psd_after_mean = np.mean(psd_after, axis=0)
    psd_after_std = np.std(psd_after, axis=0)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Spectral Analysis - Subject {subject_id}\n"
        f"Bandpass Filter: {freq_low}-{freq_high} Hz",
        fontsize=14,
    )

    # Plot 1: Before filtering (full range)
    ax1 = axes[0, 0]
    ax1.plot(freqs_before, psd_before_mean, "b-", label="Mean PSD", linewidth=2)
    ax1.fill_between(
        freqs_before,
        psd_before_mean - psd_before_std,
        psd_before_mean + psd_before_std,
        alpha=0.3,
        color="blue",
    )
    ax1.axvspan(
        freq_low,
        freq_high,
        alpha=0.2,
        color="green",
        label=f"Motor Imagery Band ({freq_low}-{freq_high} Hz)",
    )
    ax1.axvline(freq_low, color="green", linestyle="--", linewidth=2)
    ax1.axvline(freq_high, color="green", linestyle="--", linewidth=2)
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Power Spectral Density")
    ax1.set_title("Before Filtering (Full Spectrum)")
    ax1.set_xlim(0, 50)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_yscale("log")

    # Plot 2: After filtering (full range)
    ax2 = axes[0, 1]
    ax2.plot(freqs_after, psd_after_mean, "r-", label="Mean PSD", linewidth=2)
    ax2.fill_between(
        freqs_after,
        psd_after_mean - psd_after_std,
        psd_after_mean + psd_after_std,
        alpha=0.3,
        color="red",
    )
    ax2.axvspan(
        freq_low,
        freq_high,
        alpha=0.2,
        color="green",
        label=f"Motor Imagery Band ({freq_low}-{freq_high} Hz)",
    )
    ax2.axvline(freq_low, color="green", linestyle="--", linewidth=2)
    ax2.axvline(freq_high, color="green", linestyle="--", linewidth=2)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Power Spectral Density")
    ax2.set_title("After Filtering (Full Spectrum)")
    ax2.set_xlim(0, 50)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_yscale("log")

    # Plot 3: Overlay comparison (focused on motor imagery band)
    ax3 = axes[1, 0]
    # Focus on 0-50 Hz for better visualization
    mask_before = freqs_before <= 50
    mask_after = freqs_after <= 50
    ax3.plot(
        freqs_before[mask_before],
        psd_before_mean[mask_before],
        "b-",
        label="Before Filtering",
        linewidth=2,
        alpha=0.7,
    )
    ax3.plot(
        freqs_after[mask_after],
        psd_after_mean[mask_after],
        "r-",
        label="After Filtering",
        linewidth=2,
        alpha=0.7,
    )
    ax3.axvspan(
        freq_low,
        freq_high,
        alpha=0.2,
        color="green",
        label=f"Motor Imagery Band ({freq_low}-{freq_high} Hz)",
    )
    ax3.axvline(freq_low, color="green", linestyle="--", linewidth=2)
    ax3.axvline(freq_high, color="green", linestyle="--", linewidth=2)
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_ylabel("Power Spectral Density")
    ax3.set_title("Comparison: Before vs After Filtering")
    ax3.set_xlim(0, 50)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_yscale("log")

    # Plot 4: Zoomed into motor imagery band
    ax4 = axes[1, 1]
    mask_before_band = (freqs_before >= freq_low - 5) & (freqs_before <= freq_high + 5)
    mask_after_band = (freqs_after >= freq_low - 5) & (freqs_after <= freq_high + 5)
    ax4.plot(
        freqs_before[mask_before_band],
        psd_before_mean[mask_before_band],
        "b-",
        label="Before Filtering",
        linewidth=2,
        alpha=0.7,
    )
    ax4.plot(
        freqs_after[mask_after_band],
        psd_after_mean[mask_after_band],
        "r-",
        label="After Filtering",
        linewidth=2,
        alpha=0.7,
    )
    ax4.axvspan(
        freq_low,
        freq_high,
        alpha=0.3,
        color="green",
        label=f"Motor Imagery Band ({freq_low}-{freq_high} Hz)",
    )
    ax4.axvline(freq_low, color="green", linestyle="--", linewidth=2)
    ax4.axvline(freq_high, color="green", linestyle="--", linewidth=2)
    ax4.set_xlabel("Frequency (Hz)")
    ax4.set_ylabel("Power Spectral Density")
    ax4.set_title(f"Zoomed: Motor Imagery Band ({freq_low}-{freq_high} Hz)")
    ax4.set_xlim(max(0, freq_low - 5), freq_high + 5)
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # Add annotations for frequency bands
    bands = {
        "Delta (0.5-4 Hz)": (0.5, 4, "purple"),
        "Theta (4-8 Hz)": (4, 8, "orange"),
        "Alpha (8-13 Hz)": (8, 13, "cyan"),
        "Beta (13-30 Hz)": (13, 30, "yellow"),
        "Gamma (30-50 Hz)": (30, 50, "magenta"),
    }

    for band_name, (low, high, color) in bands.items():
        if high <= 50:
            ax3.axvspan(low, high, alpha=0.1, color=color)
            if low >= freq_low - 5 and high <= freq_high + 5:
                ax4.axvspan(low, high, alpha=0.15, color=color)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved spectral plot to: {save_path}")
    else:
        plt.show()

    return fig


def analyze_subject(
    subject_path: str,
    freq_low: float = FREQ_LOW,
    freq_high: float = FREQ_HIGH,
    max_files: int = 3,
):
    """Perform spectral analysis on a subject's data.

    Args:
        subject_path: Path to subject directory
        freq_low: Lower frequency cutoff for bandpass filter
        freq_high: Upper frequency cutoff for bandpass filter
        max_files: Maximum number of files to analyze per subject
    """
    subject_id = os.path.basename(subject_path)
    edf_files = sorted(glob.glob(f"{subject_path}/*.edf"))[:max_files]

    if len(edf_files) == 0:
        print(f"  No EDF files found for {subject_id}")
        return

    print(f"\n{'=' * 60}")
    print(f"Analyzing Subject: {subject_id}")
    print(f"{'=' * 60}")
    print(f"  Files to analyze: {len(edf_files)}")
    print(f"  Filter range: {freq_low}-{freq_high} Hz")

    all_psd_before = []
    all_psd_after = []
    all_freqs = None

    for file_idx, file_path in enumerate(edf_files):
        filename = os.path.basename(file_path)
        print(f"\n  Processing file {file_idx + 1}/{len(edf_files)}: {filename}")

        try:
            # Load raw data
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            print(
                f"    Channels: {raw.info['nchan']}, Sampling rate: {raw.info['sfreq']:.1f} Hz"
            )

            # Resample if needed
            if raw.info["sfreq"] != TARGET_SFREQ:
                print(
                    f"    Resampling from {raw.info['sfreq']:.1f} Hz to {TARGET_SFREQ} Hz"
                )
                raw.resample(TARGET_SFREQ, npad="auto")

            # Compute PSD before filtering using existing function
            freqs, psd_before = get_psd_from_spectral_info(raw, freq_range=(1, 50))
            psd_before_mean = np.mean(psd_before, axis=0)
            all_psd_before.append(psd_before_mean)
            if all_freqs is None:
                all_freqs = freqs

            # Apply bandpass filter (using MNE's filter function)
            raw.filter(freq_low, freq_high, verbose=False, fir_design="firwin")
            print(f"    Applied {freq_low}-{freq_high} Hz bandpass filter")

            # Compute PSD after filtering using existing function
            freqs, psd_after = get_psd_from_spectral_info(raw, freq_range=(1, 50))
            psd_after_mean = np.mean(psd_after, axis=0)
            all_psd_after.append(psd_after_mean)

            # Calculate power in motor imagery band
            band_mask = (freqs >= freq_low) & (freqs <= freq_high)
            power_in_band_before = np.mean(psd_before_mean[band_mask])
            power_in_band_after = np.mean(psd_after_mean[band_mask])
            total_power_before = np.mean(psd_before_mean)
            total_power_after = np.mean(psd_after_mean)

            print(f"    Power in {freq_low}-{freq_high} Hz band:")
            print(
                f"      Before: {power_in_band_before:.2e} ({100 * power_in_band_before / total_power_before:.1f}% of total)"
            )
            print(
                f"      After:  {power_in_band_after:.2e} ({100 * power_in_band_after / total_power_after:.1f}% of total)"
            )

        except Exception as e:
            print(f"    Error processing {filename}: {e}")
            continue

    # Create average plot across all files
    if len(all_psd_before) > 0 and len(all_psd_after) > 0:
        print("\n  Creating spectral analysis plot...")

        # Use the first file for plotting (representative sample)
        # Load it fresh to get before/after comparison
        raw_before = mne.io.read_raw_edf(edf_files[0], preload=True, verbose=False)
        if raw_before.info["sfreq"] != TARGET_SFREQ:
            raw_before.resample(TARGET_SFREQ, npad="auto")

        # Create "after" raw (with filter)
        raw_after = raw_before.copy()
        raw_after.filter(freq_low, freq_high, verbose=False, fir_design="firwin")

        # Plot
        save_path = f"spectral_analysis_{subject_id}.png"
        plot_spectral_comparison(
            raw_before,
            raw_after,
            freq_low,
            freq_high,
            subject_id=subject_id,
            save_path=save_path,
        )


def main(base_path=None, freq_low=None, freq_high=None, max_subjects=3, max_files=3):
    """Main function to run spectral analysis.

    Args:
        base_path: Path to data directory
        freq_low: Lower frequency cutoff (default: from preprocessing config)
        freq_high: Upper frequency cutoff (default: from preprocessing config)
        max_subjects: Maximum number of subjects to analyze
        max_files: Maximum number of files per subject
    """
    if base_path is None:
        base_path = DEFAULT_BASE_PATH

    if freq_low is None:
        freq_low = FREQ_LOW
    if freq_high is None:
        freq_high = FREQ_HIGH

    if not os.path.exists(base_path):
        print(f"Error: Data path not found: {base_path}")
        print("Please specify the correct path using --data-path")
        return

    subjects = sorted(glob.glob(f"{base_path}/S*"))[:max_subjects]

    if len(subjects) == 0:
        print(f"No subject directories found in: {base_path}")
        return

    print("=" * 60)
    print("SPECTRAL ANALYSIS FOR BANDPASS FILTER OPTIMIZATION")
    print("=" * 60)
    print(f"Data path: {base_path}")
    print(f"Subjects to analyze: {len(subjects)}")
    print(f"Filter range: {freq_low}-{freq_high} Hz")
    print(f"Target sampling rate: {TARGET_SFREQ} Hz")
    print()

    for subject_path in subjects:
        analyze_subject(subject_path, freq_low, freq_high, max_files)

    print("\n" + "=" * 60)
    print("SPECTRAL ANALYSIS COMPLETE")
    print("=" * 60)
    print("\nInterpretation Guide:")
    print("  - The plots show power spectral density (PSD) before and after filtering")
    print(
        "  - Motor imagery signals are typically in the 8-30 Hz range (alpha + beta bands)"
    )
    print("  - After filtering, power outside the band should be significantly reduced")
    print(
        "  - If you see strong signals outside the motor imagery band, consider adjusting filter frequencies"
    )
    print("\nFrequency Bands:")
    print("  - Delta: 0.5-4 Hz (deep sleep)")
    print("  - Theta: 4-8 Hz (drowsiness)")
    print("  - Alpha: 8-13 Hz (relaxed awareness) ← Part of motor imagery")
    print("  - Beta: 13-30 Hz (active thinking, motor control) ← Part of motor imagery")
    print("  - Gamma: 30-50 Hz (focused attention)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform spectral analysis on EEG data to optimize bandpass filtering"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help=f"Path to data directory (default: {DEFAULT_BASE_PATH})",
    )
    parser.add_argument(
        "--freq-low",
        type=float,
        default=None,
        help=f"Lower frequency cutoff in Hz (default: {FREQ_LOW})",
    )
    parser.add_argument(
        "--freq-high",
        type=float,
        default=None,
        help=f"Upper frequency cutoff in Hz (default: {FREQ_HIGH})",
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=3,
        help="Maximum number of subjects to analyze (default: 3)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=3,
        help="Maximum number of files per subject to analyze (default: 3)",
    )

    args = parser.parse_args()

    main(
        base_path=args.data_path,
        freq_low=args.freq_low,
        freq_high=args.freq_high,
        max_subjects=args.max_subjects,
        max_files=args.max_files,
    )
