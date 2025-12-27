"""Enhanced visualization functions for classification results and spectral analysis."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle


def plot_spectral_analysis(spectral_info: dict, ax: plt.Axes | None = None):
    """Plot power spectral density before/after filtering."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    freqs = spectral_info["freqs"]
    psd = spectral_info["psd"]  # (n_channels, n_freqs)
    band_range = spectral_info["band_range"]

    # Average across channels
    psd_mean = np.mean(psd, axis=0)
    psd_std = np.std(psd, axis=0)

    ax.plot(freqs, psd_mean, label="Mean PSD")
    ax.fill_between(freqs, psd_mean - psd_std, psd_mean + psd_std, alpha=0.3)

    # Highlight motor imagery band
    band_idx = np.where((freqs >= band_range[0]) & (freqs <= band_range[1]))[0]
    ax.axvspan(
        band_range[0],
        band_range[1],
        alpha=0.2,
        color="green",
        label="Motor Imagery Band (8-30 Hz)",
    )

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power Spectral Density")
    ax.set_title("Spectral Analysis - Bandpass Filter Validation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_classification_with_confidence(
    predicted_label: str,
    confidence: float,
    probabilities: np.ndarray,
    label_names: list[str],
    ax: plt.Axes | None = None,
):
    """Visualize classification result with confidence bar."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Bar plot of probabilities
    colors_list = [
        "green" if label == predicted_label else "gray" for label in label_names
    ]
    bars = ax.barh(label_names, probabilities, color=colors_list)

    # Add confidence annotation
    ax.axvline(
        x=confidence,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Confidence: {confidence:.2%}",
    )

    ax.set_xlabel("Probability")
    ax.set_xlim(0, 1)
    ax.set_title(
        f"Classification Result: {predicted_label}\nConfidence: {confidence:.2%}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")

    return ax


def visualize_motor_movement(
    predicted_label: str,
    confidence: float | None = None,
    ax: plt.Axes | None = None,
):
    """Create 2D visualization of predicted motor movement with confidence."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # Draw body outline
    head = Circle((0, 0.7), 0.15, fill=False, color="black", linewidth=2)
    ax.add_patch(head)

    body = Rectangle((-0.1, -0.3), 0.2, 0.6, fill=False, color="black", linewidth=2)
    ax.add_patch(body)

    left_arm = Rectangle((-0.4, 0.2), 0.3, 0.1, fill=False, color="gray", linewidth=2)
    right_arm = Rectangle((0.1, 0.2), 0.3, 0.1, fill=False, color="gray", linewidth=2)
    ax.add_patch(left_arm)
    ax.add_patch(right_arm)

    left_leg = Rectangle((-0.15, -0.9), 0.1, 0.6, fill=False, color="gray", linewidth=2)
    right_leg = Rectangle((0.05, -0.9), 0.1, 0.6, fill=False, color="gray", linewidth=2)
    ax.add_patch(left_leg)
    ax.add_patch(right_leg)

    # Highlight predicted movement
    color = "red"
    alpha = 0.7
    if confidence is not None:
        alpha = max(0.3, confidence)  # Transparency based on confidence

    if "Left Hand" in predicted_label:
        left_hand = Circle((-0.25, 0.25), 0.08, fill=True, color=color, alpha=alpha)
        ax.add_patch(left_hand)
    elif "Right Hand" in predicted_label:
        right_hand = Circle((0.25, 0.25), 0.08, fill=True, color=color, alpha=alpha)
        ax.add_patch(right_hand)
    elif "Both Fists" in predicted_label:
        left_hand = Circle((-0.25, 0.25), 0.08, fill=True, color=color, alpha=alpha)
        right_hand = Circle((0.25, 0.25), 0.08, fill=True, color=color, alpha=alpha)
        ax.add_patch(left_hand)
        ax.add_patch(right_hand)
    elif "Both Feet" in predicted_label:
        left_foot = Circle((-0.1, -0.9), 0.08, fill=True, color=color, alpha=alpha)
        right_foot = Circle((0.1, -0.9), 0.08, fill=True, color=color, alpha=alpha)
        ax.add_patch(left_foot)
        ax.add_patch(right_foot)

    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-1.0, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")

    title = f"Predicted: {predicted_label}"
    if confidence is not None:
        title += f"\nConfidence: {confidence:.2%}"
    ax.set_title(title, fontsize=12, fontweight="bold")

    return ax
