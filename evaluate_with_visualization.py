#!/usr/bin/env python3
"""Comprehensive evaluation script with bad data cleaning, visualization, and confidence metrics."""

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

# Import from our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from edf_ml_model.data_utils import annotation_to_motion, get_run_number
from edf_ml_model.inference import predict_with_confidence
from edf_ml_model.preprocessing import (
    clean_data,
    create_half_second_epochs,
    detect_bad_data,
    preprocess_raw,
)
from edf_ml_model.visualization import (
    visualize_motor_movement,
)

# Configuration
DEFAULT_BASE_PATH = os.path.expanduser("~/Desktop/Prosthetic Research Data/files 2")
MAX_SUBJECTS = 5


def load_and_evaluate_with_cleaning(
    subject_path: str, model_dict: dict, clean_bad_data: bool = True
) -> dict:
    """Load subject data with bad data cleaning and evaluate with confidence metrics.

    Args:
        subject_path: Path to subject directory
        model_dict: Dictionary containing model and related info
        clean_bad_data: Whether to detect and clean bad channels

    Returns:
        Dictionary with evaluation results including confidence metrics
    """
    subject_id = os.path.basename(subject_path)
    subj_files = sorted(glob.glob(f"{subject_path}/*.edf"))

    all_predictions = []
    all_true_labels = []
    all_confidences = []
    all_probabilities = []
    quality_stats = {
        "total_files": len(subj_files),
        "processed_files": 0,
        "total_bad_channels": 0,
        "total_epochs": 0,
    }

    for file_path in subj_files:
        run = get_run_number(file_path)

        try:
            # Load raw data
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

            # Detect bad data if enabled
            if clean_bad_data:
                quality_info = detect_bad_data(raw)
                quality_stats["total_bad_channels"] += len(quality_info["bad_channels"])

                if len(quality_info["bad_channels"]) > 0:
                    print(
                        f"  File {os.path.basename(file_path)}: Found {len(quality_info['bad_channels'])} bad channels"
                    )
                    print(f"    Quality score: {quality_info['quality_score']:.2%}")
                    raw = clean_data(raw, quality_info["bad_channels"])

            # Preprocess (filter, resample, normalize)
            raw_preprocessed, metadata = preprocess_raw(
                raw, apply_filter=True, clean=clean_bad_data, normalize=True
            )

            # Extract events and create epochs
            events, event_id = mne.events_from_annotations(
                raw_preprocessed, verbose=False
            )
            if len(event_id) == 0:
                continue

            epochs = create_half_second_epochs(
                raw_preprocessed, events, event_id, tmin=-0.5, tmax=0.5
            )

            if len(epochs) == 0:
                continue

            quality_stats["total_epochs"] += len(epochs)
            quality_stats["processed_files"] += 1

            # Get data and labels
            X = epochs.get_data()  # (n_epochs, n_channels, n_samples)
            y_raw = epochs.events[:, -1]
            y_labels = [annotation_to_motion(c, run) for c in y_raw]

            # Filter out "Unknown" labels
            valid_idx = [i for i, v in enumerate(y_labels) if v != "Unknown"]
            if len(valid_idx) == 0:
                continue

            X_valid = X[valid_idx]
            y_valid = [y_labels[i] for i in valid_idx]

            # Get model components
            model = model_dict["model"]
            label_map = model_dict["label_map"]
            unique_labels = model_dict["unique_labels"]

            # Reverse label map (integer -> string)
            idx_to_label = {idx: label for label, idx in label_map.items()}

            # Predict with confidence for each epoch
            for epoch_data, true_label in zip(X_valid, y_valid):
                # Convert to (n_channels, n_samples) format
                epoch_data_2d = epoch_data  # Already (n_channels, n_samples)

                # Predict with confidence
                pred_idx, confidence, probabilities = predict_with_confidence(
                    model, epoch_data_2d, device="cpu"
                )

                pred_label = idx_to_label[pred_idx]

                all_predictions.append(pred_label)
                all_true_labels.append(true_label)
                all_confidences.append(confidence)
                all_probabilities.append(probabilities)

        except Exception as e:
            print(f"  Error processing {os.path.basename(file_path)}: {e}")
            continue

    # Compute metrics
    if len(all_predictions) == 0:
        return None

    accuracy = accuracy_score(all_true_labels, all_predictions)
    mean_confidence = np.mean(all_confidences)
    median_confidence = np.median(all_confidences)

    # Confidence by prediction correctness
    correct_mask = np.array(all_predictions) == np.array(all_true_labels)
    correct_confidences = np.array(all_confidences)[correct_mask]
    incorrect_confidences = np.array(all_confidences)[~correct_mask]

    results = {
        "subject_id": subject_id,
        "accuracy": accuracy,
        "n_samples": len(all_predictions),
        "mean_confidence": mean_confidence,
        "median_confidence": median_confidence,
        "correct_mean_confidence": np.mean(correct_confidences)
        if len(correct_confidences) > 0
        else 0.0,
        "incorrect_mean_confidence": np.mean(incorrect_confidences)
        if len(incorrect_confidences) > 0
        else 0.0,
        "predictions": all_predictions,
        "true_labels": all_true_labels,
        "confidences": all_confidences,
        "probabilities": all_probabilities,
        "quality_stats": quality_stats,
        "unique_labels": unique_labels,
    }

    return results


def plot_confusion_matrix_with_confidence(
    y_true, y_pred, confidences, label_names, subject_id: str, save_path: str = None
):
    """Plot confusion matrix with confidence information.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        confidences: Confidence scores for each prediction
        label_names: List of label names
        subject_id: Subject identifier
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=label_names)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=axes[0],
    )
    axes[0].set_title(f"Confusion Matrix - Subject {subject_id}")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    # Normalized confusion matrix
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=axes[1],
    )
    axes[1].set_title(f"Normalized Confusion Matrix - Subject {subject_id}")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved confusion matrix to: {save_path}")
    else:
        plt.show()


def plot_confidence_analysis(results: dict, save_path: str = None):
    """Plot confidence metrics analysis.

    Args:
        results: Results dictionary from evaluation
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Confidence Analysis - Subject {results['subject_id']}", fontsize=14)

    confidences = np.array(results["confidences"])
    correct_mask = np.array(results["predictions"]) == np.array(results["true_labels"])

    # 1. Confidence distribution
    axes[0, 0].hist(confidences, bins=30, alpha=0.7, edgecolor="black")
    axes[0, 0].axvline(
        results["mean_confidence"],
        color="r",
        linestyle="--",
        label=f"Mean: {results['mean_confidence']:.3f}",
    )
    axes[0, 0].axvline(
        results["median_confidence"],
        color="g",
        linestyle="--",
        label=f"Median: {results['median_confidence']:.3f}",
    )
    axes[0, 0].set_xlabel("Confidence Score")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Confidence Distribution")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Confidence by correctness
    correct_conf = confidences[correct_mask]
    incorrect_conf = confidences[~correct_mask]

    axes[0, 1].boxplot([correct_conf, incorrect_conf], labels=["Correct", "Incorrect"])
    axes[0, 1].set_ylabel("Confidence Score")
    axes[0, 1].set_title("Confidence by Prediction Correctness")
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    # 3. Confidence vs accuracy by class
    unique_labels = results["unique_labels"]
    class_confidences = {label: [] for label in unique_labels}
    class_correct = {label: [] for label in unique_labels}

    for pred, true, conf in zip(
        results["predictions"], results["true_labels"], confidences
    ):
        class_confidences[pred].append(conf)
        class_correct[pred].append(pred == true)

    class_names = []
    mean_confs = []
    accuracies = []

    for label in unique_labels:
        if len(class_confidences[label]) > 0:
            class_names.append(label)
            mean_confs.append(np.mean(class_confidences[label]))
            accuracies.append(np.mean(class_correct[label]))

    x = np.arange(len(class_names))
    width = 0.35

    axes[1, 0].bar(x - width / 2, mean_confs, width, label="Mean Confidence", alpha=0.7)
    axes[1, 0].bar(x + width / 2, accuracies, width, label="Accuracy", alpha=0.7)
    axes[1, 0].set_xlabel("Class")
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].set_title("Confidence and Accuracy by Class")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(class_names, rotation=45, ha="right")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    # 4. Confidence threshold analysis
    thresholds = np.arange(0.3, 1.0, 0.05)
    accuracies_at_threshold = []
    n_samples_at_threshold = []

    for threshold in thresholds:
        mask = confidences >= threshold
        if np.sum(mask) > 0:
            acc_at_thresh = np.mean(correct_mask[mask])
            accuracies_at_threshold.append(acc_at_thresh)
            n_samples_at_threshold.append(np.sum(mask))
        else:
            accuracies_at_threshold.append(0.0)
            n_samples_at_threshold.append(0)

    ax2 = axes[1, 1]
    ax2_twin = ax2.twinx()

    line1 = ax2.plot(
        thresholds, accuracies_at_threshold, "b-o", label="Accuracy", linewidth=2
    )
    line2 = ax2_twin.plot(
        thresholds, n_samples_at_threshold, "r-s", label="N Samples", linewidth=2
    )

    ax2.set_xlabel("Confidence Threshold")
    ax2.set_ylabel("Accuracy", color="b")
    ax2_twin.set_ylabel("Number of Samples", color="r")
    ax2.set_title("Accuracy vs Confidence Threshold")
    ax2.grid(True, alpha=0.3)

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc="upper left")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved confidence analysis to: {save_path}")
    else:
        plt.show()


def visualize_sample_predictions(
    results: dict, n_samples: int = 6, save_path: str = None
):
    """Visualize sample predictions with motor movement visualization.

    Args:
        results: Results dictionary from evaluation
        n_samples: Number of samples to visualize
        save_path: Path to save figure (optional)
    """
    # Select samples with different confidence levels
    confidences = np.array(results["confidences"])
    predictions = results["predictions"]
    true_labels = results["true_labels"]

    # Select diverse samples (mix of correct/incorrect, high/low confidence)
    indices = np.arange(len(predictions))
    np.random.seed(42)  # For reproducibility
    selected_indices = np.random.choice(
        indices, size=min(n_samples, len(predictions)), replace=False
    )

    n_cols = 3
    n_rows = (n_samples + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    fig.suptitle(f"Sample Predictions - Subject {results['subject_id']}", fontsize=14)

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, ax_idx in enumerate(selected_indices):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]

        pred_label = predictions[ax_idx]
        true_label = true_labels[ax_idx]
        confidence = confidences[ax_idx]
        is_correct = pred_label == true_label

        # Visualize motor movement
        visualize_motor_movement(pred_label, confidence, ax=ax)

        # Add additional info
        color = "green" if is_correct else "red"
        status = "✓ CORRECT" if is_correct else "✗ INCORRECT"
        title = (
            f"{status}\nPred: {pred_label}\nTrue: {true_label}\nConf: {confidence:.2%}"
        )
        ax.set_title(title, fontsize=10, color=color, fontweight="bold")

    # Hide unused subplots
    for idx in range(len(selected_indices), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved sample visualizations to: {save_path}")
    else:
        plt.show()


def main(base_path=None, model_path=None, clean_bad_data=True):
    """Main function to evaluate models with cleaning, visualization, and confidence metrics.

    Args:
        base_path: Path to data directory
        model_path: Path to saved model (currently loads from run_csp_svm.py)
        clean_bad_data: Whether to detect and clean bad channels
    """
    if base_path is None:
        base_path = DEFAULT_BASE_PATH

    # For now, we'll use CSP+SVM models (can be extended to neural network models)
    # In a real scenario, you would load the trained model here
    print("=" * 70)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("=" * 70)
    print(f"Data path: {base_path}")
    print(f"Bad data cleaning: {'Enabled' if clean_bad_data else 'Disabled'}")
    print()

    # This is a placeholder - in practice, you would load your trained model
    # For demonstration, we'll show the structure
    print("Note: This script demonstrates the evaluation framework.")
    print("To use with actual models, load them from training scripts.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate models with bad data cleaning, visualization, and confidence metrics"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help=f"Path to data directory (default: {DEFAULT_BASE_PATH})",
    )
    parser.add_argument(
        "--model-path", type=str, default=None, help="Path to saved model"
    )
    parser.add_argument(
        "--no-clean", action="store_true", help="Disable bad data cleaning"
    )

    args = parser.parse_args()

    main(
        base_path=args.data_path,
        model_path=args.model_path,
        clean_bad_data=not args.no_clean,
    )
