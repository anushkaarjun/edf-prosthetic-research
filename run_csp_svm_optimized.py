#!/usr/bin/env python3
"""Optimized CSP+SVM script with expanded hyperparameter search to achieve 80%+ accuracy.
Key improvements:
1. Expanded hyperparameter search space
2. More training data (smaller test set)
3. Better cross-validation
4. Option to try LDA as alternative
"""

import argparse
import glob
import os
import sys

import mne
import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Configuration
DEFAULT_BASE_PATH = "/content/drive/MyDrive/files 2"
MAX_SUBJECTS = 5
target_sfreq = 250
tmin, tmax = -0.5, 4
freq_low, freq_high = 8.0, 30.0
n_components = 8  # Default, but grid search will try more

# Import consolidated utility functions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from edf_ml_model.data_utils import annotation_to_motion, get_run_number


def main(base_path=None, use_lda=False, test_size=0.15):
    """Main function with optimized hyperparameter search."""
    if base_path is None:
        base_path = DEFAULT_BASE_PATH

    subjects = sorted(glob.glob(f"{base_path}/S*"))[:MAX_SUBJECTS]
    print(f"Using {len(subjects)} subjects: {[os.path.basename(s) for s in subjects]}")

    subject_data = {}
    for subj in subjects:
        print(f"\n=== Processing Subject {os.path.basename(subj)} ===")
        subj_files = sorted(glob.glob(f"{subj}/*.edf"))

        X_list = []
        y_list = []

        for file in subj_files:
            run = get_run_number(file)
            try:
                raw = mne.io.read_raw_edf(file, preload=True, verbose=False)
                raw.filter(freq_low, freq_high, verbose=False, fir_design="firwin")

                if raw.info["sfreq"] != target_sfreq:
                    raw.resample(target_sfreq, npad="auto")

                events, event_id = mne.events_from_annotations(raw, verbose=False)
                if len(event_id) == 0:
                    continue

                epochs = mne.Epochs(
                    raw,
                    events,
                    event_id=event_id,
                    tmin=tmin,
                    tmax=tmax,
                    baseline=(-0.5, 0),
                    preload=True,
                    verbose=False,
                )
                epochs.crop(tmin=0, tmax=4)

                if len(epochs) == 0:
                    continue

                X = epochs.get_data()
                y_raw = epochs.events[:, -1]
                y_mapped = [annotation_to_motion(c, run) for c in y_raw]
                valid_idx = [i for i, v in enumerate(y_mapped) if v != "Unknown"]

                if len(valid_idx) > 0:
                    X_list.append(X[valid_idx])
                    y_list += [y_mapped[i] for i in valid_idx]
            except Exception as e:
                print(f"  Error processing {file}: {e}")
                continue

        if len(X_list) > 0:
            X_subj = np.concatenate(X_list, axis=0).astype(np.float32)
            subject_data[os.path.basename(subj)] = {"X": X_subj, "y": y_list}
            print(f"  Processed {X_subj.shape[0]} epochs")

    if len(subject_data) == 0:
        print("ERROR: No data processed.")
        return None, None

    # Train models per subject
    subject_models = {}
    subject_results = {}

    # EXPANDED hyperparameter search space
    expanded_param_grid = {
        "csp__n_components": [8, 10, 12, 14, 16, 18, 20],  # More components
        "svm__C": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0],  # Wider range
        "svm__gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1.0],  # More options
    }

    for subj, data in subject_data.items():
        print(f"\n{'=' * 50}")
        print(f"Training OPTIMIZED model for Subject {subj}")
        print(f"{'=' * 50}")

        X = data["X"]
        y_labels = data["y"]

        unique_labels = sorted(set(y_labels))
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        y = np.array([label_map[v] for v in y_labels], dtype=np.int64)

        print(f"Data shape: {X.shape}, Classes: {unique_labels}")
        print("Class distribution:")
        for label, idx in label_map.items():
            count = np.sum(y == idx)
            print(f"  {label}: {count} ({100 * count / len(y):.1f}%)")

        # Use more training data (smaller test set)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        print(
            f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]} (test_size={test_size})"
        )

        if use_lda:
            # Try LDA instead of SVM (often better for motor imagery)
            print("\nUsing LDA classifier (often better for motor imagery)...")
            csp = CSP(
                n_components=20, reg=None, log=True, norm_trace=False
            )  # More components for LDA
            X_train_csp = csp.fit_transform(X_train, y_train)
            X_test_csp = csp.transform(X_test)

            lda = LDA()
            lda.fit(X_train_csp, y_train)
            train_acc = lda.score(X_train_csp, y_train)
            test_acc = lda.score(X_test_csp, y_test)

            print(f"  LDA Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")

            subject_models[subj] = {
                "model": lda,
                "csp": csp,
                "label_map": label_map,
                "unique_labels": unique_labels,
                "classifier_type": "LDA",
            }
            subject_results[subj] = {
                "train_acc": train_acc,
                "test_acc": test_acc,
                "y_test": y_test,
                "y_pred": lda.predict(X_test_csp),
            }
        else:
            # Use expanded GridSearchCV with SVM
            print("\nRunning EXPANDED GridSearchCV optimization...")
            pipeline = Pipeline([
                (
                    "csp",
                    CSP(
                        n_components=n_components, reg=None, log=True, norm_trace=False
                    ),
                ),
                ("svm", SVC(kernel="rbf", probability=True, random_state=42)),
            ])

            print(
                f"  Searching {len(expanded_param_grid['csp__n_components'])} × "
                f"{len(expanded_param_grid['svm__C'])} × "
                f"{len(expanded_param_grid['svm__gamma'])} = "
                f"{len(expanded_param_grid['csp__n_components']) * len(expanded_param_grid['svm__C']) * len(expanded_param_grid['svm__gamma'])} parameter combinations"
            )

            grid_search = GridSearchCV(
                pipeline,
                expanded_param_grid,
                cv=5,  # 5-fold CV instead of 3
                scoring="accuracy",
                n_jobs=-1,
                verbose=1,
            )
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            train_acc = best_model.score(X_train, y_train)
            test_acc = best_model.score(X_test, y_test)

            print(f"  Best parameters: {grid_search.best_params_}")
            print(f"  Optimized Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")

            csp = best_model.named_steps["csp"]
            svm = best_model.named_steps["svm"]
            X_test_csp = csp.transform(X_test)

            subject_models[subj] = {
                "model": svm,
                "csp": csp,
                "label_map": label_map,
                "unique_labels": unique_labels,
                "classifier_type": "SVM",
            }
            subject_results[subj] = {
                "train_acc": train_acc,
                "test_acc": test_acc,
                "y_test": y_test,
                "y_pred": svm.predict(X_test_csp),
            }

    # Overall results
    print(f"\n{'=' * 50}")
    print("OVERALL RESULTS (OPTIMIZED)")
    print(f"{'=' * 50}")
    overall_test_acc = np.mean([r["test_acc"] for r in subject_results.values()])
    overall_train_acc = np.mean([r["train_acc"] for r in subject_results.values()])
    print(
        f"Average Train Accuracy: {overall_train_acc:.4f} ({overall_train_acc * 100:.2f}%)"
    )
    print(
        f"Average Test Accuracy:  {overall_test_acc:.4f} ({overall_test_acc * 100:.2f}%)"
    )

    if overall_test_acc >= 0.80:
        print(
            f"\n✓✓✓ TARGET ACHIEVED! Average test accuracy: {overall_test_acc * 100:.2f}% (>= 80%) ✓✓✓"
        )
    else:
        print(f"\n⚠ Still below target (80%). Current: {overall_test_acc * 100:.2f}%")
        print("  Consider:")
        print("   1. Try LDA: --use-lda")
        print("   2. Use even more training data: --test-size 0.1")
        print("   3. Try more subjects for better averaging")

    # Detailed per-subject results
    print(f"\n{'=' * 50}")
    print("PER-SUBJECT DETAILS")
    print(f"{'=' * 50}")
    for subj, result in subject_results.items():
        print(f"\nSubject {subj}: Test Accuracy = {result['test_acc'] * 100:.2f}%")
        unique_labels = subject_models[subj]["unique_labels"]
        print(
            classification_report(
                result["y_test"], result["y_pred"], target_names=unique_labels, digits=3
            )
        )

    return overall_test_acc, subject_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OPTIMIZED CSP+SVM/LDA training")
    parser.add_argument(
        "--data-path", type=str, default=None, help="Path to EDF data directory"
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=5,
        help="Maximum number of subjects to process",
    )
    parser.add_argument(
        "--use-lda",
        action="store_true",
        help="Use LDA instead of SVM (often better for motor imagery)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Test set size (default 0.15 = more training data)",
    )

    args = parser.parse_args()
    BASE_PATH = args.data_path if args.data_path else DEFAULT_BASE_PATH
    MAX_SUBJECTS = args.max_subjects

    if os.path.exists("/content"):
        print("Running in Google Colab environment")
    else:
        print("Running locally")

    if not os.path.exists(BASE_PATH):
        print(f"\nERROR: Data path does not exist: {BASE_PATH}")
        print("\nPlease provide the correct path using --data-path")
        sys.exit(1)

    print(f"Using data path: {BASE_PATH}")
    print(f"Processing up to {MAX_SUBJECTS} subjects")
    if args.use_lda:
        print("Using LDA classifier (optimized for motor imagery)")
    else:
        print("Using SVM with expanded hyperparameter search")
    print(f"Test size: {args.test_size} (training data: {1 - args.test_size:.1%})")

    try:
        test_acc, results = main(
            BASE_PATH, use_lda=args.use_lda, test_size=args.test_size
        )
        if test_acc is not None:
            print(f"\n{'=' * 60}")
            print(f"FINAL RESULT: Average Test Accuracy = {test_acc * 100:.2f}%")
            print(f"{'=' * 60}")
        else:
            print("\nFailed to process data.")
            sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
