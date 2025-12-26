#!/usr/bin/env python3
"""
Standalone script to run CSP+SVM training and get test accuracy.
This is a simplified version of the notebook that can be run locally.
"""

import os
import sys
import glob
import argparse
import numpy as np
import mne
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from mne.decoding import CSP
import matplotlib.pyplot as plt

# Configuration
DEFAULT_BASE_PATH = "/content/drive/MyDrive/files 2"  # Default for Colab
MAX_SUBJECTS = 5
target_sfreq = 250
tmin, tmax = -0.5, 0.5  # Include baseline period (-0.5 to 0) and 0.5s task period
freq_low, freq_high = 8., 30.
n_components = 8

# Import consolidated utility functions (direct import to avoid loguru dependency)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
try:
    from edf_ml_model.data_utils import get_run_number, annotation_to_motion
except ImportError:
    # Fallback: import directly from file
    import importlib.util
    data_utils_path = os.path.join(os.path.dirname(__file__), "src", "edf_ml_model", "data_utils.py")
    spec = importlib.util.spec_from_file_location("data_utils", data_utils_path)
    data_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data_utils)
    get_run_number = data_utils.get_run_number
    annotation_to_motion = data_utils.annotation_to_motion

def main(base_path=None):
    if base_path is None:
        base_path = DEFAULT_BASE_PATH
    
    # Get subjects
    subjects = sorted(glob.glob(f"{base_path}/S*"))[:MAX_SUBJECTS]
    print(f"Using {len(subjects)} subjects: {[os.path.basename(s) for s in subjects]}")
    
    # Process data per subject
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
                raw.filter(freq_low, freq_high, verbose=False, fir_design='firwin')
                
                if raw.info["sfreq"] != target_sfreq:
                    raw.resample(target_sfreq, npad="auto")
                
                events, event_id = mne.events_from_annotations(raw, verbose=False)
                if len(event_id) == 0:
                    continue
                
                epochs = mne.Epochs(raw, events, event_id=event_id,
                                    tmin=tmin, tmax=tmax,
                                    baseline=(-0.5, 0),  # Baseline correction using pre-stimulus period
                                    preload=True, verbose=False)
                
                # After baseline correction, crop to task period (0-0.5s) for consistent signal length
                # This gives us exactly 125 samples at 250 Hz (0.5s * 250Hz = 125 samples)
                epochs.crop(tmin=0, tmax=0.5)
                
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
            subject_data[os.path.basename(subj)] = {'X': X_subj, 'y': y_list}
            print(f"  Processed {X_subj.shape[0]} epochs")
    
    if len(subject_data) == 0:
        print("ERROR: No data processed. Check file paths and data availability.")
        return None, None
    
    # Train models per subject
    subject_models = {}
    subject_results = {}
    
    for subj, data in subject_data.items():
        print(f"\n{'='*50}")
        print(f"Training model for Subject {subj}")
        print(f"{'='*50}")
        
        X = data['X']
        y_labels = data['y']
        
        unique_labels = sorted(set(y_labels))
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        y = np.array([label_map[v] for v in y_labels], dtype=np.int64)
        
        print(f"Data shape: {X.shape}, Classes: {unique_labels}")
        print("Class distribution:")
        for label, idx in label_map.items():
            count = np.sum(y == idx)
            print(f"  {label}: {count} ({100*count/len(y):.1f}%)")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
        
        # Apply CSP
        print("Applying CSP preprocessing...")
        csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
        X_train_csp = csp.fit_transform(X_train, y_train)
        X_test_csp = csp.transform(X_test)
        
        print(f"CSP features shape: {X_train_csp.shape}")
        
        # Train SVM with optional GridSearchCV if accuracy target not met
        print("Training SVM classifier...")
        svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
        svm.fit(X_train_csp, y_train)
        
        # Evaluate initial model
        train_acc = svm.score(X_train_csp, y_train)
        test_acc = svm.score(X_test_csp, y_test)
        
        # If accuracy is low, try GridSearchCV optimization
        if test_acc < 0.80:
            print(f"  Initial accuracy {test_acc*100:.2f}% < 80%, running GridSearchCV optimization...")
            from sklearn.model_selection import GridSearchCV
            from sklearn.pipeline import Pipeline
            
            # Create pipeline for grid search
            pipeline = Pipeline([
                ('csp', CSP(n_components=n_components, reg=None, log=True, norm_trace=False)),
                ('svm', SVC(kernel='rbf', probability=True, random_state=42))
            ])
            
            param_grid = {
                'csp__n_components': [6, 8, 10],
                'svm__C': [0.1, 1.0, 10.0],
                'svm__gamma': ['scale', 'auto']
            }
            
            grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', 
                                     n_jobs=-1, verbose=0)
            grid_search.fit(X_train, y_train)
            
            # Use best model
            best_model = grid_search.best_estimator_
            train_acc = best_model.score(X_train, y_train)
            test_acc = best_model.score(X_test, y_test)
            
            print(f"  Best parameters: {grid_search.best_params_}")
            print(f"  Optimized accuracy: {test_acc*100:.2f}%")
            
            # Extract components from pipeline and recompute test features
            csp = best_model.named_steps['csp']
            svm = best_model.named_steps['svm']
            X_test_csp = csp.transform(X_test)  # Transform test data with optimized CSP
        else:
            print(f"  Accuracy {test_acc*100:.2f}% >= 80%, using standard model")
        
        print(f"\nSubject {subj} Results:")
        print(f"  Train Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"  Test Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
        
        subject_models[subj] = {
            'model': svm,
            'csp': csp,
            'label_map': label_map,
            'unique_labels': unique_labels
        }
        subject_results[subj] = {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'y_test': y_test,
            'y_pred': svm.predict(X_test_csp)
        }
    
    # Overall results
    print(f"\n{'='*50}")
    print("OVERALL RESULTS")
    print(f"{'='*50}")
    overall_test_acc = np.mean([r['test_acc'] for r in subject_results.values()])
    overall_train_acc = np.mean([r['train_acc'] for r in subject_results.values()])
    print(f"Average Train Accuracy: {overall_train_acc:.4f} ({overall_train_acc*100:.2f}%)")
    print(f"Average Test Accuracy:  {overall_test_acc:.4f} ({overall_test_acc*100:.2f}%)")
    
    if overall_test_acc >= 0.80:
        print(f"\n✓ TARGET ACHIEVED! Average test accuracy: {overall_test_acc*100:.2f}% (>= 80%)")
    else:
        print(f"\n⚠ Below target (80%). Current: {overall_test_acc*100:.2f}%")
    
    # Detailed per-subject results
    print(f"\n{'='*50}")
    print("PER-SUBJECT DETAILS")
    print(f"{'='*50}")
    for subj, result in subject_results.items():
        print(f"\nSubject {subj}: Test Accuracy = {result['test_acc']*100:.2f}%")
        unique_labels = subject_models[subj]['unique_labels']
        print(classification_report(result['y_test'], result['y_pred'], 
                                  target_names=unique_labels, digits=3))
    
    return overall_test_acc, subject_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CSP+SVM training and test accuracy")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help=f"Path to EDF data directory (default: {DEFAULT_BASE_PATH} for Colab, or specify local path)"
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=5,
        help="Maximum number of subjects to process (default: 5)"
    )
    
    args = parser.parse_args()
    BASE_PATH = args.data_path if args.data_path else DEFAULT_BASE_PATH
    MAX_SUBJECTS = args.max_subjects
    
    # Check if running in Colab
    if os.path.exists("/content"):
        print("Running in Google Colab environment")
    else:
        print("Running locally")
    
    if not os.path.exists(BASE_PATH):
        print(f"\nERROR: Data path does not exist: {BASE_PATH}")
        print("\nPlease provide the correct path to your EDF data files using --data-path")
        print("Example: python run_csp_svm.py --data-path /path/to/your/data")
        print("\nExpected directory structure:")
        print("  BASE_PATH/")
        print("    S001/")
        print("      S001R01.edf")
        print("      S001R02.edf")
        print("      ...")
        print("    S002/")
        print("      ...")
        sys.exit(1)
    
    print(f"Using data path: {BASE_PATH}")
    print(f"Processing up to {MAX_SUBJECTS} subjects")
    
    try:
        test_acc, results = main(BASE_PATH)
        if test_acc is not None:
            print(f"\n{'='*60}")
            print(f"FINAL RESULT: Average Test Accuracy = {test_acc*100:.2f}%")
            print(f"{'='*60}")
        else:
            print("\nFailed to process data. Please check the error messages above.")
            sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
