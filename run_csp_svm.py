#!/usr/bin/env python3
"""
Standalone script to run CSP+SVM training and get test accuracy.
This is a simplified version of the notebook that can be run locally.
"""

import os
import glob
import numpy as np
import mne
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from mne.decoding import CSP
import matplotlib.pyplot as plt

# Configuration
BASE_PATH = "/content/drive/MyDrive/files 2"  # Update this path
MAX_SUBJECTS = 5
target_sfreq = 250
tmin, tmax = 0, 4
freq_low, freq_high = 8., 30.
n_components = 8

def get_run_number(filepath):
    """Extract run number R01, R02..."""
    name = os.path.basename(filepath)
    if "R" in name:
        return int(name.split("R")[1].split(".")[0])
    return None

def annotation_to_motion(code, run):
    """Map annotation codes to motion labels based on run number."""
    if code == 0:
        return "Rest"
    if run in [3,4,7,8,11,12]:
        return "Left Hand" if code == 1 else "Right Hand"
    if run in [5,6,9,10,13,14]:
        return "Both Fists" if code == 1 else "Both Feet"
    return "Unknown"

def main():
    # Get subjects
    subjects = sorted(glob.glob(f"{BASE_PATH}/S*"))[:MAX_SUBJECTS]
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
                                    baseline=(-0.5, 0),
                                    preload=True, verbose=False)
                
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
        return
    
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
        
        # Train SVM
        print("Training SVM classifier...")
        svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
        svm.fit(X_train_csp, y_train)
        
        # Evaluate
        train_acc = svm.score(X_train_csp, y_train)
        test_acc = svm.score(X_test_csp, y_test)
        
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
    # Check if running in Colab
    if os.path.exists("/content"):
        print("Running in Google Colab environment")
    else:
        print("Running locally - update BASE_PATH to point to your data")
        print(f"Current BASE_PATH: {BASE_PATH}")
    
    try:
        test_acc, results = main()
        print(f"\n{'='*60}")
        print(f"FINAL RESULT: Average Test Accuracy = {test_acc*100:.2f}%")
        print(f"{'='*60}")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
