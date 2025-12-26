#!/usr/bin/env python3
"""
Training script with weight freezing and hyperparameter tuning workflow.
Uses 0.5-second epochs and normalized data.
"""
import os
import sys
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import mne

# Import from our modules (direct imports to avoid dependency issues)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Import data_utils directly
import importlib.util
data_utils_path = os.path.join(os.path.dirname(__file__), "src", "edf_ml_model", "data_utils.py")
spec = importlib.util.spec_from_file_location("data_utils", data_utils_path)
data_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_utils)
get_run_number = data_utils.get_run_number
annotation_to_motion = data_utils.annotation_to_motion

# Import preprocessing constants and functions
preprocessing_path = os.path.join(os.path.dirname(__file__), "src", "edf_ml_model", "preprocessing.py")
spec = importlib.util.spec_from_file_location("preprocessing", preprocessing_path)
preprocessing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preprocessing)
TARGET_SFREQ = preprocessing.TARGET_SFREQ
EPOCH_WINDOW = preprocessing.EPOCH_WINDOW
create_half_second_epochs = preprocessing.create_half_second_epochs
normalize_signal = preprocessing.normalize_signal

# Import model (requires torch)
try:
    model_path = os.path.join(os.path.dirname(__file__), "src", "edf_ml_model", "model.py")
    spec = importlib.util.spec_from_file_location("model", model_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    EEGMotorImageryNet = model_module.EEGMotorImageryNet
    train_model = model_module.train_model
    tune_hyperparameters = model_module.tune_hyperparameters
    train_model_with_hyperparams = model_module.train_model_with_hyperparams
except Exception as e:
    print(f"Warning: Could not import model module: {e}")
    print("PyTorch may not be installed. Install with: pip install torch")
    raise


class EEGDataset(Dataset):
    """PyTorch dataset for EEG data."""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_and_preprocess_data(base_path, max_subjects=5):
    """Load and preprocess EDF files with 0.5-second epochs."""
    subjects = sorted(glob.glob(f"{base_path}/S*"))[:max_subjects]
    print(f"Loading data from {len(subjects)} subjects...")
    
    subject_data = {}
    for subj in subjects:
        subj_id = os.path.basename(subj)
        subj_files = sorted(glob.glob(f"{subj}/*.edf"))
        
        X_list = []
        y_list = []
        
        for file in subj_files:
            run = get_run_number(file)
            try:
                # Load raw data
                raw = mne.io.read_raw_edf(file, preload=True, verbose=False)
                
                # Preprocess (filter, resample) - normalization done after epoch creation
                # Skip cleaning to avoid digitization errors
                raw.filter(8.0, 30.0, verbose=False, fir_design='firwin')
                if raw.info["sfreq"] != TARGET_SFREQ:
                    raw.resample(TARGET_SFREQ, npad="auto", verbose=False)
                
                # Get events
                events, event_id = mne.events_from_annotations(raw, verbose=False)
                if len(event_id) == 0:
                    continue
                
                # Create 0.5-second epochs
                epochs = create_half_second_epochs(raw, events, event_id)
                
                if len(epochs) == 0:
                    continue
                
                X = epochs.get_data()  # (n_epochs, n_channels, n_samples)
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
            
            # Ensure normalization to [-1, 1] range across all epochs
            # This ensures consistent weight scaling in the neural network
            X_subj, norm_params = normalize_signal(X_subj, method="minmax")
            
            # Verify normalization range
            actual_min = np.min(X_subj)
            actual_max = np.max(X_subj)
            print(f"  Subject {subj_id}: {X_subj.shape[0]} epochs, shape {X_subj.shape}")
            print(f"    Normalized range: [{actual_min:.4f}, {actual_max:.4f}] (target: [-1, 1])")
            
            subject_data[subj_id] = {"X": X_subj, "y": y_list}
    
    return subject_data


def main(base_path=None, max_subjects=5, freeze_after=30, epochs=50):
    """Main training function with weight freezing workflow."""
    if base_path is None:
        base_path = "./data"
    
    # Load and preprocess data
    subject_data = load_and_preprocess_data(base_path, max_subjects)
    if len(subject_data) == 0:
        print("ERROR: No data processed")
        return None, None
    
    subject_models = {}
    subject_results = {}
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    for subj, data in subject_data.items():
        print(f"\n{'='*50}")
        print(f"Training model for Subject {subj}")
        print(f"{'='*50}")
        
        X = data["X"]  # (n_epochs, n_channels, n_samples)
        y_labels = data["y"]
        
        # Encode labels
        unique_labels = sorted(set(y_labels))
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        y = np.array([label_map[v] for v in y_labels], dtype=np.int64)
        
        print(f"Data shape: {X.shape}, Classes: {unique_labels}")
        print("Class distribution:")
        for label, idx in label_map.items():
            count = np.sum(y == idx)
            print(f"  {label}: {count} ({100*count/len(y):.1f}%)")
        
        # Split data into train, validation, and test sets
        # First split: separate test set (20%)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Second split: separate train and validation from remaining data
        # This gives us approximately 64% train, 16% validation, 20% test
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
        )
        
        print(f"Train: {X_train.shape[0]}, Validation: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        
        # Get model dimensions
        n_channels = X.shape[1]
        n_samples = X.shape[2]  # Should be 125 for 0.5s at 250Hz
        n_classes = len(unique_labels)
        
        # Create model
        model = EEGMotorImageryNet(
            n_channels=n_channels, n_classes=n_classes, n_samples=n_samples
        )
        model.to(device)
        
        # Create data loaders
        train_dataset = EEGDataset(X_train, y_train)
        val_dataset = EEGDataset(X_val, y_val)
        test_dataset = EEGDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Phase 1: Train all weights
        print(f"\nPhase 1: Training all weights (epochs 1-{freeze_after})...")
        model, phase1_val_acc = train_model(
            model, train_loader, val_loader,
            epochs=freeze_after, device=device, freeze_after=None
        )
        
        # Phase 2: Freeze backbone weights
        print(f"\nPhase 2: Freezing backbone weights...")
        model.freeze_backbone()
        print("  Backbone layers frozen. Only classifier will be trained.")
        
        # Phase 3: Tune hyperparameters on validation set
        print(f"\nPhase 3: Tuning hyperparameters on validation set...")
        best_val_acc, best_hyperparams = tune_hyperparameters(
            model, train_loader, val_loader, device=device
        )
        
        # Phase 4: Fine-tune classifier with best hyperparameters
        print(f"\nPhase 4: Fine-tuning classifier with best hyperparameters (epochs {freeze_after+1}-{epochs})...")
        print(f"  Best hyperparameters: {best_hyperparams}")
        model, final_val_acc = train_model_with_hyperparams(
            model, train_loader, val_loader,
            epochs=epochs - freeze_after, device=device,
            lr=best_hyperparams['lr'],
            weight_decay=best_hyperparams['weight_decay']
        )
        
        # Evaluate final model
        model.eval()
        test_correct = 0
        test_total = 0
        all_preds = []
        all_true = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                test_total += batch_y.size(0)
                test_correct += (predicted == batch_y).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_true.extend(batch_y.cpu().numpy())
        
        test_acc = test_correct / test_total
        
        # Also evaluate on validation set
        val_correct = 0
        val_total = 0
        val_preds = []
        val_true = []
        
        model.eval()
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
                val_preds.extend(predicted.cpu().numpy())
                val_true.extend(batch_y.cpu().numpy())
        
        val_acc = val_correct / val_total
        
        print(f"\nSubject {subj} Results:")
        print(f"  Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
        print(f"  Final Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        
        subject_models[subj] = {
            "model": model,
            "label_map": label_map,
            "unique_labels": unique_labels,
            "n_channels": n_channels,
            "n_samples": n_samples,
        }
        subject_results[subj] = {
            "val_acc": val_acc,
            "test_acc": test_acc,
            "y_val": val_true,
            "y_test": all_true,
            "y_pred": all_preds,
        }
    
    # Overall results
    print(f"\n{'='*50}")
    print("OVERALL RESULTS")
    print(f"{'='*50}")
    overall_val_acc = np.mean([r["val_acc"] for r in subject_results.values()])
    overall_test_acc = np.mean([r["test_acc"] for r in subject_results.values()])
    print(f"Average Validation Accuracy: {overall_val_acc:.4f} ({overall_val_acc*100:.2f}%)")
    print(f"Average Test Accuracy: {overall_test_acc:.4f} ({overall_test_acc*100:.2f}%)")
    
    if overall_test_acc >= 0.80:
        print(f"\n✓ TARGET ACHIEVED! Average test accuracy: {overall_test_acc*100:.2f}% (>= 80%)")
    else:
        print(f"\n⚠ Below target (80%). Current test accuracy: {overall_test_acc*100:.2f}%")
    
    return subject_models, subject_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EEG model with weight freezing")
    parser.add_argument("--data-path", type=str, default="./data",
                       help="Path to EDF data directory")
    parser.add_argument("--max-subjects", type=int, default=5,
                       help="Maximum number of subjects to process")
    parser.add_argument("--freeze-after", type=int, default=30,
                       help="Epoch to freeze backbone weights")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Total number of training epochs")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_path):
        print(f"ERROR: Data path does not exist: {args.data_path}")
        sys.exit(1)
    
    models, results = main(
        base_path=args.data_path,
        max_subjects=args.max_subjects,
        freeze_after=args.freeze_after,
        epochs=args.epochs
    )
