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
from sklearn.model_selection import GridSearchCV
import mne

# Import from our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from edf_ml_model.preprocessing import (
    preprocess_raw, create_half_second_epochs, TARGET_SFREQ, EPOCH_WINDOW
)
from edf_ml_model.data_utils import get_run_number, annotation_to_motion
from edf_ml_model.model import EEGMotorImageryNet, train_model


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
                
                # Preprocess (filter, resample, normalize, clean)
                raw, metadata = preprocess_raw(raw, apply_filter=True, clean=True, normalize=True)
                
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
            subject_data[subj_id] = {"X": X_subj, "y": y_list}
            print(f"  Subject {subj_id}: {X_subj.shape[0]} epochs, shape {X_subj.shape}")
    
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
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
        
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
        test_dataset = EEGDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Phase 1: Train all weights
        print(f"\nPhase 1: Training all weights (epochs 1-{freeze_after})...")
        model, best_val_acc = train_model(
            model, train_loader, test_loader,
            epochs=freeze_after, device=device, freeze_after=None
        )
        
        # Phase 2: Freeze backbone, fine-tune classifier
        print(f"\nPhase 2: Freezing backbone, fine-tuning classifier (epochs {freeze_after+1}-{epochs})...")
        model.freeze_backbone()  # Manually freeze before starting fine-tuning
        model, final_val_acc = train_model(
            model, train_loader, test_loader,
            epochs=epochs - freeze_after, device=device, freeze_after=None  # Already frozen
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
        
        print(f"\nSubject {subj} Results:")
        print(f"  Final Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        
        subject_models[subj] = {
            "model": model,
            "label_map": label_map,
            "unique_labels": unique_labels,
            "n_channels": n_channels,
            "n_samples": n_samples,
        }
        subject_results[subj] = {
            "test_acc": test_acc,
            "y_test": all_true,
            "y_pred": all_preds,
        }
    
    # Overall results
    print(f"\n{'='*50}")
    print("OVERALL RESULTS")
    print(f"{'='*50}")
    overall_test_acc = np.mean([r["test_acc"] for r in subject_results.values()])
    print(f"Average Test Accuracy: {overall_test_acc:.4f} ({overall_test_acc*100:.2f}%)")
    
    if overall_test_acc >= 0.80:
        print(f"\n✓ TARGET ACHIEVED! Average test accuracy: {overall_test_acc*100:.2f}% (>= 80%)")
    else:
        print(f"\n⚠ Below target (80%). Current: {overall_test_acc*100:.2f}%")
    
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
