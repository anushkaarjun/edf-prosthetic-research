#!/usr/bin/env python3
"""
Train improved neural network models with increased depth and width.
Tests different architectures to find the best accuracy.
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
from sklearn.metrics import accuracy_score, classification_report
import mne
import warnings
warnings.filterwarnings('ignore')

# Import data utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import importlib.util
data_utils_path = os.path.join(os.path.dirname(__file__), "..", "src", "edf_ml_model", "data_utils.py")
spec = importlib.util.spec_from_file_location("data_utils", data_utils_path)
data_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_utils)
get_run_number = data_utils.get_run_number
annotation_to_motion = data_utils.annotation_to_motion

# Import preprocessing
preprocessing_path = os.path.join(os.path.dirname(__file__), "..", "src", "edf_ml_model", "preprocessing.py")
spec = importlib.util.spec_from_file_location("preprocessing", preprocessing_path)
preprocessing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preprocessing)
create_half_second_epochs = preprocessing.create_half_second_epochs
normalize_signal = preprocessing.normalize_signal
preprocess_raw = preprocessing.preprocess_raw
smooth_signal = preprocessing.smooth_signal

# Import improved models
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "edf_ml_model"))
from improved_model import ImprovedEEGNet, SimpleEEGNet, DeepEEGNet

# Configuration
target_sfreq = 250
tmin, tmax = -0.5, 0.5
freq_low, freq_high = 8.0, 30.0
EPOCH_WINDOW = 0.5  # Default chunk size (0.1 to 0.5 seconds)
CHUNK_SIZE_MIN = 0.1  # Minimum chunk size for faster training
CHUNK_SIZE_MAX = 0.5  # Maximum chunk size


class EEGDataset(Dataset):
    """PyTorch dataset for EEG data."""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_validation_data(base_path, max_subjects=5, chunk_size=0.5, apply_smoothing=True):
    """
    Load validation data using small chunks (0.1s to 0.5s) with optional smoothing.
    
    Args:
        base_path: Path to data directory
        max_subjects: Maximum number of subjects to load
        chunk_size: Size of chunks in seconds (0.1 to 0.5)
        apply_smoothing: Whether to apply smoothing to reduce noise
    """
    # Clamp chunk_size to valid range
    chunk_size = max(CHUNK_SIZE_MIN, min(CHUNK_SIZE_MAX, chunk_size))
    
    subjects = sorted(glob.glob(f"{base_path}/S*"))[:max_subjects]
    print(f"Loading data from {len(subjects)} subjects: {[os.path.basename(s) for s in subjects]}")
    print(f"Using chunk size: {chunk_size}s, Smoothing: {apply_smoothing}")
    
    X_list = []
    y_labels_list = []
    
    for subj_path in subjects:
        subj_id = os.path.basename(subj_path)
        subj_files = sorted(glob.glob(f"{subj_path}/*.edf"))
        print(f"  Processing {subj_id}: {len(subj_files)} files")
        
        for file in subj_files:
            run = get_run_number(file)
            try:
                raw = mne.io.read_raw_edf(file, preload=True, verbose=False)
                
                # Use unified preprocessing with smoothing
                raw, metadata = preprocess_raw(
                    raw,
                    apply_filter=True,
                    clean=True,
                    normalize=True,
                    smooth=apply_smoothing,
                )
                
                if raw.info["sfreq"] != target_sfreq:
                    raw.resample(target_sfreq, npad="auto", verbose=False)
                
                events, event_id = mne.events_from_annotations(raw, verbose=False)
                if len(event_id) == 0:
                    continue
                
                # Create epochs with configurable chunk size
                epochs = create_half_second_epochs(raw, events, event_id, chunk_size=chunk_size)
                
                if len(epochs) == 0:
                    continue
                
                X = epochs.get_data()
                y_raw = epochs.events[:, -1]
                
                y_mapped = [annotation_to_motion(c, run) for c in y_raw]
                valid_idx = [i for i, v in enumerate(y_mapped) if v != "Unknown"]
                
                if len(valid_idx) > 0:
                    X_list.append(X[valid_idx])
                    y_labels_list += [y_mapped[i] for i in valid_idx]
                    
            except Exception as e:
                print(f"    Error processing {os.path.basename(file)}: {e}")
                continue
    
    if len(X_list) == 0:
        print("ERROR: No data loaded!")
        return None, None, None, None
    
    X_all = np.concatenate(X_list, axis=0).astype(np.float32)
    y_labels_all = y_labels_list
    
    print(f"\nLoaded {X_all.shape[0]} epochs from {len(subjects)} subjects")
    print(f"Data shape: {X_all.shape} (epochs, channels, samples)")
    
    unique_labels = sorted(set(y_labels_all))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    y_all = np.array([label_map[v] for v in y_labels_all], dtype=np.int64)
    
    print(f"Classes: {unique_labels}")
    print("Class distribution:")
    for label, idx in label_map.items():
        count = np.sum(y_all == idx)
        print(f"  {label}: {count} ({100*count/len(y_all):.1f}%)")
    
    return X_all, y_labels_all, y_all, unique_labels


def train_model(model, train_loader, val_loader, device, epochs=50, lr=0.001, weight_decay=1e-4):
    """Train model with improved settings."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
    )
    
    model.to(device)
    best_val_acc = 0.0
    patience = 15
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        val_acc = val_correct / val_total
        train_acc = train_correct / train_total
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | "
                  f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return model, best_val_acc


def test_model(model_name, model_class, X_train, X_val, y_train, y_val, n_channels, n_classes, device, epochs=50):
    """Test a specific model architecture."""
    print(f"\n{'='*60}")
    print(f"Testing {model_name}")
    print(f"{'='*60}")
    
    # Normalize data
    X_train_norm, _ = normalize_signal(X_train, method="minmax")
    X_val_norm, _ = normalize_signal(X_val, method="minmax")
    
    # Create model
    if model_name == "SimpleEEGNet":
        model = model_class(n_channels=n_channels, n_classes=n_classes, n_samples=X_train.shape[2])
    else:
        model = model_class(n_channels=n_channels, n_classes=n_classes, n_samples=X_train.shape[2])
    
    # Create data loaders
    train_dataset = EEGDataset(X_train_norm, y_train)
    val_dataset = EEGDataset(X_val_norm, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Train
    model, best_val_acc = train_model(model, train_loader, val_loader, device, epochs=epochs)
    
    # Final evaluation
    model.eval()
    val_preds = []
    val_true = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            val_preds.extend(predicted.cpu().numpy())
            val_true.extend(batch_y.numpy())
    
    val_acc = accuracy_score(val_true, val_preds)
    print(f"\n{model_name} Results:")
    print(f"  Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    
    return model, val_acc


def main(base_path, max_subjects=5, epochs=50, test_all=False, chunk_size=0.5, apply_smoothing=True):
    """
    Main training function.
    
    Args:
        base_path: Path to data directory
        max_subjects: Maximum number of subjects
        epochs: Number of training epochs
        test_all: Whether to test all model architectures
        chunk_size: Size of data chunks in seconds (0.1 to 0.5)
        apply_smoothing: Whether to apply smoothing to reduce noise
    """
    print("="*60)
    print("Training Improved Neural Network Models")
    print("="*60)
    print(f"Configuration: chunk_size={chunk_size}s, smoothing={apply_smoothing}")
    
    # Load data with configurable chunk size and smoothing
    X_all, y_labels_all, y_all, unique_labels = load_validation_data(
        base_path, max_subjects, chunk_size=chunk_size, apply_smoothing=apply_smoothing
    )
    if X_all is None:
        return
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    
    print(f"\nData split:")
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")
    
    n_channels = X_all.shape[1]
    n_classes = len(unique_labels)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    results = {}
    
    if test_all:
        # Test all models
        models_to_test = [
            ("ImprovedEEGNet", ImprovedEEGNet),
            ("SimpleEEGNet", SimpleEEGNet),
            ("DeepEEGNet", DeepEEGNet),
        ]
        
        for model_name, model_class in models_to_test:
            try:
                model, acc = test_model(
                    model_name, model_class, X_train, X_val, y_train, y_val,
                    n_channels, n_classes, device, epochs=epochs
                )
                results[model_name] = acc
                
                # Save best model
                if acc == max(results.values()):
                    os.makedirs('../models', exist_ok=True)
                    model_path = f'../models/best_{model_name.lower()}.pth'
                    torch.save(model.state_dict(), model_path)
                    print(f"Saved best model: {model_path}")
            except Exception as e:
                print(f"Error testing {model_name}: {e}")
                import traceback
                traceback.print_exc()
    else:
        # Test only ImprovedEEGNet (recommended)
            model, acc = test_model(
                "ImprovedEEGNet", ImprovedEEGNet, X_train, X_val, y_train, y_val,
                n_channels, n_classes, device, epochs=epochs
            )
        results["ImprovedEEGNet"] = acc
        os.makedirs('../models', exist_ok=True)
        model_path = '../models/best_improved_eegnet.pth'
        torch.save(model.state_dict(), model_path)
        print(f"\nSaved model: {model_path}")
    
    # Summary
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    for model_name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{model_name}: {acc:.4f} ({acc*100:.2f}%)")
    
    best_model = max(results.items(), key=lambda x: x[1])
    print(f"\nBest Model: {best_model[0]} with {best_model[1]:.4f} ({best_model[1]*100:.2f}%) accuracy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train improved EEG models")
    parser.add_argument("--data-path", type=str, required=True,
                       help="Path to EDF data directory")
    parser.add_argument("--max-subjects", type=int, default=5,
                       help="Maximum number of subjects to process")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--test-all", action="store_true",
                       help="Test all model architectures")
    parser.add_argument("--chunk-size", type=float, default=0.5,
                       help="Size of data chunks in seconds (0.1 to 0.5, default: 0.5)")
    parser.add_argument("--no-smoothing", action="store_true",
                       help="Disable smoothing/averaging (smoothing enabled by default)")
    
    args = parser.parse_args()
    
    # Validate chunk size
    chunk_size = max(CHUNK_SIZE_MIN, min(CHUNK_SIZE_MAX, args.chunk_size))
    if args.chunk_size != chunk_size:
        print(f"Warning: chunk_size clamped to {chunk_size}s (valid range: {CHUNK_SIZE_MIN}-{CHUNK_SIZE_MAX})")
    
    if not os.path.exists(args.data_path):
        print(f"ERROR: Data path does not exist: {args.data_path}")
        sys.exit(1)
    
    main(
        base_path=args.data_path,
        max_subjects=args.max_subjects,
        epochs=args.epochs,
        test_all=args.test_all,
        chunk_size=chunk_size,
        apply_smoothing=not args.no_smoothing
    )
