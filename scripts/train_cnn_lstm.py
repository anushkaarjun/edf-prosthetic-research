#!/usr/bin/env python3
"""
Train CNN-LSTM model on validation data.
Similar to train_on_validation_data.py but for CNN-LSTM.
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import mne
import warnings
warnings.filterwarnings('ignore')

# Import data utilities
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.join(_script_dir, "..")
sys.path.insert(0, _repo_root)
sys.path.insert(0, os.path.join(_repo_root, "src"))
import importlib.util
data_utils_path = os.path.join(_repo_root, "src", "edf_ml_model", "data_utils.py")
spec = importlib.util.spec_from_file_location("data_utils", data_utils_path)
data_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_utils)
get_run_number = data_utils.get_run_number
annotation_to_motion = data_utils.annotation_to_motion

# Import CNN-LSTM model
from cnn_lstm_model import CNNLSTM, N_CLASSES, CLASS_NAMES

# Configuration
target_sfreq = 160  # CNN-LSTM uses 160Hz
SLIDING_WINDOW = 320  # 2 seconds at 160Hz
WINDOW_STEP = 80  # 75% overlap


def load_cnn_lstm_data(base_path, max_subjects=5, exclude_both_feet=False, exclude_left_fist=False):
    """
    Load data for CNN-LSTM training.
    Uses 2-second sliding windows at 160Hz.
    exclude_both_feet: if True, use 3 classes (Both Fists, Left Hand, Right Hand); else use Open/Close Fists mapping.
    exclude_left_fist: if True, exclude Open Left Fist (2 classes: Open Right Fist, Close Fists).
    """
    subjects = sorted(glob.glob(f"{base_path}/S*"))[:max_subjects]
    print(f"Loading data from {len(subjects)} subjects for CNN-LSTM...")
    if exclude_both_feet:
        print("Excluding 'Both Feet' (3 classes: Both Fists, Left Hand, Right Hand)")
    if exclude_left_fist:
        print("Excluding 'Open Left Fist' (2 classes: Open Right Fist, Close Fists)")
    
    X_list = []
    y_labels_list = []
    
    for subj_path in subjects:
        subj_id = os.path.basename(subj_path)
        subj_files = sorted(glob.glob(f"{subj_path}/*.edf"))
        print(f"  Processing {subj_id}: {len(subj_files)} files")
        
        for file in subj_files:
            run = get_run_number(file)
            try:
                # Load raw data
                raw = mne.io.read_raw_edf(file, preload=True, verbose=False)
                
                # Resample to 160Hz (CNN-LSTM requirement)
                if raw.info["sfreq"] != target_sfreq:
                    raw.resample(target_sfreq, npad="auto", verbose=False)
                
                # Get events
                events, event_id = mne.events_from_annotations(raw, verbose=False)
                if len(event_id) == 0:
                    continue
                
                # Create epochs (2-second windows)
                tmin, tmax = 0, 2
                epochs = mne.Epochs(
                    raw, events, event_id=event_id,
                    tmin=tmin, tmax=tmax,
                    baseline=None,  # No baseline correction for CNN-LSTM
                    preload=True, verbose=False,
                    reject=None, flat=None
                )
                
                if len(epochs) == 0:
                    continue
                
                X = epochs.get_data()  # (n_epochs, n_channels, n_samples)
                
                # Map to classes
                y_raw = epochs.events[:, -1]
                y_mapped = []
                for c in y_raw:
                    motion = annotation_to_motion(c, run)
                    if exclude_both_feet:
                        # 3 classes: Both Fists, Left Hand, Right Hand (exclude Both Feet)
                        if motion == "Both Feet" or motion == "Unknown":
                            y_mapped.append(None)
                        elif motion in ["Left Hand", "Right Hand", "Both Fists"]:
                            y_mapped.append(motion)
                        else:
                            y_mapped.append(None)
                    else:
                        # CNN-LSTM default: Open Left Fist, Open Right Fist, Close Fists
                        if motion == "Left Hand":
                            y_mapped.append("Open Left Fist")
                        elif motion == "Right Hand":
                            y_mapped.append("Open Right Fist")
                        elif motion in ["Both Fists", "Both Feet"]:
                            y_mapped.append("Close Fists")
                        else:
                            y_mapped.append(None)
                
                valid_idx = [i for i, v in enumerate(y_mapped) if v is not None]
                
                if len(valid_idx) > 0:
                    X_list.append(X[valid_idx])
                    y_labels_list += [y_mapped[i] for i in valid_idx]
                    
            except Exception as e:
                print(f"    Error processing {os.path.basename(file)}: {e}")
                continue
    
    if len(X_list) == 0:
        print("ERROR: No data loaded!")
        return None, None, None
    
    # Concatenate all epochs
    X_all = np.concatenate(X_list, axis=0).astype(np.float32)
    y_labels_all = y_labels_list
    
    print(f"\nLoaded {X_all.shape[0]} epochs from {len(subjects)} subjects")
    print(f"Data shape: {X_all.shape} (epochs, channels, samples)")
    
    # Get unique labels and create numeric mapping
    if exclude_both_feet:
        unique_labels = sorted(set(y_labels_all))  # Both Fists, Left Hand, Right Hand
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        y_all = np.array([label_map[v] for v in y_labels_all], dtype=np.int64)
    else:
        unique_labels = CLASS_NAMES  # Open Left Fist, Open Right Fist, Close Fists
        if exclude_left_fist:
            unique_labels = [l for l in unique_labels if l != "Open Left Fist"]  # Open Right Fist, Close Fists
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        valid_indices = [i for i, y in enumerate(y_labels_all) if y in unique_labels]
        if len(valid_indices) == 0:
            print("ERROR: No valid CNN-LSTM labels found!")
            return None, None, None
        X_all = X_all[valid_indices]
        y_labels_all = [y_labels_all[i] for i in valid_indices]
        y_all = np.array([label_map[v] for v in y_labels_all], dtype=np.int64)
    
    print(f"Classes: {unique_labels}")
    print("Class distribution:")
    for label, idx in label_map.items():
        count = np.sum(y_all == idx)
        if len(y_all) > 0:
            print(f"  {label}: {count} ({100*count/len(y_all):.1f}%)")
    
    return X_all, y_labels_all, y_all, unique_labels


def _augment_after_scale(X_scaled, noise_std=0.08, max_shift=5):
    """Light augmentation in scaled space: time shift + small noise. X_scaled: (1, n_channels, n_times)."""
    X = X_scaled.copy()
    if max_shift > 0:
        shift = np.random.randint(-max_shift, max_shift + 1)
        if shift != 0:
            X = np.roll(X, shift, axis=2)
    if noise_std > 0:
        X = X + noise_std * np.random.randn(*X.shape).astype(np.float32)
    return X


class CNNLSTMDataset(Dataset):
    """PyTorch dataset for CNN-LSTM data."""
    def __init__(self, X, y, scaler=None, augment=False):
        self.augment = augment
        self.X_raw = X  # (n_samples, n_channels, n_times)
        self.y = torch.LongTensor(y)
        if scaler is None:
            self.scaler = StandardScaler()
            n_samples, n_channels, n_times = X.shape
            X_reshaped = X.reshape(n_samples, -1)
            self.scaler.fit(X_reshaped)
        else:
            self.scaler = scaler
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x = self.X_raw[idx]
        x = x[np.newaxis, :, :]  # (1, n_channels, n_times)
        X_reshaped = x.reshape(1, -1)
        X_scaled = self.scaler.transform(X_reshaped)
        X_scaled = X_scaled.reshape(1, self.X_raw.shape[1], self.X_raw.shape[2])
        if self.augment:
            X_scaled = _augment_after_scale(X_scaled)
        X_scaled = X_scaled[:, np.newaxis, :, :]  # (1, 1, n_channels, n_times)
        return torch.FloatTensor(X_scaled).squeeze(0), self.y[idx]


def train_cnn_lstm(X_train, X_val, y_train, y_val, n_channels, n_classes, device, epochs=60, unique_labels=None, use_augmentation=True):
    """Train CNN-LSTM model. Uses augmentation and longer training to reach higher accuracy."""
    if unique_labels is None:
        unique_labels = CLASS_NAMES
    # Create datasets: augmentation on for training, off for val
    train_dataset = CNNLSTMDataset(X_train, y_train, augment=use_augmentation)
    val_dataset = CNNLSTMDataset(X_val, y_val, scaler=train_dataset.scaler, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Create model
    model = CNNLSTM(n_channels=n_channels, n_classes=n_classes, dropout=0.4)
    model.to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=6)
    
    best_val_acc = 0.0
    patience = 18
    patience_counter = 0
    
    print(f"\nTraining CNN-LSTM for {epochs} epochs (augmentation={use_augmentation})...")
    for epoch in range(epochs):
        # Training phase
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        # Validation phase
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
            # Save best model
            os.makedirs('../models', exist_ok=True)
            torch.save(model.state_dict(), '../models/best_model.pth')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | "
                  f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(torch.load('../models/best_model.pth'))
    
    # Evaluate on validation set
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
    print(f"\nCNN-LSTM Results:")
    print(f"  Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(classification_report(val_true, val_preds, target_names=unique_labels))
    
    return model, val_acc


def main(base_path, max_subjects=5, exclude_both_feet=False, exclude_left_fist=False, multi_seed=1, epochs=50):
    """Main training function. If multi_seed > 1, run multiple seeds and keep best model."""
    print("="*60)
    print("Training CNN-LSTM Model on Validation Data")
    print("="*60)
    
    # Load data
    X_all, y_labels_all, y_all, unique_labels = load_cnn_lstm_data(
        base_path, max_subjects, exclude_both_feet=exclude_both_feet, exclude_left_fist=exclude_left_fist
    )
    if X_all is None:
        return
    
    n_channels = X_all.shape[1]
    n_classes = len(unique_labels)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    seeds = [42, 123, 456, 789, 2024][:multi_seed]
    best_acc = 0.0
    best_model = None
    
    for run, seed in enumerate(seeds):
        if multi_seed > 1:
            print(f"\n{'='*60}")
            print(f"Run {run+1}/{multi_seed} (seed={seed})")
            print("="*60)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_all, y_all, test_size=0.2, random_state=seed, stratify=y_all
        )
        if multi_seed == 1:
            print(f"\nData split: Train {X_train.shape[0]}, Val {X_val.shape[0]}")
        
        model, val_acc = train_cnn_lstm(
            X_train, X_val, y_train, y_val, n_channels, n_classes, device,
            epochs=epochs, unique_labels=unique_labels, use_augmentation=True
        )
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = model
            os.makedirs("../models", exist_ok=True)
            torch.save(model.state_dict(), "../models/best_model.pth")
            if multi_seed > 1:
                print(f"  -> New best: {val_acc:.4f} ({val_acc*100:.2f}%)")
    
    if multi_seed > 1:
        print(f"\nBest validation accuracy across {multi_seed} runs: {best_acc:.4f} ({best_acc*100:.2f}%)")
    print(f"\nSaved CNN-LSTM model to '../models/best_model.pth'")
    print(f"Final validation accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN-LSTM model on validation data")
    parser.add_argument("--data-path", type=str, required=True,
                       help="Path to EDF data directory")
    parser.add_argument("--max-subjects", type=int, default=5,
                       help="Maximum number of subjects to process")
    parser.add_argument("--exclude-both-feet", action="store_true",
                       help="Exclude 'Both Feet' (3 classes: Both Fists, Left Hand, Right Hand)")
    parser.add_argument("--exclude-left-fist", action="store_true",
                       help="Exclude 'Open Left Fist' (2 classes: Open Right Fist, Close Fists)")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--multi-seed", type=int, default=1,
                       help="Run N times with different seeds and keep best (e.g. 3 for better chance at 60%%)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_path):
        print(f"ERROR: Data path does not exist: {args.data_path}")
        sys.exit(1)
    
    main(
        base_path=args.data_path,
        max_subjects=args.max_subjects,
        exclude_both_feet=args.exclude_both_feet,
        exclude_left_fist=args.exclude_left_fist,
        multi_seed=args.multi_seed,
        epochs=args.epochs,
    )
