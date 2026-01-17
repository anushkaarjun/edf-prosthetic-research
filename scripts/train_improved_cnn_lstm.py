#!/usr/bin/env python3
"""
Train improved CNN-LSTM model on validation data.
Uses improved architecture with increased depth and width.
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import importlib.util
data_utils_path = os.path.join(os.path.dirname(__file__), "..", "src", "edf_ml_model", "data_utils.py")
spec = importlib.util.spec_from_file_location("data_utils", data_utils_path)
data_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_utils)
get_run_number = data_utils.get_run_number
annotation_to_motion = data_utils.annotation_to_motion

# Import improved CNN-LSTM model
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from improved_cnn_lstm_model import ImprovedCNNLSTM, N_CLASSES, CLASS_NAMES

# Configuration
target_sfreq = 160  # CNN-LSTM uses 160Hz
SLIDING_WINDOW = 320  # 2 seconds at 160Hz
WINDOW_STEP = 80  # 75% overlap


def load_cnn_lstm_data(base_path, max_subjects=5):
    """Load data for CNN-LSTM training. Uses 2-second sliding windows at 160Hz."""
    subjects = sorted(glob.glob(f"{base_path}/S*"))[:max_subjects]
    print(f"Loading data from {len(subjects)} subjects for Improved CNN-LSTM...")
    
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
                
                # Resample to 160Hz
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
                    baseline=None,
                    preload=True, verbose=False,
                    reject=None, flat=None
                )
                
                if len(epochs) == 0:
                    continue
                
                X = epochs.get_data()  # (n_epochs, n_channels, n_samples)
                
                # Map to CNN-LSTM classes
                y_raw = epochs.events[:, -1]
                y_mapped = []
                for c in y_raw:
                    motion = annotation_to_motion(c, run)
                    if motion == "Left Hand":
                        y_mapped.append("Open Left Fist")
                    elif motion == "Right Hand":
                        y_mapped.append("Open Right Fist")
                    elif motion in ["Both Fists", "Both Feet"]:
                        y_mapped.append("Close Fists")
                    else:
                        continue
                
                # Create sliding windows
                n_epochs, n_channels, n_samples = X.shape
                for epoch_idx in range(n_epochs):
                    epoch_data = X[epoch_idx]  # (n_channels, n_samples)
                    
                    # Standardize each channel
                    scaler = StandardScaler()
                    epoch_data = scaler.fit_transform(epoch_data.T).T
                    
                    # Create sliding windows
                    for start in range(0, n_samples - SLIDING_WINDOW + 1, WINDOW_STEP):
                        window = epoch_data[:, start:start + SLIDING_WINDOW]
                        X_list.append(window)
                        y_labels_list.append(y_mapped[epoch_idx])
            
            except Exception as e:
                print(f"    Error processing {file}: {e}")
                continue
    
    return np.array(X_list), y_labels_list


class ImprovedCNNLSTMDataset(Dataset):
    """PyTorch dataset for improved CNN-LSTM data."""
    def __init__(self, X, y, scaler=None):
        self.X = torch.FloatTensor(X)
        # Reshape for Conv2d: (batch, 1, channels, time)
        if len(self.X.shape) == 3:
            self.X = self.X.unsqueeze(1)  # Add channel dimension
        
        # Convert labels to indices
        label_to_idx = {label: idx for idx, label in enumerate(CLASS_NAMES)}
        self.y = torch.LongTensor([label_to_idx[label] for label in y])
        
        self.scaler = scaler
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def compute_class_weights(y_train):
    """Compute class weights for imbalanced dataset."""
    from collections import Counter
    counts = Counter(y_train)
    total = len(y_train)
    weights = {cls: total / (len(CLASS_NAMES) * count) for cls, count in counts.items()}
    return torch.FloatTensor([weights[CLASS_NAMES[i]] for i in range(N_CLASSES)])


def train_improved_cnn_lstm(X_train, y_train, X_val, y_val, n_channels, n_classes, epochs=50, device='cpu'):
    """Train improved CNN-LSTM model."""
    train_dataset = ImprovedCNNLSTMDataset(X_train, y_train)
    val_dataset = ImprovedCNNLSTMDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create improved model
    model = ImprovedCNNLSTM(n_channels=n_channels, n_classes=n_classes, dropout=0.5)
    model.to(device)
    
    # Class weights for imbalanced data
    class_weights = compute_class_weights(y_train).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer with weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Early stopping
    best_val_acc = 0.0
    patience = 15
    patience_counter = 0
    best_model_state = None
    
    print(f"\nTraining Improved CNN-LSTM for {epochs} epochs...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
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
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        train_acc = train_correct / train_total
        
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
        scheduler.step(val_acc)
        
        # Early stopping and best model tracking
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | "
                  f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, best_val_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train improved CNN-LSTM model")
    parser.add_argument("--data-path", type=str, required=True,
                       help="Path to data directory containing subject folders")
    parser.add_argument("--max-subjects", type=int, default=5,
                       help="Maximum number of subjects to use")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    
    args = parser.parse_args()
    
    # Load data
    X, y_labels = load_cnn_lstm_data(args.data_path, args.max_subjects)
    
    if len(X) == 0:
        print("ERROR: No data loaded!")
        sys.exit(1)
    
    # Filter to only include CNN-LSTM classes
    unique_labels = CLASS_NAMES
    valid_indices = [i for i, label in enumerate(y_labels) if label in unique_labels]
    X = X[valid_indices]
    y_labels = [y_labels[i] for i in valid_indices]
    
    if len(X) == 0:
        print("ERROR: No valid CNN-LSTM labels found!")
        sys.exit(1)
    
    print(f"\nLoaded {len(X)} samples")
    print(f"Classes: {unique_labels}")
    print(f"Class distribution: {dict(zip(*np.unique(y_labels, return_counts=True)))}")
    
    # Convert labels to indices for training
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    y = [label_to_idx[label] for label in y_labels]
    
    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain: {len(X_train)} samples")
    print(f"Validation: {len(X_val)} samples")
    
    # Get dimensions
    n_channels = X.shape[1]
    n_classes = len(unique_labels)
    
    # Train model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    model, val_acc = train_improved_cnn_lstm(
        X_train, y_train, X_val, y_val,
        n_channels, n_classes,
        epochs=args.epochs,
        device=device
    )
    
    # Save model
    os.makedirs('../models', exist_ok=True)
    model_path = '../models/best_improved_cnn_lstm.pth'
    torch.save(model.state_dict(), model_path)
    
    print(f"\nImproved CNN-LSTM Results:")
    print(f"  Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"\nSaved model to: {model_path}")
