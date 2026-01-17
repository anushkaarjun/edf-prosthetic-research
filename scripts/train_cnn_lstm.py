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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import importlib.util
data_utils_path = os.path.join(os.path.dirname(__file__), "src", "edf_ml_model", "data_utils.py")
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


def load_cnn_lstm_data(base_path, max_subjects=5):
    """
    Load data for CNN-LSTM training.
    Uses 2-second sliding windows at 160Hz.
    """
    subjects = sorted(glob.glob(f"{base_path}/S*"))[:max_subjects]
    print(f"Loading data from {len(subjects)} subjects for CNN-LSTM...")
    
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
                
                # Map to CNN-LSTM classes
                # CNN-LSTM uses 3 classes: Open Left Fist, Open Right Fist, Close Fists
                # Map from original labels
                y_raw = epochs.events[:, -1]
                y_mapped = []
                for c in y_raw:
                    motion = annotation_to_motion(c, run)
                    # Map to CNN-LSTM classes
                    if motion == "Left Hand":
                        y_mapped.append("Open Left Fist")
                    elif motion == "Right Hand":
                        y_mapped.append("Open Right Fist")
                    elif motion in ["Both Fists", "Both Feet"]:
                        y_mapped.append("Close Fists")
                    else:
                        continue  # Skip Rest and Unknown
                
                valid_idx = [i for i, v in enumerate(y_mapped) if v is not None and v != "Unknown"]
                
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
    
    # Get unique labels and create numeric mapping (should be CNN-LSTM classes)
    unique_labels = CLASS_NAMES  # Use CNN-LSTM class names
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    
    # Filter to only include CNN-LSTM classes
    valid_labels = [y for y in y_labels_all if y in unique_labels]
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


class CNNLSTMDataset(Dataset):
    """PyTorch dataset for CNN-LSTM data."""
    def __init__(self, X, y, scaler=None):
        # StandardScaler normalization
        if scaler is None:
            self.scaler = StandardScaler()
            # Reshape for scaler: (n_samples, n_features)
            n_samples, n_channels, n_times = X.shape
            X_reshaped = X.reshape(n_samples, -1)
            self.scaler.fit(X_reshaped)
        else:
            self.scaler = scaler
        
        n_samples, n_channels, n_times = X.shape
        X_reshaped = X.reshape(n_samples, -1)
        X_scaled = self.scaler.transform(X_reshaped)
        X_scaled = X_scaled.reshape(n_samples, n_channels, n_times)
        
        # Add channel dimension for CNN: (n_samples, 1, n_channels, n_times)
        X_scaled = X_scaled[:, np.newaxis, :, :]
        
        self.X = torch.FloatTensor(X_scaled)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_cnn_lstm(X_train, X_val, y_train, y_val, n_channels, n_classes, device, epochs=25):
    """Train CNN-LSTM model."""
    # Create datasets with normalization
    train_dataset = CNNLSTMDataset(X_train, y_train)
    val_dataset = CNNLSTMDataset(X_val, y_val, scaler=train_dataset.scaler)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Create model
    model = CNNLSTM(n_channels=n_channels, n_classes=n_classes)
    model.to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0
    
    print(f"\nTraining CNN-LSTM for {epochs} epochs...")
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
    print(classification_report(val_true, val_preds, target_names=CLASS_NAMES))
    
    return model, val_acc


def main(base_path, max_subjects=5):
    """Main training function."""
    print("="*60)
    print("Training CNN-LSTM Model on Validation Data")
    print("="*60)
    
    # Load data
    X_all, y_labels_all, y_all, unique_labels = load_cnn_lstm_data(base_path, max_subjects)
    if X_all is None:
        return
    
    # Split data: 80% train, 20% validation
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
    
    # Train model
    model, val_acc = train_cnn_lstm(X_train, X_val, y_train, y_val, n_channels, n_classes, device)
    
    print(f"\nSaved CNN-LSTM model to '../models/best_model.pth'")
    print(f"Final validation accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN-LSTM model on validation data")
    parser.add_argument("--data-path", type=str, required=True,
                       help="Path to EDF data directory")
    parser.add_argument("--max-subjects", type=int, default=5,
                       help="Maximum number of subjects to process")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_path):
        print(f"ERROR: Data path does not exist: {args.data_path}")
        sys.exit(1)
    
    main(
        base_path=args.data_path,
        max_subjects=args.max_subjects
    )
