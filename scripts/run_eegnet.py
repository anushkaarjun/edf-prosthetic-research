#!/usr/bin/env python3
"""
Standalone script to run EEGNet training and get test accuracy.
"""

import os
import sys
import glob
import argparse
import warnings
import numpy as np
import mne
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Suppress MNE warnings about dropped epochs (we handle this explicitly)
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*epochs were dropped.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*drop_log.*')

# Configuration
DEFAULT_BASE_PATH = "/content/drive/MyDrive/files 2"  # Default for Colab
MAX_SUBJECTS = 20
target_sfreq = 250
tmin, tmax = -0.5, 0.5  # Include baseline period (-0.5 to 0) and 0.5s task period
freq_low, freq_high = 8., 30.

# Import consolidated utility functions
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


class EEGNet(nn.Module):
    """
    EEGNet architecture adapted for short windows (0.5 seconds).
    Reduced kernel sizes to better match 125-126 sample windows.
    """
    def __init__(self, n_classes, Chans=64, Samples=125, dropout_rate=0.25):
        super(EEGNet, self).__init__()
        self.n_classes = n_classes
        
        # Block 1: Temporal Convolution (reduced kernel size for short windows)
        # Original uses (1, 64) but that's too large for 125 samples
        # Use (1, 32) instead - still captures temporal patterns but fits better
        self.conv1 = nn.Conv2d(1, 16, (1, 32), padding=(0, 16), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(16, momentum=0.01, affine=True, eps=1e-3)
        
        # Block 2: Depthwise Convolution (spatial)
        self.conv2 = nn.Conv2d(16, 32, (Chans, 1), groups=16, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(32, momentum=0.01, affine=True, eps=1e-3)
        self.elu1 = nn.ELU()
        # Reduced pooling to preserve more information
        self.avgpool1 = nn.AvgPool2d((1, 2))  # Changed from 4 to 2
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Block 3: Separable Convolution
        # Reduced kernel size for short windows
        self.conv3_sep = nn.Conv2d(32, 32, (1, 8), padding=(0, 4), groups=32, bias=False)
        self.conv3_point = nn.Conv2d(32, 32, (1, 1), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(32, momentum=0.01, affine=True, eps=1e-3)
        self.elu2 = nn.ELU()
        # Reduced pooling
        self.avgpool2 = nn.AvgPool2d((1, 4))  # Changed from 8 to 4
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Use adaptive pooling to handle variable sizes robustly
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(32, n_classes)
        
    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.batchnorm1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.elu1(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)
        
        # Block 3
        x = self.conv3_sep(x)
        x = self.conv3_point(x)
        x = self.batchnorm3(x)
        x = self.elu2(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)
        
        # Adaptive pooling and classify
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class EEGDataset(Dataset):
    def __init__(self, X, y, normalize=True):
        # Convert to torch tensors
        # X shape: (n_samples, n_channels, n_timepoints)
        # Normalize to [-1, 1] range for better training stability
        if normalize:
            # Normalize per sample to [-1, 1]
            X_norm = np.zeros_like(X)
            for i in range(len(X)):
                x = X[i]
                x_min, x_max = x.min(), x.max()
                if x_max - x_min > 1e-8:
                    X_norm[i] = 2 * (x - x_min) / (x_max - x_min) - 1
                else:
                    X_norm[i] = x
            self.X = torch.FloatTensor(X_norm)
        else:
            self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # Add channel dimension for EEGNet: (channels, time) -> (1, channels, time)
        x = self.X[idx].unsqueeze(0)
        return x, self.y[idx]


def load_subject_data(subject_path: str):
    """Load and preprocess all EDF files for a subject."""
    subject_id = os.path.basename(subject_path)
    subj_files = sorted(glob.glob(f"{subject_path}/*.edf"))
    
    X_list = []
    y_list = []
    
    for file_path in subj_files:
        run = get_run_number(file_path)
        try:
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            raw.filter(freq_low, freq_high, verbose=False, fir_design='firwin')
            
            if raw.info["sfreq"] != target_sfreq:
                raw.resample(target_sfreq, npad="auto")
            
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            if len(event_id) == 0:
                continue
            
            # Create epochs with lenient criteria to avoid dropping epochs
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                epochs = mne.Epochs(
                    raw, events, event_id=event_id,
                    tmin=tmin, tmax=tmax,
                    baseline=(-0.5, 0),
                    preload=True, 
                    verbose=False,
                    reject=None,  # Don't reject epochs based on amplitude
                    flat=None,    # Don't reject flat channels
                    reject_by_annotation=False  # Don't reject based on annotations
                )
            
            # Check if epochs are empty before cropping
            if len(epochs) == 0:
                continue
            
            # Crop to task period (0-0.5s)
            try:
                epochs.crop(tmin=0, tmax=0.5)
            except Exception as e:
                # If cropping fails (e.g., empty epochs), skip this file
                continue
            
            if len(epochs) == 0:
                continue
            
            X = epochs.get_data()  # Shape: (n_epochs, n_channels, n_samples)
            y_raw = epochs.events[:, -1]
            y_mapped = [annotation_to_motion(c, run) for c in y_raw]
            
            valid_idx = [i for i, v in enumerate(y_mapped) if v != "Unknown"]
            if len(valid_idx) > 0:
                X_list.append(X[valid_idx])
                y_list.extend([y_mapped[i] for i in valid_idx])
        except Exception as e:
            print(f"  Error processing {os.path.basename(file_path)}: {e}")
            continue
    
    if len(X_list) > 0:
        X_subj = np.concatenate(X_list, axis=0).astype(np.float32)
        return {'X': X_subj, 'y': y_list}
    return None


def main(base_path=None):
    if base_path is None:
        base_path = DEFAULT_BASE_PATH
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get subjects
    subjects = sorted(glob.glob(f"{base_path}/S*"))[:MAX_SUBJECTS]
    print(f"Using {len(subjects)} subjects: {[os.path.basename(s) for s in subjects]}")
    
    # Process data per subject
    subject_data = {}
    for subj in subjects:
        print(f"\n=== Processing Subject {os.path.basename(subj)} ===")
        data = load_subject_data(subj)
        if data is not None:
            subject_data[os.path.basename(subj)] = data
            print(f"  Processed {data['X'].shape[0]} epochs, shape {data['X'].shape}")
    
    if len(subject_data) == 0:
        print("ERROR: No data processed. Check file paths and data availability.")
        return None, None
    
    # Train models per subject
    subject_results = {}
    
    for subj, data in subject_data.items():
        print(f"\n{'='*50}")
        print(f"Training EEGNet for Subject {subj}")
        print(f"{'='*50}")
        
        X = data['X']  # Shape: (n_epochs, n_channels, n_samples)
        y_labels = data['y']
        
        unique_labels = sorted(set(y_labels))
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        y = np.array([label_map[v] for v in y_labels], dtype=np.int64)
        
        print(f"Data shape: {X.shape}, Classes: {unique_labels}")
        print("Class distribution:")
        for label, idx in label_map.items():
            count = np.sum(y == idx)
            print(f"  {label}: {count} ({100*count/len(y):.1f}%)")
        
        # Split data into train, validation, and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
        )
        
        print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        
        # Create datasets and loaders
        train_dataset = EEGDataset(X_train, y_train, normalize=True)
        val_dataset = EEGDataset(X_val, y_val, normalize=True)
        test_dataset = EEGDataset(X_test, y_test, normalize=True)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # Get model parameters
        n_channels = X.shape[1]
        n_samples = X.shape[2]
        n_classes = len(unique_labels)
        
        print(f"Model parameters: n_classes={n_classes}, n_channels={n_channels}, n_samples={n_samples}")
        
        # Create model
        model = EEGNet(n_classes=n_classes, Chans=n_channels, Samples=n_samples, dropout_rate=0.5).to(device)
        
        # Training setup with better hyperparameters
        criterion = nn.CrossEntropyLoss()
        # Higher learning rate for faster learning
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.7, patience=7, min_lr=1e-6
        )
        
        # Train model with early stopping
        num_epochs = 150
        best_val_acc = 0.0
        best_model_state = None
        patience = 15  # More patience to allow learning
        patience_counter = 0
        
        print("Training with early stopping...")
        for epoch in range(num_epochs):
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
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/len(train_loader):.4f}, "
                      f"Train Acc: {train_acc*100:.2f}%, Val Acc: {val_acc*100:.2f}%")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"Loaded best model with validation accuracy: {best_val_acc*100:.2f}%")
        
        # Final evaluation
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        test_acc = accuracy_score(all_labels, all_preds)
        
        print(f"\nSubject {subj} Results:")
        print(f"  Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
        print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        
        subject_results[subj] = {
            'val_acc': best_val_acc,
            'test_acc': test_acc,
            'y_test': all_labels,
            'y_pred': all_preds,
            'unique_labels': unique_labels
        }
    
    # Overall results
    print(f"\n{'='*50}")
    print("OVERALL RESULTS")
    print(f"{'='*50}")
    overall_val_acc = np.mean([r['val_acc'] for r in subject_results.values()])
    overall_test_acc = np.mean([r['test_acc'] for r in subject_results.values()])
    print(f"Average Validation Accuracy: {overall_val_acc:.4f} ({overall_val_acc*100:.2f}%)")
    print(f"Average Test Accuracy: {overall_test_acc:.4f} ({overall_test_acc*100:.2f}%)")
    
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
        print(classification_report(result['y_test'], result['y_pred'], 
                                  target_names=result['unique_labels'], digits=3))
    
    return overall_test_acc, subject_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EEGNet training and test accuracy")
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
    
    if not os.path.exists(BASE_PATH):
        print(f"\nERROR: Data path does not exist: {BASE_PATH}")
        print("\nPlease provide the correct path to your EDF data files using --data-path")
        print("Example: python run_eegnet.py --data-path /path/to/your/data")
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

