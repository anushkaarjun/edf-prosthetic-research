#!/usr/bin/env python3
"""
Train models using validation data loading approach.
This script loads data the same way the API server does and trains models.
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
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from mne.decoding import CSP
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

# Import preprocessing
preprocessing_path = os.path.join(os.path.dirname(__file__), "src", "edf_ml_model", "preprocessing.py")
spec = importlib.util.spec_from_file_location("preprocessing", preprocessing_path)
preprocessing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preprocessing)
create_half_second_epochs = preprocessing.create_half_second_epochs
normalize_signal = preprocessing.normalize_signal

# Import EEGNet model
try:
    from cnn_lstm_model import CNNLSTM
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("CNN-LSTM model not found, will only train CSP+SVM")

# Configuration (matching API server)
target_sfreq = 250
tmin, tmax = -0.5, 0.5
freq_low, freq_high = 8.0, 30.0
EPOCH_WINDOW = 0.5


def load_validation_data(base_path, max_subjects=5):
    """
    Load validation data using the same approach as the API server.
    Returns X (n_samples, n_channels, n_times), y (labels), y_idx (numeric labels).
    """
    subjects = sorted(glob.glob(f"{base_path}/S*"))[:max_subjects]
    print(f"Loading data from {len(subjects)} subjects: {[os.path.basename(s) for s in subjects]}")
    
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
                
                # Filter
                raw.filter(freq_low, freq_high, verbose=False, fir_design='firwin')
                
                # Resample if needed
                if raw.info["sfreq"] != target_sfreq:
                    raw.resample(target_sfreq, npad="auto", verbose=False)
                
                # Get events
                events, event_id = mne.events_from_annotations(raw, verbose=False)
                if len(event_id) == 0:
                    continue
                
                # Create 0.5-second epochs (same as API server)
                epochs = create_half_second_epochs(raw, events, event_id)
                
                if len(epochs) == 0:
                    continue
                
                X = epochs.get_data()  # (n_epochs, n_channels, n_samples)
                y_raw = epochs.events[:, -1]
                
                # Map annotations to motion labels
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
        return None, None, None
    
    # Concatenate all epochs
    X_all = np.concatenate(X_list, axis=0).astype(np.float32)
    y_labels_all = y_labels_list
    
    print(f"\nLoaded {X_all.shape[0]} epochs from {len(subjects)} subjects")
    print(f"Data shape: {X_all.shape} (epochs, channels, samples)")
    
    # Get unique labels and create numeric mapping
    unique_labels = sorted(set(y_labels_all))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    y_all = np.array([label_map[v] for v in y_labels_all], dtype=np.int64)
    
    print(f"Classes: {unique_labels}")
    print("Class distribution:")
    for label, idx in label_map.items():
        count = np.sum(y_all == idx)
        print(f"  {label}: {count} ({100*count/len(y_all):.1f}%)")
    
    return X_all, y_labels_all, y_all, unique_labels


class EEGDataset(Dataset):
    """PyTorch dataset for EEG data."""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_eegnet(X_train, X_val, y_train, y_val, n_channels, n_classes, device, epochs=50):
    """Train EEGNet model."""
    # Import model directly to avoid loguru dependency
    import importlib.util
    model_path = os.path.join(os.path.dirname(__file__), "src", "edf_ml_model", "model.py")
    spec = importlib.util.spec_from_file_location("model", model_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    EEGMotorImageryNet = model_module.EEGMotorImageryNet
    
    n_samples = X_train.shape[2]
    
    # Normalize data to [-1, 1]
    X_train_norm, _ = normalize_signal(X_train, method="minmax")
    X_val_norm, _ = normalize_signal(X_val, method="minmax")
    
    # Create model
    model = EEGMotorImageryNet(
        n_channels=n_channels,
        n_classes=n_classes,
        n_samples=n_samples
    )
    model.to(device)
    
    # Create data loaders
    train_dataset = EEGDataset(X_train_norm, y_train)
    val_dataset = EEGDataset(X_val_norm, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0
    
    print(f"\nTraining EEGNet for {epochs} epochs...")
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)  # Shape: (batch, channels, time)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)  # Shape: (batch, channels, time)
                batch_y = batch_y.to(device)
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        val_acc = val_correct / val_total
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            os.makedirs('../models', exist_ok=True)
            torch.save(model.state_dict(), '../models/eegnet_trained.pth')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(torch.load('../models/eegnet_trained.pth'))
    
    # Evaluate on validation set
    model.eval()
    val_preds = []
    val_true = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)  # Shape: (batch, channels, time)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            val_preds.extend(predicted.cpu().numpy())
            val_true.extend(batch_y.numpy())
    
    val_acc = accuracy_score(val_true, val_preds)
    print(f"\nEEGNet Results:")
    print(f"  Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    
    return model, val_acc


def train_csp_svm(X_train, X_val, y_train, y_val, n_classes, unique_labels):
    """Train CSP+SVM model."""
    n_components = min(8, X_train.shape[1])
    
    print(f"\nTraining CSP+SVM with {n_components} CSP components...")
    
    # Apply CSP
    csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
    X_train_csp = csp.fit_transform(X_train, y_train)
    X_val_csp = csp.transform(X_val)
    
    print(f"CSP features shape: {X_train_csp.shape}")
    
    # Train SVM
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    svm.fit(X_train_csp, y_train)
    
    # Evaluate
    train_acc = svm.score(X_train_csp, y_train)
    val_acc = svm.score(X_val_csp, y_val)
    val_pred = svm.predict(X_val_csp)
    
    print(f"\nCSP+SVM Results:")
    print(f"  Train Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(classification_report(y_val, val_pred, target_names=unique_labels))
    
    # Save model
    import pickle
    os.makedirs('../models', exist_ok=True)
    with open('../models/csp_svm_model.pkl', 'wb') as f:
        pickle.dump({'csp': csp, 'svm': svm, 'classes': unique_labels}, f)
    print("\nSaved CSP+SVM model to '../models/csp_svm_model.pkl'")
    
    return csp, svm, val_acc


def main(base_path, max_subjects=5, train_eegnet_flag=True, train_csp_svm_flag=True):
    """Main training function."""
    print("="*60)
    print("Training Models on Validation Data")
    print("="*60)
    
    # Load data
    X_all, y_labels_all, y_all, unique_labels = load_validation_data(base_path, max_subjects)
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
    
    results = {}
    
    # Train CSP+SVM
    if train_csp_svm_flag:
        print("\n" + "="*60)
        print("Training CSP+SVM Model")
        print("="*60)
        csp, svm, csp_acc = train_csp_svm(X_train, X_val, y_train, y_val, n_classes, unique_labels)
        results['csp_svm'] = csp_acc
    
    # Train EEGNet
    if train_eegnet_flag:
        print("\n" + "="*60)
        print("Training EEGNet Model")
        print("="*60)
        try:
            model, eegnet_acc = train_eegnet(X_train, X_val, y_train, y_val, n_channels, n_classes, device)
            results['eegnet'] = eegnet_acc
            print("\nSaved EEGNet model to '../models/eegnet_trained.pth'")
        except Exception as e:
            print(f"Error training EEGNet: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    for model_name, acc in results.items():
        print(f"{model_name.upper()}: {acc:.4f} ({acc*100:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models on validation data")
    parser.add_argument("--data-path", type=str, required=True,
                       help="Path to EDF data directory")
    parser.add_argument("--max-subjects", type=int, default=5,
                       help="Maximum number of subjects to process")
    parser.add_argument("--eegnet", action="store_true", default=True,
                       help="Train EEGNet model")
    parser.add_argument("--no-eegnet", dest="eegnet", action="store_false",
                       help="Skip EEGNet training")
    parser.add_argument("--csp-svm", action="store_true", default=True,
                       help="Train CSP+SVM model")
    parser.add_argument("--no-csp-svm", dest="csp_svm", action="store_false",
                       help="Skip CSP+SVM training")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_path):
        print(f"ERROR: Data path does not exist: {args.data_path}")
        sys.exit(1)
    
    main(
        base_path=args.data_path,
        max_subjects=args.max_subjects,
        train_eegnet_flag=args.eegnet,
        train_csp_svm_flag=args.csp_svm
    )
