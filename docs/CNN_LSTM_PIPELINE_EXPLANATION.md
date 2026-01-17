# CNN-LSTM Pipeline: End-to-End Explanation

## Overview
This document provides a comprehensive explanation of the end-to-end pipeline for training a deep learning model to classify motion events from time-series data (EEG/EMG signals).

## 1. Dataset Loading and Preprocessing

### Reading EDF Files
- **Purpose**: Load EEG/EMG signal data stored in EDF (European Data Format) files
- **Function**: `get_edf_files()` loads all `.edf` files from a specified directory and returns the list of file paths
- **Data Structure**: Each file contains signal data for different subjects

### Event Extraction
- **MNE Library**: Python library for processing EEG/MEG data
  - `mne.io.read_raw_edf()`: Reads raw EDF files
  - `mne.events_from_annotations()`: Extracts event annotations that label motion segments
- **Event Types**: Motion events such as:
  - "Open Left Fist"
  - "Open Right Fist"
  - "Close Both Fists"

### Sliding Window for Temporal Segmentation
- **Window Size**: `SLIDING_WINDOW = 320` samples (2 seconds at 160Hz)
- **Overlap**: `WINDOW_STEP = 80` samples (75% overlap)
- **Purpose**: 
  - Break continuous signals into smaller, manageable chunks
  - Create overlapping windows to capture temporal patterns
  - Essential for sequential data processing

### Standardization
- **Method**: `StandardScaler` normalization
- **Process**: Each data segment is normalized so that all features (channels) have:
  - Mean = 0
  - Standard deviation = 1
- **Why**: Neural networks perform better when features are on similar scales

## 2. Dataset Class (PyTorch Dataset)

### PreloadedMotionDataset
- **Type**: Custom PyTorch Dataset class
- **Purpose**: Prepare preprocessed motion data for training
- **Features**:
  - **Data Augmentation**: 
    - Adds random noise (0.02 * Gaussian noise)
    - Random scaling (0.9x to 1.1x)
  - **Benefits**: Helps model generalize by making it less sensitive to small input variations
- **Method**: `__getitem__()` returns data and labels as PyTorch tensors for easy model input

## 3. Model Architecture

### CNN-LSTM Hybrid Model
Combines the strengths of Convolutional Neural Networks (CNNs) and Long Short-Term Memory Networks (LSTMs).

#### CNN Layers
1. **Conv1D (Convolutional Layer)**
   - Extracts local temporal patterns from each time step
   - First layer captures low-level features (edges, peaks in signals)
   - Architecture:
     - Conv1: `Conv2d(1, 32, (1, 5))` - extracts features from single time dimension
     - Conv2: `Conv2d(32, 64, (n_channels, 5))` - spatial convolution across channels

2. **Batch Normalization (BN)**
   - Normalizes layer outputs after each convolution
   - Benefits:
     - Improves training speed
     - Increases training stability
     - Acts as a form of regularization

3. **Max Pooling**
   - Reduces spatial dimensions
   - Keeps only the most important features
   - Reduces computational complexity

4. **Dropout**
   - Randomly sets neurons to zero during training
   - Prevents overfitting by reducing model complexity

#### LSTM Layer
- **Purpose**: Learn long-range temporal dependencies
- **Why Important**: Motion patterns may have sequences across time steps that influence classification
- **Architecture**: 
  - Input size: 64 (from CNN output)
  - Hidden size: 128
  - Uses last LSTM output for classification

#### Fully Connected Layer
- **Type**: Linear layer
- **Input**: 128 (LSTM hidden size)
- **Output**: 3 classes (motion events)

## 4. Loss Function and Optimization

### Cross-Entropy Loss
- **Purpose**: Compute difference between predicted probabilities and true labels
- **Output**: Probability distribution across 3 classes
- **Class Weights**: 
  - `compute_class_weights()` calculates weights based on class frequency
  - Important for handling imbalanced datasets
  - Ensures model doesn't bias toward frequent classes

### Optimizer
- **Algorithm**: Adam (Adaptive Moment Estimation)
- **Advantages**: 
  - Combines benefits of Momentum and RMSprop
  - Adaptive learning rate for each parameter
  - Efficient convergence

### Learning Rate Scheduling
- **Method**: Cyclic Learning Rate (CyclicLR)
- **Behavior**: Periodically increases and decreases learning rate
- **Benefits**:
  - Helps escape local minima
  - Improves overall performance
  - Better training dynamics

## 5. Training and Evaluation

### Training Loop
1. **Forward Pass**: Data flows through model to produce predictions
2. **Loss Calculation**: Cross-entropy loss computed
3. **Backward Pass**: Gradients calculated using backpropagation
4. **Optimization**: Adam optimizer updates model weights
5. **Learning Rate Update**: CyclicLR scheduler adjusts learning rate

### Early Stopping
- **Purpose**: Prevent overfitting
- **Mechanism**: Halts training if validation accuracy doesn't improve for `PATIENCE` epochs
- **Model Saving**: Best model (highest validation accuracy) is saved and reloaded after training

### Evaluation
- **Test Set**: Separate data not seen during training
- **Metrics Computed**:
  - Accuracy
  - Predictions and labels for detailed analysis

### Metrics

#### Accuracy
- Percentage of correctly classified samples
- Formula: `(Correct Predictions / Total Predictions) * 100`

#### Confusion Matrix
- **Purpose**: Detailed breakdown of classification performance
- **Components**:
  - True Positives (TP): Correctly predicted positive cases
  - False Positives (FP): Incorrectly predicted as positive
  - True Negatives (TN): Correctly predicted negative cases
  - False Negatives (FN): Incorrectly predicted as negative

#### Classification Report
- **Metrics per class**:
  - **Precision**: `TP / (TP + FP)` - Accuracy of positive predictions
  - **Recall**: `TP / (TP + FN)` - Ability to find all positive cases
  - **F1-Score**: Harmonic mean of precision and recall
  - **Support**: Number of true instances for each class

## 6. Visualizations

### Accuracy Plot
- **Content**: Training and validation accuracy over epochs
- **Purpose**: Visualize model performance and detect overfitting

### Confusion Matrix
- **Content**: Grid showing predicted vs. actual classes
- **Purpose**: Understand classification errors and class confusion

### Classification Report
- **Content**: Precision, recall, F1-score for each class
- **Purpose**: Detailed performance summary

## 7. Key Concepts Used

1. **Supervised Learning**
   - Model learns from labeled data (motion events)
   - Predicts labels for new, unseen data

2. **Convolutional Neural Networks (CNNs)**
   - Extract local patterns from data
   - Effective for spatial and temporal feature extraction

3. **Long Short-Term Memory (LSTM)**
   - Captures temporal dependencies
   - Remembers information over long sequences

4. **Cross-Entropy Loss**
   - Standard loss function for classification
   - Measures probability distribution differences

5. **Adam Optimizer**
   - Efficient optimization algorithm
   - Combines Momentum and RMSprop benefits

6. **Data Augmentation**
   - Adds noise and scaling variations
   - Improves model generalization

7. **Early Stopping**
   - Prevents overfitting
   - Stops training when validation performance plateaus

8. **Cyclic Learning Rate (CyclicLR)**
   - Varies learning rate cyclically
   - Helps escape local minima

9. **Confusion Matrix and Classification Report**
   - Comprehensive evaluation tools
   - Provides detailed performance insights

## Implementation Details

### Configuration Parameters
```python
SLIDING_WINDOW = 320     # 2 seconds at 160Hz
WINDOW_STEP = 80         # 75% overlap
N_CLASSES = 3
BATCH_SIZE = 64
EPOCHS = 25
PATIENCE = 10
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.4
```

### Model Architecture Summary
```
Input: (batch, 1, 64 channels, 320 time samples)
  ↓
Conv2d(1 → 32) + BatchNorm + ReLU
  ↓
Conv2d(32 → 64) + BatchNorm + ReLU
  ↓
MaxPool2d + Dropout
  ↓
Reshape → (batch, time, features)
  ↓
LSTM(64 → 128)
  ↓
Fully Connected (128 → 3 classes)
  ↓
Output: (batch, 3) - Class probabilities
```

## Conclusion

This pipeline leverages deep learning techniques (CNN + LSTM) to classify motion events from time-series data. Key components include:

- **Robust Preprocessing**: EDF file reading, event extraction, sliding windows, standardization
- **Hybrid Architecture**: CNN for feature extraction, LSTM for temporal dependencies
- **Training Best Practices**: Data augmentation, class weighting, early stopping, learning rate scheduling
- **Comprehensive Evaluation**: Accuracy, confusion matrix, classification metrics

The combination of CNN and LSTM allows the model to capture both local temporal patterns and long-range dependencies, making it well-suited for motion classification from EEG/EMG signals.
