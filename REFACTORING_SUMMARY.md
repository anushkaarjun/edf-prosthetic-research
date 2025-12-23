# Refactoring Summary

This document summarizes all the changes made to improve the EEG motor imagery classification system.

## New Modules Created

### 1. `src/edf_ml_model/preprocessing.py`
- **Normalization**: Signals normalized to [-1, 1] range for consistent weight scaling
- **Spectral Analysis**: Power spectral density computation to validate bandpass filtering
- **Bad Data Detection**: Statistical methods to detect flat channels, outliers, NaN/Inf values
- **Data Cleaning**: Automatic bad channel detection and interpolation
- **0.5-second Epochs**: Support for half-second training windows
- **Complete Pipeline**: `preprocess_raw()` function that handles all preprocessing steps

### 2. `src/edf_ml_model/data_utils.py`
- **Consolidated Functions**: `get_run_number()` and `annotation_to_motion()` moved here to eliminate duplication

### 3. `src/edf_ml_model/model.py`
- **EEGMotorImageryNet**: CNN-based neural network optimized for 0.5-second windows (125 samples at 250Hz)
- **Weight Freezing**: `freeze_backbone()` method to freeze convolutional layers while fine-tuning classifier
- **Training Function**: `train_model()` with support for weight freezing at specified epochs

### 4. `src/edf_ml_model/inference.py`
- **InferenceBuffer**: 30-second buffer that must be filled before classification
- **Confidence Tracking**: `predict_with_confidence()` returns predicted class, confidence score, and all probabilities

### 5. `src/edf_ml_model/visualization.py`
- **Spectral Analysis Plot**: Visualization of power spectral density with motor imagery band highlighted
- **Classification with Confidence**: Bar plot showing probabilities with confidence annotation
- **Motor Movement Visualization**: 2D body diagram highlighting predicted movement with confidence-based transparency

### 6. `src/edf_ml_model/realtime_dashboard.py` (PyQt5)
- **Real-time EEG Plot**: Live visualization of EEG signals (showing subset of channels)
- **Probability Timeline**: Time-series plot of classification probabilities
- **Radar Chart**: Spider chart showing current class probabilities
- **Statistics Panel**: Current time, prediction, confidence, and classification count
- **30-second Buffer**: Only classifies when buffer has 30 seconds of data

### 7. `src/edf_ml_model/web_dashboard.py` (Gradio)
- **Web Interface**: Hugging Face Spaces-compatible dashboard
- **File Upload**: Support for EDF file upload and processing
- **Interactive Plots**: Plotly-based probability timeline and radar chart
- **Real-time Mode**: Toggle for continuous processing

## New Scripts

### `train_model.py`
- **Training Workflow**:
  1. Phase 1: Train all weights for initial epochs
  2. Phase 2: Freeze backbone, fine-tune classifier only
- **0.5-second Epochs**: Uses half-second windows for training
- **Normalized Data**: All data normalized to [-1, 1] range
- **Subject-specific**: Trains one model per subject

## Key Features Implemented

✅ **Weight Freezing**: Train → Freeze backbone → Fine-tune classifier workflow  
✅ **30-second Buffer**: Inference requires minimum 30 seconds of data  
✅ **0.5-second Training**: All training uses half-second epochs  
✅ **Spectral Analysis**: Validation of bandpass filtering effectiveness  
✅ **Bad Data Cleaning**: Automatic detection and cleaning of corrupted channels  
✅ **Normalization to [-1, 1]**: Consistent signal scaling  
✅ **Confidence Metrics**: All predictions include confidence scores  
✅ **Real-time Visualization**: PyQt5 dashboard with multiple plots  
✅ **Web Dashboard**: Gradio interface for Hugging Face deployment  
✅ **Radar Charts**: Multi-class probability visualization  
✅ **Consolidated Functions**: Eliminated duplicate code  

## Configuration Constants

All key parameters are centralized in `preprocessing.py`:
- `TARGET_SFREQ = 250` Hz
- `FREQ_LOW, FREQ_HIGH = 8.0, 30.0` Hz (motor imagery band)
- `EPOCH_WINDOW = 0.5` seconds
- `INFERENCE_BUFFER_SECONDS = 30` seconds

## Updated Dependencies

Added to `pyproject.toml`:
- `mne>=1.0.0`
- `scikit-learn>=1.0.0`
- `scipy>=1.9.0`
- `matplotlib>=3.5.0`
- `torch>=1.12.0`
- `PyQt5>=5.15.0`
- `gradio>=3.0.0`
- `plotly>=5.0.0`

## Usage Examples

### Training
```bash
python train_model.py --data-path ./data --max-subjects 5 --freeze-after 30 --epochs 50
```

### Real-time Dashboard (PyQt5)
```python
from edf_ml_model.realtime_dashboard import run_dashboard
run_dashboard(model, preprocess_fn, class_names)
```

### Web Dashboard (Gradio)
```python
from edf_ml_model.web_dashboard import WebDashboard
dashboard = WebDashboard(model, preprocess_fn, class_names)
dashboard.launch(share=True)  # For public URL
```

## Backward Compatibility

- `run_csp_svm.py` updated to use `data_utils` module but maintains CSP+SVM approach
- Old scripts continue to work, but can be gradually migrated to use new modules

