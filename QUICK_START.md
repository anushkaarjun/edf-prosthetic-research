# Quick Start Guide

## Installation

Install dependencies:
```bash
pip install -e .
# Or install manually:
pip install mne scikit-learn scipy matplotlib torch PyQt5 gradio plotly numpy
```

## Training a Model

### Neural Network (with weight freezing):
```bash
python train_model.py --data-path ./data --max-subjects 5 --freeze-after 30 --epochs 50
```

This will:
1. Train all weights for 30 epochs
2. Freeze backbone layers
3. Fine-tune classifier for remaining 20 epochs
4. Use 0.5-second epochs with normalized data

### CSP+SVM (fast baseline):
```bash
python run_csp_svm.py --data-path ./data --max-subjects 5
```

## Running Real-time Dashboard (PyQt5)

```python
import torch
from edf_ml_model.model import EEGMotorImageryNet
from edf_ml_model.realtime_dashboard import run_dashboard
from edf_ml_model.preprocessing import preprocess_raw

# Load your trained model
model = EEGMotorImageryNet(n_channels=64, n_classes=4, n_samples=125)
model.load_state_dict(torch.load('path/to/model.pth'))
model.eval()

# Define preprocessing function
def preprocess_fn(data):
    # Your preprocessing logic here
    return torch.FloatTensor(data)

# Define class names
class_names = ['Rest', 'Left Hand', 'Right Hand', 'Both Fists', 'Both Feet']

# Run dashboard
run_dashboard(model, preprocess_fn, class_names)
```

## Running Web Dashboard (Gradio)

```python
from edf_ml_model.web_dashboard import WebDashboard

dashboard = WebDashboard(model, preprocess_fn, class_names)
dashboard.launch(share=True)  # share=True creates public URL
```

## Key Features

- **0.5-second training windows**: All training uses half-second epochs
- **30-second inference buffer**: Classification requires 30 seconds of data
- **Weight freezing**: Train → Freeze → Fine-tune workflow
- **Normalized data**: All signals normalized to [-1, 1] range
- **Bad data cleaning**: Automatic detection and cleaning
- **Spectral analysis**: Validation of bandpass filtering
- **Confidence metrics**: All predictions include confidence scores
- **Real-time visualization**: PyQt5 dashboard with multiple plots
- **Web dashboard**: Gradio interface for deployment

## Module Overview

- `preprocessing.py`: Data normalization, filtering, cleaning, spectral analysis
- `data_utils.py`: Consolidated utility functions
- `model.py`: Neural network with weight freezing
- `inference.py`: 30-second buffer and confidence tracking
- `visualization.py`: Plotting functions
- `realtime_dashboard.py`: PyQt5 desktop dashboard
- `web_dashboard.py`: Gradio web dashboard

