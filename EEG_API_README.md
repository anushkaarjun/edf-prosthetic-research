# EEG Motor Imagery API Server

This API server provides real-time predictions from trained CSP+SVM or EEGNet models for the React EEG simulator.

## Setup

1. Install dependencies:
```bash
pip install fastapi uvicorn pydantic
```

Or if using the project's dependency management:
```bash
# Install from pyproject.toml (includes fastapi, uvicorn, pydantic)
```

2. Start the API server:
```bash
python eeg_api_server.py
```

The server will start at `http://localhost:8000`

## API Endpoints

### Health Check
```bash
GET /health
```
Returns the status of the API and whether a model is loaded.

### Load Model
```bash
POST /load_model
Body: {
  "model_type": "csp_svm" | "eegnet",
  "model_path": "/path/to/model.pkl",
  "csp_path": "/path/to/csp.pkl"  # Only for CSP+SVM
}
```

### Predict
```bash
POST /predict
Body: {
  "channels": [[...], [...], ...],  # 64 channels, each with ~125 samples
  "sample_rate": 250.0
}
```

### Simulate (for testing)
```bash
GET /simulate
```
Returns simulated predictions (useful when no model is loaded).

## React Component Integration

The updated React component (`eeg_simulator_react.jsx`) will:
1. Automatically detect if the API is running
2. Use real model predictions when available
3. Fall back to simulated data when API is unavailable

### Motor Classes

The component now matches the Python model classes:
- **Left Hand** (blue)
- **Right Hand** (red)
- **Both Feet** (green)
- **Both Fists** (orange) - *Updated from "Tongue"*
- **Rest** (gray)

## Usage Example

1. Train a model using `run_eegnet.py` or `run_csp_svm_optimized.py`
2. Save the model (you may need to modify the scripts to save models)
3. Start the API server: `python eeg_api_server.py`
4. Load the model via the API or modify the server to auto-load
5. Run your React app - it will automatically connect to the API

## Notes

- The API expects 64 channels of EEG data
- Each channel should have ~125 samples (0.5 seconds at 250 Hz)
- Data is automatically preprocessed (filtered, resampled, normalized) to match training format
- The API handles both CSP+SVM and EEGNet models

