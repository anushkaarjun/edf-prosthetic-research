#!/usr/bin/env python3
"""
FastAPI server for real-time EEG motor imagery predictions.
Serves predictions from trained CSP+SVM or EEGNet models.
"""
import os
import sys
import glob
import numpy as np
import mne
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import warnings

warnings.filterwarnings('ignore')

# Import model utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
try:
    from edf_ml_model.data_utils import get_run_number, annotation_to_motion
except ImportError:
    import importlib.util
    data_utils_path = os.path.join(os.path.dirname(__file__), "src", "edf_ml_model", "data_utils.py")
    spec = importlib.util.spec_from_file_location("data_utils", data_utils_path)
    data_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data_utils)
    get_run_number = data_utils.get_run_number
    annotation_to_motion = data_utils.annotation_to_motion

# Configuration
target_sfreq = 250
tmin, tmax = -0.5, 0.5
freq_low, freq_high = 8., 30.

app = FastAPI(title="EEG Motor Imagery API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model storage
models = {}
model_type = None  # 'csp_svm' or 'eegnet'

# Request/Response models
class EEGDataRequest(BaseModel):
    """Request model for EEG data."""
    channels: List[List[float]]  # List of channel data (64 channels, ~125 samples each)
    sample_rate: float = 250.0

class PredictionResponse(BaseModel):
    """Response model for predictions."""
    probabilities: List[float]  # Probabilities for each class
    predicted_class: int
    predicted_label: str
    classes: List[str]

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_type: Optional[str]
    classes: Optional[List[str]]


def load_csp_svm_model(model_path: str, csp_path: str):
    """Load CSP+SVM model from saved files."""
    import pickle
    from sklearn.svm import SVC
    from mne.decoding import CSP
    
    with open(model_path, 'rb') as f:
        svm = pickle.load(f)
    with open(csp_path, 'rb') as f:
        csp = pickle.load(f)
    
    return csp, svm


def load_eegnet_model(model_path: str, device='cpu'):
    """Load EEGNet model from saved file."""
    # Import EEGNet class
    sys.path.insert(0, os.path.dirname(__file__))
    from run_eegnet import EEGNet
    
    # Load model config from path or use defaults
    n_classes = 5  # Adjust based on your model
    n_channels = 64
    n_samples = 125
    
    model = EEGNet(n_classes=n_classes, Chans=n_channels, Samples=n_samples)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    return model, device


def preprocess_eeg_data(channels: List[List[float]], sample_rate: float = 250.0):
    """
    Preprocess EEG data to match training format.
    Returns: numpy array of shape (1, n_channels, n_samples)
    """
    # Convert to numpy array
    data = np.array(channels, dtype=np.float32)
    
    # Ensure correct shape: (n_channels, n_samples)
    if data.shape[0] > data.shape[1]:
        data = data.T
    
    # Resample if needed
    if sample_rate != target_sfreq:
        # Create MNE Raw object for resampling
        info = mne.create_info(ch_names=[f'EEG{i+1}' for i in range(data.shape[0])], 
                              sfreq=sample_rate, ch_types='eeg')
        raw = mne.io.RawArray(data, info, verbose=False)
        raw.resample(target_sfreq, npad="auto", verbose=False)
        data = raw.get_data()
    
    # Bandpass filter
    info = mne.create_info(ch_names=[f'EEG{i+1}' for i in range(data.shape[0])], 
                          sfreq=target_sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.filter(freq_low, freq_high, verbose=False, fir_design='firwin')
    data = raw.get_data()
    
    # Crop to 0.5 seconds (125 samples at 250 Hz)
    if data.shape[1] > 125:
        data = data[:, :125]
    elif data.shape[1] < 125:
        # Pad with zeros if too short
        padding = np.zeros((data.shape[0], 125 - data.shape[1]), dtype=np.float32)
        data = np.concatenate([data, padding], axis=1)
    
    # Normalize to [-1, 1] range
    data_max = np.abs(data).max()
    if data_max > 0:
        data = data / (data_max + 1e-8)
    
    # Reshape for model: (1, n_channels, n_samples) for EEGNet
    # or (1, n_channels, n_samples) for CSP+SVM
    return data.reshape(1, data.shape[0], data.shape[1])


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        model_loaded=len(models) > 0,
        model_type=model_type,
        classes=models.get('classes', None)
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: EEGDataRequest):
    """Predict motor imagery class from EEG data."""
    if not models:
        raise HTTPException(status_code=503, detail="Model not loaded. Please load a model first.")
    
    try:
        # Preprocess data
        processed_data = preprocess_eeg_data(request.channels, request.sample_rate)
        
        if model_type == 'csp_svm':
            # CSP+SVM prediction
            csp = models['csp']
            svm = models['svm']
            classes = models['classes']
            
            # Transform with CSP
            X_csp = csp.transform(processed_data)
            
            # Get probabilities
            probs = svm.predict_proba(X_csp)[0]
            predicted_idx = np.argmax(probs)
            predicted_label = classes[predicted_idx]
            
        elif model_type == 'eegnet':
            # EEGNet prediction
            model = models['model']
            device = models['device']
            classes = models['classes']
            
            # Convert to tensor and add channel dimension
            X_tensor = torch.tensor(processed_data, dtype=torch.float32).unsqueeze(1).to(device)
            
            # Predict
            with torch.no_grad():
                output = model(X_tensor)
                probs = torch.softmax(output, dim=1)[0].cpu().numpy()
                predicted_idx = np.argmax(probs)
                predicted_label = classes[predicted_idx]
        else:
            raise HTTPException(status_code=500, detail="Unknown model type")
        
        return PredictionResponse(
            probabilities=probs.tolist(),
            predicted_class=int(predicted_idx),
            predicted_label=predicted_label,
            classes=classes
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/load_model")
async def load_model(model_type_param: str, model_path: str, csp_path: Optional[str] = None):
    """
    Load a trained model.
    
    Args:
        model_type_param: 'csp_svm' or 'eegnet'
        model_path: Path to model file
        csp_path: Path to CSP file (required for CSP+SVM)
    """
    """Load a trained model."""
    global models, model_type
    
    try:
        if model_type_param == 'csp_svm':
            if not csp_path:
                raise HTTPException(status_code=400, detail="csp_path required for CSP+SVM model")
            csp, svm = load_csp_svm_model(model_path, csp_path)
            models = {
                'csp': csp,
                'svm': svm,
                'classes': ['Left Hand', 'Right Hand', 'Both Fists', 'Both Feet', 'Rest']
            }
            model_type = 'csp_svm'
        
        elif model_type_param == 'eegnet':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model, device = load_eegnet_model(model_path, device)
            models = {
                'model': model,
                'device': device,
                'classes': ['Left Hand', 'Right Hand', 'Both Fists', 'Both Feet', 'Rest']
            }
            model_type = 'eegnet'
        
        else:
            raise HTTPException(status_code=400, detail="model_type must be 'csp_svm' or 'eegnet'")
        
        return {"status": "Model loaded successfully", "model_type": model_type}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")


@app.get("/simulate")
async def simulate():
    """Generate simulated EEG data and predictions for testing."""
    # Generate random EEG-like data (64 channels, 125 samples)
    np.random.seed()
    channels = []
    for _ in range(64):
        # Generate realistic EEG signal (mixture of frequencies)
        t = np.linspace(0, 0.5, 125)
        signal = (np.sin(2 * np.pi * 10 * t) * 50 + 
                 np.sin(2 * np.pi * 20 * t) * 30 +
                 np.random.randn(125) * 20)
        channels.append(signal.tolist())
    
    # Create request
    request = EEGDataRequest(channels=channels, sample_rate=250.0)
    
    # If model is loaded, get real prediction, otherwise return simulated
    if models:
        try:
            response = await predict(request)
            return response.dict()
        except:
            pass
    
    # Simulated probabilities
    probs = np.random.dirichlet([1, 1, 1, 1, 1])
    predicted_idx = np.argmax(probs)
    classes = ['Left Hand', 'Right Hand', 'Both Fists', 'Both Feet', 'Rest']
    
    return PredictionResponse(
        probabilities=probs.tolist(),
        predicted_class=int(predicted_idx),
        predicted_label=classes[predicted_idx],
        classes=classes
    ).dict()


if __name__ == "__main__":
    import uvicorn
    print("Starting EEG Motor Imagery API server...")
    print("API will be available at http://localhost:8000")
    print("API docs at http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)

