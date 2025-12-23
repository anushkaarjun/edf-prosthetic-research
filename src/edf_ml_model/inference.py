"""
Inference with 30-second buffer and confidence tracking.
"""
import numpy as np
import torch
from collections import deque
from typing import Tuple, Optional, Dict
import mne

from .preprocessing import INFERENCE_BUFFER_SECONDS
from .data_utils import annotation_to_motion, get_run_number


class InferenceBuffer:
    """
    Buffer that accumulates 30 seconds of data before classification.
    """
    def __init__(self, sfreq: int = 250, buffer_seconds: int = 30):
        self.sfreq = sfreq
        self.buffer_seconds = buffer_seconds
        self.buffer_samples = buffer_seconds * sfreq
        self.data_buffer = deque(maxlen=self.buffer_samples)
        self.is_ready = False
        
    def add_data(self, data: np.ndarray):
        """Add new data to buffer."""
        if len(data.shape) == 1:
            data = data[np.newaxis, :]  # Add channel dimension
        
        for sample in data.T:  # Iterate over time
            self.data_buffer.append(sample)
            
        # Check if buffer is ready (has 30 seconds)
        self.is_ready = len(self.data_buffer) >= self.buffer_samples
    
    def get_buffer(self) -> Optional[np.ndarray]:
        """Get current buffer as numpy array."""
        if not self.is_ready:
            return None
        return np.array(self.data_buffer).T  # (n_channels, n_samples)
    
    def reset(self):
        """Reset buffer."""
        self.data_buffer.clear()
        self.is_ready = False


def predict_with_confidence(
    model: torch.nn.Module, X: np.ndarray, device: str = "cpu"
) -> Tuple[int, float, np.ndarray]:
    """
    Make prediction with confidence score.
    
    Args:
        model: Trained model
        X: Input data (n_channels, n_samples) or (1, n_channels, n_samples)
        device: 'cpu' or 'cuda'
    
    Returns:
        Tuple of (predicted_class, confidence_score, all_probabilities)
    """
    model.eval()
    
    # Add batch dimension if needed
    if len(X.shape) == 2:
        X = X[np.newaxis, ...]
    
    X_tensor = torch.FloatTensor(X).to(device)
    
    with torch.no_grad():
        logits = model(X_tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    return predicted.item(), confidence.item(), probabilities.cpu().numpy()[0]
