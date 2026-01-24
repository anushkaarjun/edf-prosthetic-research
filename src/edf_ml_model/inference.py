"""
Inference with small chunk processing (0.1s to 0.5s) and confidence tracking.
"""
import numpy as np
import torch
from collections import deque
from typing import Tuple, Optional, Dict, List
import mne

from .preprocessing import (
    INFERENCE_BUFFER_SECONDS, 
    EPOCH_WINDOW_MIN, 
    EPOCH_WINDOW_MAX,
    create_sliding_chunks,
    smooth_signal
)
from .data_utils import annotation_to_motion, get_run_number


class InferenceBuffer:
    """
    Buffer that accumulates small chunks (0.1s to 0.5s) for fast, responsive classification.
    """
    def __init__(self, sfreq: int = 250, chunk_size: float = 0.5):
        """
        Initialize inference buffer with small chunk processing.
        
        Args:
            sfreq: Sampling frequency in Hz
            chunk_size: Size of each chunk in seconds (0.1 to 0.5)
        """
        self.sfreq = sfreq
        # Clamp chunk_size to valid range
        self.chunk_size = max(EPOCH_WINDOW_MIN, min(EPOCH_WINDOW_MAX, chunk_size))
        self.chunk_samples = int(self.chunk_size * sfreq)
        self.data_buffer = deque(maxlen=self.chunk_samples)
        self.is_ready = False
        
    def add_data(self, data: np.ndarray):
        """Add new data to buffer."""
        if len(data.shape) == 1:
            data = data[np.newaxis, :]  # Add channel dimension
        
        for sample in data.T:  # Iterate over time
            self.data_buffer.append(sample)
            
        # Check if buffer is ready (has enough samples for one chunk)
        self.is_ready = len(self.data_buffer) >= self.chunk_samples
    
    def get_chunk(self) -> Optional[np.ndarray]:
        """Get current chunk as numpy array."""
        if not self.is_ready:
            return None
        chunk = np.array(list(self.data_buffer)[-self.chunk_samples:]).T  # (n_channels, n_samples)
        return chunk
    
    def reset(self):
        """Reset buffer."""
        self.data_buffer.clear()
        self.is_ready = False


def predict_with_confidence(
    model: torch.nn.Module, 
    X: np.ndarray, 
    device: str = "cpu",
    apply_smoothing: bool = True,
) -> Tuple[int, float, np.ndarray]:
    """
    Make prediction with confidence score on a small chunk of data.
    
    Args:
        model: Trained model
        X: Input data (n_channels, n_samples) or (1, n_channels, n_samples)
        device: 'cpu' or 'cuda'
        apply_smoothing: Whether to apply smoothing before prediction
    
    Returns:
        Tuple of (predicted_class, confidence_score, all_probabilities)
    """
    model.eval()
    
    # Apply smoothing if requested
    if apply_smoothing and len(X.shape) >= 2:
        if len(X.shape) == 2:
            X = smooth_signal(X, method="moving_average")
        else:
            # Batch dimension present
            X_smooth = np.zeros_like(X)
            for i in range(X.shape[0]):
                X_smooth[i] = smooth_signal(X[i], method="moving_average")
            X = X_smooth
    
    # Add batch dimension if needed
    if len(X.shape) == 2:
        X = X[np.newaxis, ...]
    
    X_tensor = torch.FloatTensor(X).to(device)
    
    with torch.no_grad():
        logits = model(X_tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    return predicted.item(), confidence.item(), probabilities.cpu().numpy()[0]


def predict_on_chunks(
    model: torch.nn.Module,
    data: np.ndarray,
    sfreq: int = 250,
    chunk_size: float = 0.5,
    device: str = "cpu",
    apply_smoothing: bool = True,
    aggregation: str = "majority_vote",
) -> Tuple[int, float, np.ndarray]:
    """
    Make predictions on multiple chunks and aggregate results.
    
    Args:
        model: Trained model
        data: Continuous data (n_channels, n_samples)
        sfreq: Sampling frequency in Hz
        chunk_size: Size of each chunk in seconds (0.1 to 0.5)
        device: 'cpu' or 'cuda'
        apply_smoothing: Whether to apply smoothing before prediction
        aggregation: 'majority_vote' or 'average_probabilities'
    
    Returns:
        Tuple of (predicted_class, confidence_score, all_probabilities)
    """
    # Create chunks from continuous data
    chunks = create_sliding_chunks(data, sfreq, chunk_size=chunk_size, overlap=0.0)
    
    if len(chunks) == 0:
        raise ValueError("No chunks created from data")
    
    # Get predictions for each chunk
    predictions = []
    probabilities_list = []
    
    for chunk in chunks:
        pred, conf, probs = predict_with_confidence(
            model, chunk, device=device, apply_smoothing=apply_smoothing
        )
        predictions.append(pred)
        probabilities_list.append(probs)
    
    # Aggregate predictions
    if aggregation == "majority_vote":
        # Use majority vote
        from collections import Counter
        pred_counts = Counter(predictions)
        final_pred = pred_counts.most_common(1)[0][0]
        final_confidence = pred_counts[final_pred] / len(predictions)
        # Average probabilities for the winning class
        avg_probs = np.mean(probabilities_list, axis=0)
        final_probs = avg_probs
    else:  # average_probabilities
        # Average probabilities across chunks
        avg_probs = np.mean(probabilities_list, axis=0)
        final_pred = int(np.argmax(avg_probs))
        final_confidence = float(avg_probs[final_pred])
        final_probs = avg_probs
    
    return final_pred, final_confidence, final_probs
