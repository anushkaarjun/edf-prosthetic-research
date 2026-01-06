# CNN-LSTM Model Integration with Simulator

## Overview
The simulator now supports the CNN-LSTM model for 3-class motion classification:
- **Open Left Fist**
- **Open Right Fist**  
- **Close Fists**

## Model Details

### Architecture
- **CNN Layers**: Extract temporal and spatial features
- **LSTM Layer**: Captures long-range temporal dependencies
- **Input**: 320 samples (2 seconds at 160Hz)
- **Output**: 3 motion classes

### Preprocessing
- Resamples to 160Hz (from 250Hz or other rates)
- StandardScaler normalization (channel-wise)
- No bandpass filtering (uses raw signals)
- Sliding window: 320 samples with 75% overlap

## Loading the Model

### Via API
```bash
curl -X POST "http://localhost:8000/load_model" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type_param": "cnn_lstm",
    "model_path": "/path/to/best_model.pth",
    "n_channels": 64
  }'
```

### Model Requirements
- Trained model saved as `.pth` file
- Default: 64 channels (configurable)
- Model expects input shape: `(1, 1, n_channels, 320)`

## React Component

The React component automatically:
- Detects CNN-LSTM model type from API
- Updates to show 3 classes instead of 5
- Adjusts UI colors and labels accordingly
- Works with validation data or real-time predictions

## Data Format

### Input to API
```json
{
  "channels": [[...], [...], ...],  // 64 channels, any sample rate
  "sample_rate": 250.0
}
```

### Processing Pipeline
1. Resample to 160Hz if needed
2. Apply StandardScaler (channel-wise)
3. Pad/trim to 320 samples
4. Reshape to `(1, 1, 64, 320)`
5. Pass through CNN-LSTM model

## Example Usage

1. **Train CNN-LSTM model** (using `cnn-lstm.ipynb`)
2. **Save model**: `torch.save(model.state_dict(), 'best_model.pth')`
3. **Start API server**: `python3 eeg_api_server.py`
4. **Load model via API** (see above)
5. **Run React app** - it will automatically detect and use CNN-LSTM

## Differences from Other Models

| Feature | CNN-LSTM | EEGNet | CSP+SVM |
|---------|----------|--------|---------|
| Classes | 3 | 5 | 5 |
| Sample Rate | 160Hz | 250Hz | 250Hz |
| Window Size | 320 samples | 125 samples | 125 samples |
| Preprocessing | StandardScaler | Normalize to [-1,1] | CSP transform |
| Filtering | None | 8-30Hz bandpass | 8-30Hz bandpass |

## Notes

- CNN-LSTM uses raw signals (no filtering)
- StandardScaler is applied per channel
- Model expects exactly 320 samples (2 seconds at 160Hz)
- React component automatically adapts to 3-class output

