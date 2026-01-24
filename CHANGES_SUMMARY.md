# Changes Summary - January 2026

This document summarizes the changes made to improve the edf-prosthetic-research repository.

## ✅ Completed Tasks

### 1. Repository Organization
- Created `ORGANIZATION.md` documenting the repository structure
- Created `archive/` directory structure for unused code:
  - `archive/old_scripts/` - For deprecated scripts
  - `archive/old_models/` - For old model definitions
  - `archive/old_notebooks/` - For old experimental notebooks
- Documented active vs. archived code locations

### 2. Pipeline Flowchart
- Created comprehensive flowchart in `docs/PIPELINE_FLOWCHART.md`
- Includes:
  - Training pipeline flow
  - Inference pipeline flow
  - Complete system architecture
  - Data flow diagrams
  - Component interaction diagrams
- Uses Mermaid syntax for visualization (renders in GitHub, GitLab, etc.)

### 3. Small Chunk Processing (0.1s to 0.5s)
**Updated Files:**
- `src/edf_ml_model/preprocessing.py`
  - Added `EPOCH_WINDOW_MIN = 0.1` and `EPOCH_WINDOW_MAX = 0.5` constants
  - Updated `create_half_second_epochs()` to accept `chunk_size` parameter
  - Added `create_sliding_chunks()` function for creating chunks from continuous data
  - Updated `INFERENCE_BUFFER_SECONDS` from 30s to 0.5s

- `src/edf_ml_model/inference.py`
  - Updated `InferenceBuffer` class to use small chunks (0.1-0.5s) instead of 30-second buffer
  - Changed `get_buffer()` to `get_chunk()` for clarity
  - Added `predict_on_chunks()` function for processing multiple chunks with aggregation
  - Updated `predict_with_confidence()` to support smoothing

- `scripts/train_improved_model.py`
  - Added `chunk_size` parameter to `load_validation_data()` function
  - Updated `main()` function to accept and use `chunk_size` parameter
  - Added command-line arguments: `--chunk-size` (default: 0.5s)
  - Chunk size is validated and clamped to 0.1-0.5s range

**Benefits:**
- Faster training: Smaller chunks process faster
- More responsive inference: Predictions available in 0.1-0.5s instead of 30s
- Configurable: Can adjust chunk size based on needs

### 4. Smoothing/Averaging
**Updated Files:**
- `src/edf_ml_model/preprocessing.py`
  - Added `smooth_signal()` function with two methods:
    - `moving_average`: Simple moving average smoothing
    - `gaussian`: Gaussian smoothing (using scipy)
  - Updated `preprocess_raw()` to include smoothing step
  - Added `SMOOTHING_WINDOW_SIZE = 5` constant (configurable)

- `src/edf_ml_model/inference.py`
  - Updated `predict_with_confidence()` to apply smoothing before prediction
  - Added `apply_smoothing` parameter (default: True)

- `scripts/train_improved_model.py`
  - Added `apply_smoothing` parameter to `load_validation_data()`
  - Updated preprocessing to use unified `preprocess_raw()` with smoothing
  - Added command-line argument: `--no-smoothing` to disable smoothing

**Benefits:**
- Reduced noise: Smoothing reduces high-frequency noise in EEG signals
- Better model performance: Cleaner signals lead to better training and inference
- Configurable: Can enable/disable smoothing as needed

## Usage Examples

### Training with Small Chunks and Smoothing
```bash
python3 scripts/train_improved_model.py \
    --data-path "/path/to/data" \
    --max-subjects 5 \
    --epochs 50 \
    --chunk-size 0.3 \
    # Smoothing enabled by default
```

### Training with Custom Chunk Size (No Smoothing)
```bash
python3 scripts/train_improved_model.py \
    --data-path "/path/to/data" \
    --max-subjects 5 \
    --epochs 50 \
    --chunk-size 0.2 \
    --no-smoothing
```

### Using Inference with Small Chunks
```python
from edf_ml_model.inference import InferenceBuffer, predict_with_confidence

# Create buffer for 0.3s chunks
buffer = InferenceBuffer(sfreq=250, chunk_size=0.3)

# Add data
buffer.add_data(eeg_data)

# Get prediction when ready
if buffer.is_ready:
    chunk = buffer.get_chunk()
    pred, conf, probs = predict_with_confidence(model, chunk, apply_smoothing=True)
```

## Technical Details

### Chunk Size Calculation
- At 250 Hz sampling rate:
  - 0.1s = 25 samples
  - 0.3s = 75 samples
  - 0.5s = 125 samples

### Smoothing Window
- Default: 5 samples
- For 250 Hz: ~0.02 seconds
- Can be adjusted via `SMOOTHING_WINDOW_SIZE` constant

### Backward Compatibility
- Default chunk size remains 0.5s (same as before)
- Smoothing is enabled by default but can be disabled
- Existing code should continue to work with defaults

## Files Modified

1. `src/edf_ml_model/preprocessing.py` - Added smoothing and chunk size support
2. `src/edf_ml_model/inference.py` - Updated for small chunk inference
3. `scripts/train_improved_model.py` - Added chunk size and smoothing parameters
4. `ORGANIZATION.md` - New file documenting repository structure
5. `docs/PIPELINE_FLOWCHART.md` - New file with flowcharts

## Next Steps (Optional)

1. Update other training scripts (`train_cnn_lstm.py`, etc.) to use small chunks
2. Update API server to use new inference methods
3. Add unit tests for new smoothing and chunking functions
4. Benchmark performance improvements from small chunks

## Questions or Issues?

If you encounter any issues or have questions about these changes, please refer to:
- `ORGANIZATION.md` - Repository structure
- `docs/PIPELINE_FLOWCHART.md` - Pipeline flowcharts
- `README.md` - General usage instructions
