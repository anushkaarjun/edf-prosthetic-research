# Repository Organization

This document describes the organization of the edf-prosthetic-research repository.

## Directory Structure

### Active Code (Regularly Used)

- **`src/edf_ml_model/`** - Core ML models and utilities
  - `preprocessing.py` - Data preprocessing with smoothing and chunking
  - `model.py` - Neural network models
  - `inference.py` - Inference with small chunk processing
  - `data_utils.py` - Data loading utilities
  - `app.py` - Main application entry point

- **`scripts/`** - Training and evaluation scripts
  - `train_improved_model.py` - Main training script (uses small chunks)
  - `train_cnn_lstm.py` - CNN-LSTM training
  - `train_on_validation_data.py` - Validation data training
  - `eeg_api_server.py` - FastAPI server for inference
  - `load_models.py` - Model loading utilities

- **`models/`** - Trained model files (.pth, .pkl)

- **`docs/`** - Documentation files

- **`tests/`** - Unit tests

- **`notebooks/`** - Jupyter notebooks for experimentation

### Archive (Unused but Kept for Reference)

- **`archive/old_scripts/`** - Old or deprecated scripts
- **`archive/old_models/`** - Old model definitions
- **`archive/old_notebooks/`** - Old experimental notebooks

### Root Level Files

- **Active files:**
  - `cnn_lstm_model.py` - CNN-LSTM model definition (used by scripts)
  - `improved_cnn_lstm_model.py` - Improved CNN-LSTM model (used by scripts)
  - `README.md` - Main documentation
  - `Makefile` - Build and run commands
  - `pyproject.toml` - Python project configuration

- **Utility files:**
  - `validate_setup.py` - Setup validation
  - `repo_tree.py` - Repository tree generator

## Key Features

### Small Chunk Processing (0.1s to 0.5s)
- Training and inference now use small chunks for faster, more responsive processing
- Configurable chunk size via command-line arguments
- Default: 0.5 seconds

### Smoothing/Averaging
- Moving average smoothing applied to reduce noise
- Configurable smoothing window size
- Applied before neural network processing

## Usage

See `README.md` for detailed usage instructions.

## Migration Notes

If you have old scripts or models that are no longer used, consider moving them to the `archive/` directory to keep the repository clean while preserving them for reference.
