# EDF Prosthetic Research - Motor Imagery Classification

End-to-end pipeline for training deep learning models to classify motor imagery events from EEG signals.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended) or pip

### Installation

```bash
# Clone the repository
git clone git@github.com:anushkaarjun/edf-prosthetic-research.git
cd edf-prosthetic-research

# Initialize environment
make init
```

## 📊 Models Available

| Model | Validation Accuracy | Classes | Best Use Case |
|-------|-------------------|---------|---------------|
| **CNN-LSTM** | **51.94%** 🏆 | 3 classes | Temporal patterns |
| **CSP+SVM** | 44.83% | 4 classes | Fast, interpretable |
| **EEGNet** | 43.10% | 4 classes | Deep learning baseline |
| **ImprovedEEGNet** | Training... | 4 classes | Maximum accuracy potential |

See [MODEL_ACCURACIES.md](docs/MODEL_ACCURACIES.md) for detailed results.

## 🎯 Training Models

### Train All Models

```bash
# Train all models at once
make train-all

# Or train individually:
make train-csp-svm    # CSP+SVM model
make train-eegnet     # EEGNet model
make train-cnn-lstm   # CNN-LSTM model
make train-improved   # Improved neural network (recommended)
```

### Training Options

```bash
# Train with custom settings
python3 train_improved_model.py \
    --data-path "/path/to/your/data" \
    --max-subjects 5 \
    --epochs 50 \
    --test-all  # Test all model architectures
```

**Note**: Update `DATA_PATH` in Makefile or provide `--data-path` when running scripts.

## 🌐 API Server

### Start API Server

```bash
# Start the FastAPI server
make api-server

# Or manually:
python3 eeg_api_server.py
```

The API will be available at `http://localhost:8000`

### Load Models into API

```bash
# Load all models
make load-models

# Or load individually:
make load-csp-svm    # Load CSP+SVM model
make load-eegnet     # Load EEGNet model
make load-cnn-lstm   # Load CNN-LSTM model
```

### Check API Status

```bash
make api-health
```

### API Endpoints

- `GET /health` - Check API status and loaded models
- `POST /predict` - Get predictions from EEG data
- `POST /load_model` - Load a trained model
- `POST /load_validation_data` - Load validation dataset
- `GET /get_validation_sample` - Get next validation sample
- `GET /docs` - Interactive API documentation

See [docs/EEG_API_README.md](docs/EEG_API_README.md) for detailed API documentation.

## 🖥️ React Simulator

### Setup

1. Navigate to React project:
```bash
cd ../eeg-simulator-ui-2
npm install
```

2. Start React app:
```bash
npm start
```

3. Make sure API server is running (see above)

The simulator will automatically connect to `http://localhost:8000`

## 📁 Project Structure

```
edf-prosthetic-research/
├── src/
│   └── edf_ml_model/          # Core ML models and utilities
│       ├── model.py           # Original EEGNet model
│       ├── improved_model.py  # Improved models (depth/width)
│       ├── preprocessing.py   # Data preprocessing functions
│       └── data_utils.py      # Data loading utilities
│
├── scripts/                    # Training and evaluation scripts
│   ├── train_on_validation_data.py  # Train CSP+SVM and EEGNet
│   ├── train_cnn_lstm.py      # Train CNN-LSTM model
│   ├── train_improved_model.py # Train improved models
│   ├── eeg_api_server.py      # FastAPI server
│   └── load_models.py         # Load models into API
│
├── models/                     # Trained model files
│   ├── csp_svm_model.pkl
│   ├── eegnet_trained.pth
│   └── best_model.pth (CNN-LSTM)
│
├── docs/                       # Documentation
│   ├── MODEL_ACCURACIES.md
│   ├── CNN_LSTM_PIPELINE_EXPLANATION.md
│   └── ...
│
├── notebooks/                  # Jupyter notebooks
│   ├── cnn-lstm.ipynb
│   └── ...
│
└── results/                    # Training results and outputs
    └── *.txt
```

## 🔧 Makefile Commands

### Environment Setup
```bash
make init          # Initialize environment
make update        # Update dependencies
make clean         # Clean temporary files
```

### Code Quality
```bash
make format        # Format and lint code
make test          # Run tests
make lint          # Lint code
make typecheck     # Type checking
```

### Model Training
```bash
make train-csp-svm      # Train CSP+SVM
make train-eegnet       # Train EEGNet
make train-cnn-lstm     # Train CNN-LSTM
make train-improved     # Train improved model
make train-all          # Train all models
```

### API Server
```bash
make api-server    # Start API server
make api-server-kill  # Stop API server
make load-models   # Load all models into API
make api-health    # Check API status
```

## 📚 Documentation

- [Model Accuracies](docs/MODEL_ACCURACIES.md) - Detailed accuracy results
- [CNN-LSTM Pipeline Explanation](docs/CNN_LSTM_PIPELINE_EXPLANATION.md) - Deep dive into CNN-LSTM
- [API Documentation](docs/EEG_API_README.md) - API usage guide
- [Quick Start Guide](docs/QUICK_START_GUIDE.md) - Step-by-step setup

## 🎓 Model Architectures

### ImprovedEEGNet (Recommended)
- **Depth**: 3 convolutional blocks + 4 fully connected layers
- **Width**: 64→128→256 channels, 512→256→128→n_classes neurons
- **Features**: Batch normalization, dropout, adaptive pooling
- **Best for**: Maximum accuracy potential

### SimpleEEGNet
- **Architecture**: Fully connected layers (similar to SimpleNet)
- **Depth**: 5 layers
- **Width**: 512→256→128→64→n_classes neurons
- **Best for**: Quick experimentation

### DeepEEGNet
- **Depth**: Very deep with residual connections
- **Width**: Up to 512 channels, 1024 neurons
- **Features**: Residual blocks, maximum capacity
- **Best for**: Maximum model capacity

## 🔬 Data Format

- **Input**: EDF (European Data Format) files with EEG signals
- **Sampling Rate**: 250Hz (EEGNet/CSP+SVM) or 160Hz (CNN-LSTM)
- **Window Size**: 0.5 seconds (125 samples) or 2 seconds (320 samples)
- **Channels**: 64 EEG channels
- **Classes**: 
  - 4-class: Both Feet, Both Fists, Left Hand, Right Hand
  - 3-class: Open Left Fist, Open Right Fist, Close Fists

## 📈 Improving Accuracy

See [docs/IMPROVE_ACCURACY.md](docs/IMPROVE_ACCURACY.md) for strategies to improve model accuracy:

1. Expand hyperparameter search space
2. Increase CSP components
3. Use more training data
4. Add data augmentation
5. Try different algorithms (LDA, Random Forest)
6. Ensemble methods

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## 📄 License

See [LICENSE](LICENSE) for details.

## 📞 Support

For issues or questions, please open an issue on GitHub.
