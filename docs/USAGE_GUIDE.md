# Quick Usage Guide

## 🚀 Quick Start

### 1. Setup Environment
```bash
make init
```

### 2. Train Models
```bash
# Train improved model (recommended - best accuracy)
make train-improved

# Or train all models
make train-all
```

### 3. Start API Server
```bash
make api-server
```

### 4. Load Models
```bash
make load-models
```

### 5. Start React Simulator
```bash
cd ../eeg-simulator-ui-2
npm start
```

## 📁 Project Structure

```
edf-prosthetic-research/
├── scripts/          # All training and API scripts
│   ├── train_improved_model.py
│   ├── train_on_validation_data.py
│   ├── train_cnn_lstm.py
│   ├── eeg_api_server.py
│   └── load_models.py
│
├── models/           # Trained model files
│   ├── best_model.pth (CNN-LSTM)
│   ├── csp_svm_model.pkl
│   └── eegnet_trained.pth
│
├── docs/             # All documentation
│   ├── MODEL_ACCURACIES.md
│   ├── IMPROVEMENTS_SUMMARY.md
│   └── ...
│
├── notebooks/        # Jupyter notebooks
│   ├── cnn-lstm.ipynb
│   └── ...
│
└── results/          # Training outputs
    └── *.txt
```

## 📝 Key Commands

### Training
- `make train-improved` - Train improved neural network (recommended)
- `make train-csp-svm` - Train CSP+SVM
- `make train-eegnet` - Train EEGNet
- `make train-cnn-lstm` - Train CNN-LSTM

### API Server
- `make api-server` - Start server
- `make api-server-kill` - Stop server
- `make load-models` - Load models into API
- `make api-health` - Check API status

### Code Quality
- `make format` - Format and lint code
- `make test` - Run tests
- `make clean` - Clean temporary files

## 📊 Model Accuracies

| Model | Accuracy | Status |
|-------|----------|--------|
| CNN-LSTM | 51.94% | ✅ Trained |
| CSP+SVM | 44.83% | ✅ Trained |
| EEGNet | 43.10% | ✅ Trained |
| ImprovedEEGNet | Training... | 🔄 In Progress |

## 🔧 Configuration

Update `DATA_PATH` in Makefile or provide `--data-path` when running scripts:

```bash
python3 scripts/train_improved_model.py \
    --data-path "/path/to/your/data" \
    --max-subjects 5 \
    --epochs 50
```

## 📖 Documentation

- [Model Accuracies](docs/MODEL_ACCURACIES.md)
- [Improvements Summary](docs/IMPROVEMENTS_SUMMARY.md)
- [API Documentation](docs/EEG_API_README.md)
- [CNN-LSTM Pipeline](docs/CNN_LSTM_PIPELINE_EXPLANATION.md)
