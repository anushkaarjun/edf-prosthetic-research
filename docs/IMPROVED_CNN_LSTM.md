# Improved CNN-LSTM Model

## ✅ Improvements Made to CNN-LSTM

The CNN-LSTM model has been improved with the same techniques used for EEGNet:

### 📊 Architecture Improvements

| Component | Original CNN-LSTM | Improved CNN-LSTM | Improvement |
|-----------|------------------|-------------------|-------------|
| **CNN Depth** | 2 conv layers | 4 conv blocks | 2x deeper |
| **CNN Width** | 32→64 channels | 64→128→128→256 channels | 4x wider |
| **LSTM Depth** | 1 layer, 128 units | 2 layers, 256 units | 2x deeper, 2x wider |
| **Classifier** | 1 FC layer | 3 FC layers (256→512→256→3) | 3x deeper, wider |
| **Parameters** | ~1M | ~3.5M | 3.5x more |

### 🎯 Specific Changes

#### 1. Deeper CNN (4 blocks vs 2)
- **Block 1**: Conv2d(1→64) with BatchNorm
- **Block 2**: Conv2d(64→128) spatial + MaxPool
- **Block 3**: Conv2d(128→128) temporal (deeper)
- **Block 4**: Conv2d(128→256) + MaxPool + AdaptivePool (widest)

#### 2. Wider Channels
- **Original**: 32→64 channels
- **Improved**: 64→128→128→256 channels (4x wider at max)

#### 3. Deeper LSTM
- **Original**: 1 layer, 128 hidden units
- **Improved**: 2 layers, 256 hidden units
- **Benefit**: Better temporal pattern learning

#### 4. Deeper Classifier
- **Original**: Linear(128→3)
- **Improved**: 
  - Linear(256→512) with BatchNorm
  - Linear(512→256) with BatchNorm
  - Linear(256→3)
- **Benefit**: More complex decision boundaries

#### 5. Better Regularization
- **Progressive Dropout**: 0.5 → 0.35 → 0.25 → 0.15 → 0.1
- **LSTM Dropout**: 0.2
- **Batch Normalization**: After each conv and FC layer
- **Gradient Clipping**: max_norm=1.0

#### 6. Better Training
- **Weight Decay**: 1e-4
- **Learning Rate Scheduler**: ReduceLROnPlateau (patience=5)
- **Early Stopping**: Patience=15
- **Class Weights**: Handles imbalanced data

## 🚀 Training the Improved Model

```bash
# Train improved CNN-LSTM
make train-improved-cnn-lstm

# Or manually:
python3 scripts/train_improved_cnn_lstm.py \
    --data-path "/path/to/data" \
    --max-subjects 5 \
    --epochs 50
```

## 📈 Expected Results

Based on improvements:
- **Current CNN-LSTM**: 51.94% accuracy
- **Expected Improved**: 55-65% accuracy
- **Expected Improvement**: +3-13% over baseline

The improved model should outperform the original CNN-LSTM due to:
1. More parameters (3.5M vs 1M)
2. Deeper architecture (better feature learning)
3. Wider channels (richer representations)
4. Deeper LSTM (better temporal patterns)
5. Better regularization (less overfitting)

## 📝 Model File

Trained model will be saved to:
- `models/best_improved_cnn_lstm.pth`

## 🔧 Loading into API

Once trained, load with:
```bash
python3 scripts/load_improved_cnn_lstm.py
```

Or update the API server to support `improved_cnn_lstm` model type.

---

*The improved CNN-LSTM is ready to train with increased capacity!*
