# Model Accuracy Summary

This document summarizes the validation and test accuracies for all trained models.

## Training Configuration
- **Dataset**: 5 subjects (S001-S005)
- **Total Epochs**: 1740 (4-class models), 1800 (CNN-LSTM)
- **Train/Validation Split**: 80/20
- **Data Format**: 0.5-second epochs at 250Hz (EEGNet/CSP+SVM), 2-second epochs at 160Hz (CNN-LSTM)

---

## 1. CSP+SVM Model

### Results
- **Train Accuracy**: **50.93%**
- **Validation Accuracy**: **44.83%**

### Class-wise Performance (Validation Set, n=348)
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Both Feet | 0.42 | 0.43 | 0.43 | 90 |
| Both Fists | 0.50 | 0.43 | 0.46 | 84 |
| Left Hand | 0.37 | 0.38 | 0.37 | 84 |
| Right Hand | 0.51 | 0.54 | 0.53 | 90 |
| **Macro Avg** | **0.45** | **0.45** | **0.45** | **348** |
| **Weighted Avg** | **0.45** | **0.45** | **0.45** | **348** |

### Model File
- `csp_svm_model.pkl` (187KB)

---

## 2. EEGNet Model

### Results
- **Validation Accuracy**: **41.95%** (best epoch: 25, early stopping at 26)
- **Final Validation Accuracy**: **43.10%** (from best saved model)

### Training Details
- Early stopping at epoch 26 (patience: 10)
- Best model saved at epoch 25
- Normalized data to [-1, 1] range

### Model File
- `eegnet_trained.pth` (285KB)

---

## 3. CNN-LSTM Model

### Results
- **Validation Accuracy**: **51.94%**
- **Best epoch**: 20 (early stopping at 23)

### Class-wise Performance (Validation Set, n=360)
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Open Left Fist | 0.44 | 0.34 | 0.39 | 90 |
| Open Right Fist | 0.46 | 0.37 | 0.41 | 90 |
| Close Fists | 0.56 | 0.68 | 0.62 | 180 |
| **Macro Avg** | **0.49** | **0.46** | **0.47** | **360** |
| **Weighted Avg** | **0.51** | **0.52** | **0.51** | **360** |

### Training Details
- Training epochs: 23 (early stopping)
- Best model saved at epoch 20
- Uses 2-second windows at 160Hz (320 samples)
- 3-class classification: Open Left Fist, Open Right Fist, Close Fists

### Model File
- `best_model.pth` (2.9MB)

---

## Model Comparison

| Model | Validation Accuracy | Classes | Data Format | Model Size |
|-------|-------------------|---------|-------------|------------|
| **CNN-LSTM** | **51.94%** 🏆 | 3 classes | 2s @ 160Hz | 2.9MB |
| **CSP+SVM** | **44.83%** | 4 classes | 0.5s @ 250Hz | 187KB |
| **EEGNet** | **43.10%** | 4 classes | 0.5s @ 250Hz | 285KB |

---

## Key Observations

1. **Best Performing Model**: CNN-LSTM achieves the highest validation accuracy (51.94%)
   - Uses 2-second windows for better temporal context
   - 3-class classification may be easier than 4-class

2. **CSP+SVM**: Second best (44.83%)
   - Fastest to train and smallest model size
   - Good balance between accuracy and efficiency

3. **EEGNet**: Third place (43.10%)
   - More complex architecture than CSP+SVM
   - Requires more training time

4. **Class Balance**:
   - 4-class models: Balanced classes (~25% each)
   - CNN-LSTM: Imbalanced (50% "Close Fists", 25% each for open fists)
   - CNN-LSTM performs best on "Close Fists" class (68% recall)

---

## Training Notes

- All models were trained on the same 5 subjects (S001-S005)
- Data preprocessing varies by model:
  - **EEGNet/CSP+SVM**: 0.5-second epochs, 8-30Hz bandpass filter, normalized to [-1,1]
  - **CNN-LSTM**: 2-second epochs, resampled to 160Hz, StandardScaler normalization
- Early stopping was used for all neural network models to prevent overfitting

---

## Next Steps for Improvement

1. **Increase training data**: Use more than 5 subjects
2. **Hyperparameter tuning**: Adjust learning rates, batch sizes, architecture
3. **Data augmentation**: Apply more aggressive augmentation techniques
4. **Ensemble methods**: Combine predictions from multiple models
5. **Feature engineering**: Experiment with additional preprocessing steps
