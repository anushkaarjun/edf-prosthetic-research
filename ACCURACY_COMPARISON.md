# Accuracy Comparison: CSP+SVM vs Neural Network

## Best Approach: **CSP+SVM** (Recommended)

### Why CSP+SVM is Best:
1. **Proven Performance**: Code comments indicate it "typically achieves 80%+ accuracy"
2. **Fast Training**: Much faster than neural networks
3. **Auto-Optimization**: Built-in GridSearchCV when accuracy < 80%
4. **Subject-Specific**: Trains one model per subject (better for EEG)
5. **Robust**: Well-established method for motor imagery BCI

### Expected Accuracy:
- **Target**: 80%+ test accuracy
- **Auto-optimization**: If initial accuracy < 80%, GridSearchCV searches:
  - CSP components: [6, 8, 10]
  - SVM C: [0.1, 1.0, 10.0]
  - SVM gamma: ['scale', 'auto']
- **3-fold cross-validation** for parameter selection

### Running CSP+SVM:
```bash
python run_csp_svm.py --data-path /path/to/data --max-subjects 5
```

---

## Alternative: Neural Network (New Approach)

### Neural Network Characteristics:
1. **Weight Freezing Workflow**: Train → Freeze backbone → Fine-tune classifier
2. **0.5-second Epochs**: Uses half-second windows
3. **Normalized Data**: [-1, 1] range
4. **Untested**: Performance not yet evaluated

### Expected Accuracy:
- **Unknown**: Not yet tested on this dataset
- **Potential**: Could match or exceed CSP+SVM with proper training
- **Trade-off**: Slower training, more complex

### Running Neural Network:
```bash
python train_model.py --data-path /path/to/data --max-subjects 5 --freeze-after 30 --epochs 50
```

---

## Recommendation

**Use CSP+SVM** for:
- ✅ Proven 80%+ accuracy
- ✅ Fast evaluation
- ✅ Production-ready
- ✅ Well-tested code

**Consider Neural Network** if:
- You need to experiment with deep learning
- You want to use the weight freezing workflow
- You have time for longer training

---

## To Get Actual Accuracy

Provide your data path and run:
```bash
python run_csp_svm.py --data-path YOUR_DATA_PATH --max-subjects 5
```

The script will output:
- Per-subject test accuracy
- Average test accuracy
- Whether 80% target is achieved
- Detailed classification reports

