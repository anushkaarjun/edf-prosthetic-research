# How to Run CNN-LSTM Simulator

## Prerequisites

1. **Trained CNN-LSTM Model**
   - You need a trained model saved as `.pth` file
   - If you haven't trained one yet, use `cnn-lstm.ipynb` to train it
   - The model should be saved as `best_model.pth` (or any name you prefer)

2. **Dependencies Installed**
   ```bash
   pip3 install fastapi uvicorn pydantic torch numpy scikit-learn mne
   ```

## Step-by-Step Instructions

### Step 1: Start the API Server

Open Terminal 1:
```bash
cd /Users/anushkaarjun/synopsys/edf-prosthetic-research
python3 eeg_api_server.py
```

You should see:
```
Starting EEG Motor Imagery API server...
API will be available at http://localhost:8000
API docs at http://localhost:8000/docs
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Keep this terminal open!**

### Step 2: Load the CNN-LSTM Model

Open Terminal 2 (new terminal):
```bash
# Replace /path/to/best_model.pth with your actual model path
curl -X POST "http://localhost:8000/load_model" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type_param": "cnn_lstm",
    "model_path": "/Users/anushkaarjun/synopsys/edf-prosthetic-research/best_model.pth",
    "n_channels": 64
  }'
```

**Expected Response:**
```json
{
  "status": "Model loaded successfully",
  "model_type": "cnn_lstm"
}
```

### Step 3: Verify Model is Loaded

```bash
curl http://localhost:8000/health
```

**Expected Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_type": "cnn_lstm",
  "classes": ["Open Left Fist", "Open Right Fist", "Close Fists"]
}
```

### Step 4: Test a Prediction

```bash
curl http://localhost:8000/simulate
```

This will return a prediction with probabilities for the 3 classes.

### Step 5: Run Your React Component

In your React project directory:
```bash
npm start
# or
yarn start
```

The React component will:
- Automatically connect to `http://localhost:8000`
- Detect the CNN-LSTM model
- Show 3 classes: "Open Left Fist", "Open Right Fist", "Close Fists"
- Display real-time predictions

## Quick Test (All in One)

If you want to test everything quickly:

**Terminal 1:**
```bash
cd /Users/anushkaarjun/synopsys/edf-prosthetic-research
python3 eeg_api_server.py
```

**Terminal 2:**
```bash
# Wait a few seconds for server to start, then:
curl -X POST "http://localhost:8000/load_model" \
  -H "Content-Type: application/json" \
  -d '{"model_type_param": "cnn_lstm", "model_path": "/path/to/best_model.pth", "n_channels": 64}'

# Test it
curl http://localhost:8000/simulate
```

## Troubleshooting

### Model Not Found
```
Error: [Errno 2] No such file or directory: 'best_model.pth'
```
**Solution:** Make sure the model path is correct. Use absolute path.

### Port Already in Use
```
ERROR: [Errno 48] Address already in use
```
**Solution:** 
- Kill the existing process: `lsof -ti:8000 | xargs kill`
- Or change port in `eeg_api_server.py`

### Model Loading Fails
**Solution:** 
- Check that model was trained with same architecture
- Verify `n_channels` matches your data (usually 64)
- Make sure model file is a PyTorch `.pth` file

### React Component Not Connecting
**Solution:**
- Check API server is running: `curl http://localhost:8000/health`
- Check CORS is enabled (it should be by default)
- Verify API_URL in React component matches server URL

## Alternative: Using Python Interactive Shell

You can also test the API directly:

```python
import requests

# Load model
response = requests.post(
    "http://localhost:8000/load_model",
    json={
        "model_type_param": "cnn_lstm",
        "model_path": "/path/to/best_model.pth",
        "n_channels": 64
    }
)
print(response.json())

# Test prediction
response = requests.get("http://localhost:8000/simulate")
print(response.json())
```

## What You Should See

1. **API Server Terminal:**
   - Server running messages
   - Request logs when you make API calls

2. **React Component:**
   - 3 classes displayed (not 5)
   - Real-time predictions updating
   - Status indicator showing "API Connected (Using Real Model)"
   - Model type showing "cnn_lstm"

## Next Steps

- Load validation data for real samples: See `VALIDATION_DATA_SETUP.md`
- Try different models: Switch between CNN-LSTM, EEGNet, and CSP+SVM
- Customize the UI: Modify colors and labels in React component

