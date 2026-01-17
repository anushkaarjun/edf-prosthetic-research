# Quick Start Guide - Get Running Now!

## Option 1: Run Without Model (Simulation Mode)

You can test the simulator immediately without a trained model!

### Step 1: Start API Server

**Terminal 1:**
```bash
cd /Users/anushkaarjun/synopsys/edf-prosthetic-research
python3 eeg_api_server.py
```

Wait until you see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Keep this terminal open!**

### Step 2: Test the API

**Terminal 2 (new terminal):**
```bash
# Test health
curl http://localhost:8000/health

# Get simulated prediction (works without model!)
curl http://localhost:8000/simulate
```

### Step 3: Run React App

In your React project:
```bash
npm start
```

The simulator will work with **simulated data** - no model needed!

---

## Option 2: Train and Use CNN-LSTM Model

### Step 1: Train the Model

You need to train the CNN-LSTM model first. You have two options:

#### Option A: Use the Jupyter Notebook
```bash
# Open cnn-lstm.ipynb in Jupyter
jupyter notebook cnn-lstm.ipynb
```

Run all cells. The model will be saved as `best_model.pth` in the current directory.

#### Option B: Convert Notebook to Python Script

I can help you create a Python script version if you prefer.

### Step 2: Start API Server

**Terminal 1:**
```bash
cd /Users/anushkaarjun/synopsys/edf-prosthetic-research
python3 eeg_api_server.py
```

### Step 3: Load Your Trained Model

**Terminal 2:**
```bash
# Find where your model was saved
ls -la *.pth

# Load it (replace with actual path)
curl -X POST "http://localhost:8000/load_model" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type_param": "cnn_lstm",
    "model_path": "/Users/anushkaarjun/synopsys/edf-prosthetic-research/best_model.pth",
    "n_channels": 64
  }'
```

### Step 4: Verify Model Loaded

```bash
curl http://localhost:8000/health
```

Should show `"model_loaded": true` and `"model_type": "cnn_lstm"`

### Step 5: Run React App

```bash
npm start
```

---

## Troubleshooting

### "Couldn't connect to server"
**Solution:** Make sure API server is running in Terminal 1. Check with:
```bash
curl http://localhost:8000/health
```

### "No such file or directory: best_model.pth"
**Solution:** 
- Train the model first (see Option 2, Step 1)
- Or use simulation mode (Option 1) - no model needed!

### "Port 8000 already in use"
**Solution:**
```bash
# Find and kill process on port 8000
lsof -ti:8000 | xargs kill
```

---

## Recommended: Start with Simulation Mode

**Easiest way to get started:**

1. **Terminal 1:** Start API server
   ```bash
   cd /Users/anushkaarjun/synopsys/edf-prosthetic-research
   python3 eeg_api_server.py
   ```

2. **Terminal 2:** Test it works
   ```bash
   curl http://localhost:8000/simulate
   ```

3. **React App:** Run your simulator
   ```bash
   npm start
   ```

The simulator will work with simulated predictions. You can train and load a real model later!

