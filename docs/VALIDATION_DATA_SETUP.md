# Using Real Validation Data in React Component

## Overview
The React component now supports using real validation data from your dataset instead of simulated data.

## Setup

### 1. Update Data Path in React Component
In `eeg_simulator_react.jsx`, update the data path to match your system:

```javascript
body: JSON.stringify({
  base_path: '/Users/anushkaarjun/Desktop/Outside of School/Prosethic Research Data/files 2',
  max_subjects: 5
})
```

### 2. Start API Server
```bash
cd /Users/anushkaarjun/synopsys/edf-prosthetic-research
python3 eeg_api_server.py
```

### 3. Load Validation Data
The React component will automatically try to load validation data when it connects to the API.

Or manually load it via API:
```bash
curl -X POST "http://localhost:8000/load_validation_data" \
  -H "Content-Type: application/json" \
  -d '{
    "base_path": "/Users/anushkaarjun/Desktop/Outside of School/Prosethic Research Data/files 2",
    "max_subjects": 5
  }'
```

### 4. Get Validation Samples
The React component automatically fetches validation samples via:
```bash
curl http://localhost:8000/get_validation_sample
```

## How It Works

1. **Data Loading**: The API server loads validation data from your dataset
2. **Sample Streaming**: Each call to `/get_validation_sample` returns the next validation sample
3. **Real Predictions**: If a model is loaded, predictions use the actual model
4. **Display**: The component shows:
   - Real EEG signals from validation data (125 samples per channel)
   - Actual labels from the dataset
   - Model predictions (if model is loaded)
   - Sample index (e.g., "Sample 42 / 348")

## Features

- ✅ Uses real validation data from your dataset
- ✅ Shows actual class labels from annotations
- ✅ Displays full 0.5-second time series (125 samples)
- ✅ Automatically cycles through validation samples
- ✅ Falls back to simulation if validation data unavailable
- ✅ Works with or without a trained model

## Status Indicators

- **Green**: API connected with model loaded
- **Yellow**: API connected but no model
- **Red**: API disconnected
- **Blue**: Using real validation data (shows sample count)

## Notes

- Validation data is split 80/20 from the full dataset
- Samples cycle through all validation data
- Update rate is 200ms (5 samples/second) for validation data
- The component handles missing data gracefully

