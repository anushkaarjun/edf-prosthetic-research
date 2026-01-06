# Commands to Run EEG API Server and React Component

## 1. Install Dependencies

```bash
# Install FastAPI and related packages
pip install fastapi uvicorn pydantic

# Or if using the project's dependency management
pip install -e .
```

## 2. Run the API Server

### Basic Run (Development)
```bash
cd /Users/anushkaarjun/synopsys/edf-prosthetic-research
python eeg_api_server.py
```

### Using Uvicorn Directly
```bash
cd /Users/anushkaarjun/synopsys/edf-prosthetic-research
uvicorn eeg_api_server:app --host 0.0.0.0 --port 8000 --reload
```

The server will start at:
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs (Interactive Swagger UI)
- **Health Check**: http://localhost:8000/health

## 3. Test the API

### Health Check
```bash
curl http://localhost:8000/health
```

### Simulate Prediction (no model needed)
```bash
curl http://localhost:8000/simulate
```

### Load a Model (if you have trained models)
```bash
# For EEGNet
curl -X POST "http://localhost:8000/load_model" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type_param": "eegnet",
    "model_path": "/path/to/eegnet_model.pth"
  }'

# For CSP+SVM
curl -X POST "http://localhost:8000/load_model" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type_param": "csp_svm",
    "model_path": "/path/to/svm_model.pkl",
    "csp_path": "/path/to/csp_model.pkl"
  }'
```

## 4. Run React Component

### Option A: If using Create React App
```bash
# In your React project directory
npm start
# or
yarn start
```

### Option B: If using Vite
```bash
npm run dev
# or
yarn dev
```

### Option C: If using Next.js
```bash
npm run dev
# or
yarn dev
```

The React component will automatically:
- Try to connect to `http://localhost:8000`
- Show connection status in the top-right corner
- Use real predictions if API/model is available
- Fall back to simulation if API is unavailable

## 5. Quick Start (All-in-One)

### Terminal 1: Start API Server
```bash
cd /Users/anushkaarjun/synopsys/edf-prosthetic-research
python eeg_api_server.py
```

### Terminal 2: Start React App
```bash
cd /path/to/your/react/project
npm start
```

## Troubleshooting

### API Server Issues
- **Port already in use**: Change port in `eeg_api_server.py` or use `--port 8001`
- **Module not found**: Install dependencies with `pip install fastapi uvicorn pydantic`
- **CORS errors**: The server already has CORS enabled, but check your React app's API_URL

### React Component Issues
- **API not connecting**: Check that API server is running on port 8000
- **CORS errors**: Make sure API server is running and CORS is enabled
- **No predictions**: The component will use simulation if no model is loaded

## Example: Full Workflow

```bash
# 1. Start API server
cd /Users/anushkaarjun/synopsys/edf-prosthetic-research
python eeg_api_server.py &

# 2. Check it's running
curl http://localhost:8000/health

# 3. Test simulation endpoint
curl http://localhost:8000/simulate

# 4. Start React app (in another terminal)
cd /path/to/react/app
npm start
```

## Notes

- The API server runs on port 8000 by default
- The React component expects the API at `http://localhost:8000`
- You can modify the API_URL in the React component if needed
- The API will work with simulated data even without a trained model loaded

