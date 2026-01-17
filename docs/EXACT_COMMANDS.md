# Exact Commands to Run the Simulator

## Terminal 1: Start API Server

```bash
cd /Users/anushkaarjun/synopsys/edf-prosthetic-research
lsof -ti:8000 | xargs kill -9 2>/dev/null
python3 eeg_api_server.py
```

**Wait for:** `INFO:     Uvicorn running on http://0.0.0.0:8000`

**Keep this terminal open!**

---

## Terminal 2: Test API (Optional)

```bash
curl http://localhost:8000/health
```

**Expected:** JSON response with `"status": "ok"`

---

## Terminal 3: Start React App (OPTIONAL)

**You can skip React!** Just use the API web interface at `http://localhost:8000/docs`

### Option A: Use API Web Interface (EASIEST!)

After starting the API server (Terminal 1), open your browser to:
```
http://localhost:8000/docs
```

### Option B: Use React (For Visual Interface)

**Step 1: Create React project**
```bash
npx create-react-app eeg-simulator
cd eeg-simulator
```

**Step 2: Install dependencies**
```bash
npm install recharts lucide-react
```

**Step 3: Copy component**
```bash
cp /Users/anushkaarjun/synopsys/edf-prosthetic-research/eeg_simulator_react.jsx src/EEGSimulator.jsx
```

**Step 4: Update src/App.js** (replace all content with):
```javascript
import React from "react";
import EEGSimulator from "./EEGSimulator";

function App() {
  return <EEGSimulator />;
}

export default App;
```

**Step 5: Start React**
```bash
npm start
```

**See `SETUP_REACT_SIMULATOR.md` for detailed instructions!**

---

## That's It!

The simulator should now be running in your browser at `http://localhost:3000`

---

## Optional: Load CNN-LSTM Model

If you have a trained model:

```bash
curl -X POST "http://localhost:8000/load_model" \
  -H "Content-Type: application/json" \
  -d '{"model_type_param": "cnn_lstm", "model_path": "/Users/anushkaarjun/synopsys/edf-prosthetic-research/best_model.pth", "n_channels": 64}'
```

---

## All Commands in One Place

**Terminal 1:**
```bash
cd /Users/anushkaarjun/synopsys/edf-prosthetic-research && lsof -ti:8000 | xargs kill -9 2>/dev/null && python3 eeg_api_server.py
```

**Terminal 2 (Test):**
```bash
curl http://localhost:8000/health
```

**Terminal 3 (React):**
```bash
cd /path/to/react/project && npm start
```

