# Setup React Simulator - Step by Step

## Prerequisites

1. **Node.js installed** - Download from https://nodejs.org/
2. **API server running** (from previous steps)

---

## Step 1: Create a React Project

Open Terminal and run:

```bash
npx create-react-app eeg-simulator
cd eeg-simulator
```

This creates a new React project called `eeg-simulator`.

**Wait for it to finish** (takes 1-2 minutes)

---

## Step 2: Install Required Dependencies

```bash
npm install recharts lucide-react
```

This installs the chart libraries needed for the simulator.

---

## Step 3: Copy the Simulator Component

```bash
cp /Users/anushkaarjun/synopsys/edf-prosthetic-research/eeg_simulator_react.jsx src/EEGSimulator.jsx
```

This copies the EEG simulator code into your React project.

---

## Step 4: Update App.js

Open `src/App.js` in a text editor and **replace everything** with:

```javascript
import React from "react";
import EEGSimulator from "./EEGSimulator";

function App() {
  return (
    <div className="App">
      <EEGSimulator />
    </div>
  );
}

export default App;
```

**Save the file** (Cmd+S or Ctrl+S)

---

## Step 5: Make Sure API Server is Running

**In a separate terminal**, start the API server:

```bash
cd /Users/anushkaarjun/synopsys/edf-prosthetic-research
python3 eeg_api_server.py
```

**Keep this terminal open!**

---

## Step 6: Test the API (Optional)

In another terminal:

```bash
curl http://localhost:8000/health
```

You should see a JSON response with `"status": "ok"`

---

## Step 7: Start the React App

**Back in your React project terminal:**

```bash
npm start
```

**Expected output:**
```
Compiled successfully!

You can now view eeg-simulator in the browser.

  Local:            http://localhost:3000
  On Your Network:  http://192.168.x.x:3000
```

The browser should **automatically open** to `http://localhost:3000`

---

## Step 8: Verify It's Working

You should see:

✅ **EEG Signals** graph (left side)
✅ **Motor Function Probability** bars (center)
✅ **Confidence Radar** chart (right side)
✅ **Status indicator** (top right) showing API connection

---

## Troubleshooting

### Problem: "Module not found: recharts"
**Solution:**
```bash
npm install recharts lucide-react
```

### Problem: React app shows "API Disconnected"
**Solution:**
- Make sure API server is running (Step 5)
- Check API server terminal shows "Uvicorn running"
- Test: `curl http://localhost:8000/health`

### Problem: Browser doesn't open automatically
**Solution:**
- Manually open: `http://localhost:3000`

### Problem: "Cannot find module './EEGSimulator'"
**Solution:**
- Make sure you copied the file correctly (Step 3)
- Check file exists: `ls src/EEGSimulator.jsx`

---

## All Commands in Order

```bash
# 1. Create React project
npx create-react-app eeg-simulator
cd eeg-simulator

# 2. Install dependencies
npm install recharts lucide-react

# 3. Copy simulator component
cp /Users/anushkaarjun/synopsys/edf-prosthetic-research/eeg_simulator_react.jsx src/EEGSimulator.jsx

# 4. Edit src/App.js (use text editor)
# Replace with the code from Step 4 above

# 5. Start API server (separate terminal)
cd /Users/anushkaarjun/synopsys/edf-prosthetic-research
python3 eeg_api_server.py

# 6. Start React app
npm start
```

---

## What You Should See

### ✅ Working Correctly:
- Browser opens to `http://localhost:3000`
- Three-panel layout with graphs
- Real-time updating predictions
- Status indicator shows "API Connected"
- Probability bars changing

### ❌ Not Working:
- "API Disconnected" message
- No graphs showing
- Error messages in browser console

---

## Quick Reference

**Terminal 1 (API Server):**
```bash
cd /Users/anushkaarjun/synopsys/edf-prosthetic-research
python3 eeg_api_server.py
```

**Terminal 2 (React App):**
```bash
cd eeg-simulator
npm start
```

**Browser:**
- Opens automatically to `http://localhost:3000`
- Or manually navigate to that URL

---

## Summary

1. ✅ Create React project: `npx create-react-app eeg-simulator`
2. ✅ Install dependencies: `npm install recharts lucide-react`
3. ✅ Copy component: `cp .../eeg_simulator_react.jsx src/EEGSimulator.jsx`
4. ✅ Update App.js with the code above
5. ✅ Start API server: `python3 eeg_api_server.py`
6. ✅ Start React: `npm start`
7. ✅ Browser opens automatically!

**That's it!** The simulator should now be running in your browser.

