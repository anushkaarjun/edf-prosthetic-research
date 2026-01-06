# Fix: Update App.js to Show Simulator

## The Problem

Your React app is showing the default "Edit src/App.js" message because `App.js` hasn't been updated yet.

## The Solution

You need to update `src/App.js` in your React project.

---

## Step 1: Find Your React Project

Your React project should be in a folder called `eeg-simulator` (or wherever you created it).

Navigate to it:
```bash
cd eeg-simulator
# or wherever your React project is
```

---

## Step 2: Open App.js

Open the file `src/App.js` in any text editor:
- VS Code
- TextEdit (Mac)
- Notepad (Windows)
- Or use command line: `nano src/App.js` or `code src/App.js`

---

## Step 3: Replace Everything in App.js

**Delete all the current content** and replace it with:

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

---

## Step 4: Save the File

- **Mac**: Cmd + S
- **Windows/Linux**: Ctrl + S
- Or File → Save

---

## Step 5: Check React Auto-Reloads

React should **automatically reload** when you save. You should see:
- The page refreshes
- The simulator UI appears instead of "Edit src/App.js"

---

## If It Doesn't Work

### Check 1: Make sure EEGSimulator.jsx exists
```bash
ls src/EEGSimulator.jsx
```

If it doesn't exist, copy it:
```bash
cp /Users/anushkaarjun/synopsys/edf-prosthetic-research/eeg_simulator_react.jsx src/EEGSimulator.jsx
```

### Check 2: Make sure dependencies are installed
```bash
npm install recharts lucide-react
```

### Check 3: Check browser console for errors
- Open browser developer tools (F12 or Cmd+Option+I)
- Look at the Console tab for error messages

### Check 4: Restart React
```bash
# Stop React (Ctrl+C in the terminal running npm start)
# Then start again:
npm start
```

---

## Quick Fix Command

If you're in your React project directory:

```bash
# Make sure component exists
cp /Users/anushkaarjun/synopsys/edf-prosthetic-research/eeg_simulator_react.jsx src/EEGSimulator.jsx

# Create/update App.js
cat > src/App.js << 'EOF'
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
EOF
```

Then React should automatically reload!

---

## What You Should See After Fixing

✅ **Instead of "Edit src/App.js":**
- EEG Signals graph (left)
- Motor Function Probability bars (center)
- Confidence Radar chart (right)
- Status indicator (top right)

---

## Summary

1. Open `src/App.js`
2. Replace all content with the code above
3. Save the file
4. React auto-reloads
5. Simulator appears!

