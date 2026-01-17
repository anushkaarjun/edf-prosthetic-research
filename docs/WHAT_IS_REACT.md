# What is React and How to Use the Simulator

## What is React?

**React** is a JavaScript library for building web interfaces (the visual part of websites).

Think of it like:
- **HTML** = Structure of a webpage
- **CSS** = Styling/colors
- **React** = Makes the page interactive and dynamic

## Do You Need React?

**You have two options:**

### Option 1: Use React (Recommended for Web Interface)

If you want a **visual web interface** in your browser showing:
- Real-time graphs
- Probability bars
- Interactive charts

Then you need React.

### Option 2: Use API Only (No React Needed)

If you just want to **test the model** or **get predictions**, you can use the API directly without React!

---

## Option 1: Setting Up React

### Step 1: Create a React Project

If you don't have a React project yet, create one:

```bash
# Install Node.js first (if not installed)
# Download from: https://nodejs.org/

# Create a new React project
npx create-react-app eeg-simulator
cd eeg-simulator
```

### Step 2: Copy the Component

Copy the React component file into your project:

```bash
# Copy the component file
cp /Users/anushkaarjun/synopsys/edf-prosthetic-research/eeg_simulator_react.jsx \
   /path/to/eeg-simulator/src/EEGSimulator.jsx
```

### Step 3: Install Dependencies

```bash
cd eeg-simulator
npm install recharts lucide-react
```

### Step 4: Use the Component

Edit `src/App.js`:

```javascript
import EEGSimulator from './EEGSimulator';

function App() {
  return <EEGSimulator />;
}

export default App;
```

### Step 5: Run It

```bash
npm start
```

---

## Option 2: Use API Only (No React - Easier!)

**You don't need React!** You can test everything using just the API.

### Step 1: Start API Server

```bash
cd /Users/anushkaarjun/synopsys/edf-prosthetic-research
python3 eeg_api_server.py
```

### Step 2: Test in Browser

Open your browser and go to:
```
http://localhost:8000/docs
```

This shows an **interactive API interface** where you can:
- Test predictions
- Load models
- See all available endpoints

### Step 3: Test Predictions

In a new terminal:
```bash
# Get a prediction
curl http://localhost:8000/simulate

# Or use the web interface at http://localhost:8000/docs
```

---

## Which Option Should You Choose?

### Choose Option 1 (React) if:
- ✅ You want a beautiful visual interface
- ✅ You want real-time graphs and charts
- ✅ You're comfortable with web development
- ✅ You want to show this to others visually

### Choose Option 2 (API Only) if:
- ✅ You just want to test the model
- ✅ You don't need a fancy interface
- ✅ You want something quick and simple
- ✅ You're not familiar with React/web development

---

## Quick Start: API Only (Easiest!)

**Just run these commands:**

```bash
# Terminal 1: Start API
cd /Users/anushkaarjun/synopsys/edf-prosthetic-research
python3 eeg_api_server.py

# Terminal 2: Test it
curl http://localhost:8000/simulate
```

Then open your browser to: **http://localhost:8000/docs**

You'll see a web interface to test everything!

---

## Summary

- **React** = JavaScript library for web interfaces
- **You don't need React** to use the API
- **Easiest way**: Use the API directly at `http://localhost:8000/docs`
- **React way**: Set up a React project if you want a custom visual interface

**Recommendation:** Start with Option 2 (API only) to test everything, then set up React later if you want a custom interface!

