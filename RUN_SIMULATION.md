# Run the EEG Simulation

The **frontend** (React on port 3000) talks to the **backend** (Flask API on port 5001). They are already connected:

- Frontend uses `API_BASE = http://localhost:5001` (see `src/constants.js`).
- Backend enables CORS so the browser can call the API from the React app.
- To use a different API URL, create a `.env` file with `VITE_API_URL=http://your-backend-url:5001`.

---

## How to run

### Option A: One command (easiest)

From the project root, run:

```bash
cd /Users/anushkaarjun/synopsys/eeg-simulator-ui-2
chmod +x run_simulation.sh
./run_simulation.sh
```

This script will:

1. Install backend deps and (if needed) train the 3-class model.
2. Start the **backend** at **http://localhost:5001**.
3. Install frontend deps and start the **frontend** at **http://localhost:3000**.

Then open **http://localhost:3000** in your browser. Use **Ctrl+C** in the terminal to stop both.

---

### Option B: Two terminals (backend + frontend separately)

**Terminal 1 — Backend**

```bash
cd /Users/anushkaarjun/synopsys/eeg-simulator-ui-2/backend
pip3 install -r requirements.txt
pip3 install tensorflow
export DATA_DIR="/path/to/your/files 2"
# If you don't have the model yet:
python3 train_model_3class.py
# Start the API
python3 app.py
```

Leave it running. You should see something like: `[Model] Loaded 3-class CNN-LSTM` and the API listening on port 5001.

**Terminal 2 — Frontend**

```bash
cd /Users/anushkaarjun/synopsys/eeg-simulator-ui-2
npm install
npm run dev
```

Leave it running, then open **http://localhost:3000** in your browser.

---

### If the model is already trained

**One command:** same as Option A — `./run_simulation.sh` will skip training and start backend + frontend.

**Two terminals:**

- Terminal 1: `cd backend && export DATA_DIR="/path/to/your/files 2" && python3 app.py`
- Terminal 2: `cd eeg-simulator-ui-2 && npm run dev`

Then open **http://localhost:3000**.

---

## Ports

| Service  | URL                     |
|----------|-------------------------|
| Frontend | http://localhost:3000   |
| Backend  | http://localhost:5001   |

The UI at 3000 calls the API at 5001; no extra config is needed for localhost.
