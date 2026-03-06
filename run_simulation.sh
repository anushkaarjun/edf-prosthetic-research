#!/usr/bin/env bash
# Run the full EEG simulation: train model (if needed), start backend, then frontend.
# Usage: ./run_simulation.sh

set -e
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
BACKEND="$PROJECT_ROOT/backend"
DATA_DIR="${DATA_DIR:-/Users/anushkaarjun/Desktop/Outside of School/Prosethic Research Data/files 2}"

echo "=== EEG Simulation ==="
echo "DATA_DIR=$DATA_DIR"
echo ""

# 1) Backend: install deps
echo ">>> Installing backend dependencies..."
cd "$BACKEND"
pip3 install -q -r requirements.txt
pip3 install -q tensorflow 2>/dev/null || true

# 2) Train 3-class model if not present
if [ ! -f "$BACKEND/saved_model_3class.keras" ]; then
  echo ">>> Training 3-class CNN-LSTM (this may take a few minutes)..."
  export DATA_DIR
  python3 train_model_3class.py
else
  echo ">>> Using existing saved_model_3class.keras (delete it to retrain)"
fi

# 3) Start backend in background
echo ">>> Starting API on http://localhost:5001..."
export DATA_DIR
python3 app.py &
BACKEND_PID=$!
sleep 3

# 4) Frontend: install and run
echo ">>> Installing frontend dependencies..."
cd "$PROJECT_ROOT"
npm install --silent 2>/dev/null || true
echo ">>> Starting React app on http://localhost:3000..."
npm run dev &
FRONTEND_PID=$!

echo ""
echo "=============================================="
echo "  Backend:  http://localhost:5001  (PID $BACKEND_PID)"
echo "  Frontend: http://localhost:3000  (PID $FRONTEND_PID)"
echo "  Open http://localhost:3000 in your browser."
echo "  Press Ctrl+C to stop both servers."
echo "=============================================="

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
wait
