#!/bin/bash
# Start backend (Flask) then frontend (Vite). Backend must be on port 5001 for the app to connect.
cd "$(dirname "$0")"
ROOT="$(pwd)"

echo ">>> Starting backend on http://localhost:5001 ..."
cd "$ROOT/backend"
python3 app.py &
BACKEND_PID=$!
cd "$ROOT"

echo ">>> Waiting 3 seconds for backend to start..."
sleep 3

echo ">>> Starting frontend at http://localhost:3000 ..."
echo ">>> Open http://localhost:3000 in your browser."
npm run dev

echo ">>> Shutting down backend (PID $BACKEND_PID)..."
kill $BACKEND_PID 2>/dev/null
