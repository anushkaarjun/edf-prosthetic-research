#!/bin/bash
# EEG Simulator: start backend then frontend. Frontend talks to http://localhost:5001
cd "$(dirname "$0")"
echo ">>> Backend (Flask) will run at http://localhost:5001"
echo ">>> Start it in another terminal with:  cd backend && python3 app.py"
echo ">>> Then run:  npm run dev   and open http://localhost:5173"
echo ""
npm run dev
