#!/usr/bin/env bash
# Start the backend with your EDF data. Uses python3 (macOS).
cd "$(dirname "$0")"
export DATA_DIR="${DATA_DIR:-/Users/anushkaarjun/Desktop/Outside of School/Prosethic Research Data/files 2}"
echo "DATA_DIR=$DATA_DIR"
exec python3 app.py
