#!/usr/bin/env bash
# Run the backend using your EDF data folder so the visualization shows real validation data.
# Replace the path below with your actual "files 2" path if different.

DATA_DIR="${DATA_DIR:-/Users/anushkaarjun/Desktop/Outside of School/Prosethic Research Data/files 2}"
export DATA_DIR

# Optional: set if you have a trained CNN-LSTM model
# export MODEL_PATH=./model.keras

echo "Using DATA_DIR=$DATA_DIR"
python3 app.py
