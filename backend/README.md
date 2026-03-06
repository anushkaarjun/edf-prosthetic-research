# EEG CNN-LSTM Backend

Serves validation data and runs your CNN-LSTM model for the React UI.

## Setup

```bash
cd backend
pip install -r requirements.txt
```

## Using your EDF data ("files 2")

To use your BCI-style EDF folder (e.g. **Prosthetic Research Data / files 2** with S001, S002, … and `.edf` / `.edf.event` files):

```bash
export DATA_DIR="/Users/anushkaarjun/Desktop/Outside of School/Prosethic Research Data/files 2"
python3 app.py
```
(On macOS use **python3**, not `python`.)

Or run the script (it uses that path by default):

```bash
chmod +x run_with_data.sh
./run_with_data.sh
```

The backend will load trials from a few subjects (S001–S005) and runs R03–R07, parse event files for labels (T0=Rest, T1=Left Hand, T2=Right Hand, T3=Both Feet, T4=Both Fists), and serve them as validation samples. The React app will show "API Connected" and real EEG when you click **Run validation** or **Next sample**.

## Validation data (alternative: NumPy)

You can instead use precomputed NumPy arrays:

- **VALIDATION_DATA_PATH**: NumPy file shape `(N, 64, T)` or `(N, T, 64)` (N samples, 64 channels, T time points).
- **VALIDATION_LABELS_PATH**: NumPy file shape `(N,)` with class indices 0–4 (Left Hand, Right Hand, Both Feet, Both Fists, Rest).

If neither DATA_DIR nor these are set, the API returns mock data so the UI still runs.

## 2-class CNN-LSTM model (train and load)

The app uses a **2-class** model: **Rest** vs **Motor**. To train the included CNN-LSTM on your EDF data and have the app load it:

```bash
cd backend
export DATA_DIR="/Users/anushkaarjun/Desktop/Outside of School/Prosethic Research Data/files 2"
pip install -r requirements.txt   # includes tensorflow
python3 train_model.py
```

This saves **saved_model.keras** in the backend folder. When you start the API with `python3 app.py`, it will load this file automatically (no need to set `MODEL_PATH`). The UI will then show **"Model Loaded"** and predictions will come from the trained model.

To use your own saved model instead, set **MODEL_PATH**:

- Keras: `.h5` or `.keras` (input shape `(batch, 64, T)`).
- PyTorch: `.pt` (input `(1, 64, T)`).

Server runs at **http://localhost:5001**. The React app will show "API Connected (Model Loaded)" when the model is loaded.
