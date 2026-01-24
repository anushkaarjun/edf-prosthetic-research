# EDF Prosthetic Research - Pipeline Flowchart

This document shows the flow of data and processing steps in the EDF prosthetic research pipeline.

## Training Pipeline

```mermaid
flowchart TD
    A[Start: EDF Files] --> B[Load EDF File]
    B --> C[Extract Raw EEG Data]
    C --> D[Preprocessing Pipeline]
    
    D --> D1[Detect Bad Channels]
    D1 --> D2[Clean Bad Data]
    D2 --> D3[Bandpass Filter 8-30 Hz]
    D3 --> D4[Resample to 250 Hz]
    D4 --> D5[Apply Smoothing/Averaging]
    D5 --> D6[Normalize to -1, 1]
    
    D6 --> E[Extract Events/Annotations]
    E --> F[Create Small Chunks 0.1-0.5s]
    F --> G[Split Train/Validation]
    
    G --> H[Training Loop]
    H --> H1[Forward Pass]
    H1 --> H2[Compute Loss]
    H2 --> H3[Backward Pass]
    H3 --> H4[Update Weights]
    H4 --> H5{More Epochs?}
    H5 -->|Yes| H1
    H5 -->|No| I[Save Trained Model]
    
    I --> J[End: Model Ready]
    
    style D fill:#e1f5ff
    style F fill:#fff4e1
    style H fill:#e8f5e9
```

## Inference Pipeline

```mermaid
flowchart TD
    A[Start: New EEG Data] --> B[Inference Buffer]
    B --> B1[Accumulate Small Chunk 0.1-0.5s]
    B1 --> B2{Buffer Ready?}
    B2 -->|No| B1
    B2 -->|Yes| C[Apply Smoothing]
    
    C --> D[Load Trained Model]
    D --> E[Preprocess Chunk]
    E --> E1[Normalize]
    E1 --> E2[Format for Model]
    
    E2 --> F[Model Forward Pass]
    F --> G[Get Predictions]
    G --> G1[Class Probabilities]
    G1 --> G2[Confidence Score]
    G2 --> G3[Predicted Class]
    
    G3 --> H{Multiple Chunks?}
    H -->|Yes| I[Aggregate Predictions]
    I --> I1[Majority Vote or Average]
    I1 --> J[Final Prediction]
    H -->|No| J
    
    J --> K[Return Result]
    K --> L[End: Prediction Ready]
    
    style B fill:#fff4e1
    style C fill:#e1f5ff
    style F fill:#e8f5e9
    style I fill:#f3e5f5
```

## Complete System Architecture

```mermaid
flowchart LR
    subgraph "Data Input"
        A1[EDF Files]
        A2[Real-time EEG Stream]
    end
    
    subgraph "Preprocessing Module"
        B1[Load Data]
        B2[Clean & Filter]
        B3[Smoothing]
        B4[Normalize]
        B5[Chunk Creation]
    end
    
    subgraph "Training Pipeline"
        C1[Create Epochs]
        C2[Train Model]
        C3[Validate]
        C4[Save Model]
    end
    
    subgraph "Inference Pipeline"
        D1[Buffer Chunks]
        D2[Preprocess]
        D3[Model Inference]
        D4[Aggregate Results]
    end
    
    subgraph "Models"
        E1[EEGNet]
        E2[ImprovedEEGNet]
        E3[CNN-LSTM]
    end
    
    subgraph "Output"
        F1[Predicted Class]
        F2[Confidence Score]
        F3[Probabilities]
    end
    
    A1 --> B1
    A2 --> B1
    B1 --> B2
    B2 --> B3
    B3 --> B4
    B4 --> B5
    
    B5 --> C1
    C1 --> C2
    C2 --> C3
    C3 --> C4
    
    B5 --> D1
    D1 --> D2
    D2 --> D3
    D3 --> D4
    
    C4 --> E1
    C4 --> E2
    C4 --> E3
    
    D3 --> E1
    D3 --> E2
    D3 --> E3
    
    E1 --> F1
    E2 --> F1
    E3 --> F1
    F1 --> F2
    F2 --> F3
    
    style B3 fill:#fff4e1
    style B5 fill:#fff4e1
    style D1 fill:#fff4e1
    style D4 fill:#f3e5f5
```

## Key Processing Steps

### 1. Data Loading
- **Input**: EDF files with EEG signals
- **Output**: Raw MNE Raw objects
- **Location**: `src/edf_ml_model/data_utils.py`, `scripts/train_*.py`

### 2. Preprocessing
- **Steps**:
  1. Detect and clean bad channels
  2. Apply bandpass filter (8-30 Hz)
  3. Resample to target frequency (250 Hz)
  4. **Apply smoothing/averaging** (NEW)
  5. Normalize to [-1, 1]
- **Location**: `src/edf_ml_model/preprocessing.py`

### 3. Chunk Creation
- **Method**: Create small chunks (0.1s to 0.5s) from continuous data
- **For Training**: Extract chunks around event markers
- **For Inference**: Sliding window chunks from continuous stream
- **Location**: `src/edf_ml_model/preprocessing.py::create_sliding_chunks()`

### 4. Model Training
- **Input**: Chunks with labels
- **Process**: Forward pass, loss computation, backpropagation
- **Output**: Trained model weights
- **Location**: `scripts/train_improved_model.py`, `src/edf_ml_model/model.py`

### 5. Model Inference
- **Input**: Small chunk of EEG data (0.1-0.5s)
- **Process**: 
  1. Apply smoothing
  2. Normalize
  3. Forward pass through model
  4. Aggregate if multiple chunks
- **Output**: Predicted class, confidence, probabilities
- **Location**: `src/edf_ml_model/inference.py`

## Data Flow Summary

```
EDF File → Raw Data → Preprocessing → Small Chunks (0.1-0.5s) → Model → Prediction
                ↓
         [Smoothing Applied]
                ↓
         [Normalization]
                ↓
         [Chunk Creation]
```

## Key Improvements (Latest Updates)

1. **Small Chunk Processing**: Changed from 30-second buffers to 0.1-0.5s chunks for faster, more responsive processing
2. **Smoothing/Averaging**: Added moving average smoothing to reduce noise before neural network processing
3. **Configurable Parameters**: Chunk size and smoothing can be configured via command-line arguments

## Component Interactions

- **Preprocessing** ↔ **Training**: Preprocessing prepares data for training
- **Preprocessing** ↔ **Inference**: Same preprocessing applied to inference data
- **Model** ↔ **Training**: Model is trained on preprocessed chunks
- **Model** ↔ **Inference**: Trained model processes inference chunks
- **Inference Buffer** ↔ **Chunk Creation**: Buffer accumulates data into chunks

## API Server Flow

```
Client Request → API Server → Load Model → Preprocess Data → 
Inference → Aggregate Results → Return Prediction
```

See `docs/EEG_API_README.md` for API details.
