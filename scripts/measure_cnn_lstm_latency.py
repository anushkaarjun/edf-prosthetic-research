#!/usr/bin/env python3
"""
Measure inference latency for CNN-LSTM model.
Loads the model and runs inference multiple times to report latency statistics.
"""
import os
import sys
import time
import argparse
import numpy as np
import torch

# Add scripts dir for imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.join(_script_dir, "..")
sys.path.insert(0, _script_dir)
sys.path.insert(0, _repo_root)

from cnn_lstm_model import CNNLSTM, N_CLASSES, CLASS_NAMES


def measure_latency(
    model_path: str,
    n_channels: int = 64,
    n_classes: int = None,
    n_samples: int = 100,
    warmup: int = 10,
    device: str = None,
) -> dict:
    """
    Measure CNN-LSTM inference latency.
    
    Args:
        model_path: Path to saved model (.pth)
        n_channels: Number of EEG channels (default 64)
        n_classes: Number of classes (inferred from state_dict if None)
        n_samples: Number of inference runs for latency measurement
        warmup: Number of warm-up runs before measurement
        device: 'cpu' or 'cuda' (auto-detected if None)
    
    Returns:
        dict with avg_ms, std_ms, min_ms, max_ms, median_ms
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    state_dict = torch.load(model_path, map_location=device)
    
    # Infer n_classes from state_dict if not provided
    if n_classes is None:
        fc_weight = state_dict.get("fc.weight", state_dict.get("module.fc.weight"))
        if fc_weight is not None:
            n_classes = fc_weight.shape[0]
        else:
            n_classes = N_CLASSES
    
    model = CNNLSTM(n_channels=n_channels, n_classes=n_classes, dropout=0.0)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device)
    
    # Create dummy input: (batch, 1, n_channels, n_samples)
    # CNN-LSTM expects 320 time samples (2s at 160Hz)
    n_times = 320
    X = torch.randn(1, 1, n_channels, n_times, dtype=torch.float32).to(device)
    
    # Warm-up
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(X)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Measure latency
    latencies = []
    with torch.no_grad():
        for _ in range(n_samples):
            start = time.perf_counter()
            _ = model(X)
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms
    
    latencies = np.array(latencies)
    return {
        "avg_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "median_ms": float(np.median(latencies)),
    }


def main():
    parser = argparse.ArgumentParser(description="Measure CNN-LSTM inference latency")
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.path.join(_repo_root, "models", "best_model.pth"),
        help="Path to CNN-LSTM model (.pth)",
    )
    parser.add_argument("--n-channels", type=int, default=64, help="Number of EEG channels")
    parser.add_argument("--n-classes", type=int, default=None, help="Number of classes (auto from state_dict)")
    parser.add_argument("--n-samples", type=int, default=100, help="Number of runs for latency measurement")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warm-up runs")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"], help="Device (auto if not set)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"ERROR: Model not found at {args.model_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("CNN-LSTM Inference Latency Measurement")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Device: {args.device or ('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"Samples: {args.n_samples} (warmup: {args.warmup})")
    print()
    
    stats = measure_latency(
        model_path=args.model_path,
        n_channels=args.n_channels,
        n_classes=args.n_classes,
        n_samples=args.n_samples,
        warmup=args.warmup,
        device=args.device,
    )
    
    print("Results:")
    print(f"  Average:  {stats['avg_ms']:.2f} ms")
    print(f"  Std Dev:  {stats['std_ms']:.2f} ms")
    print(f"  Min:      {stats['min_ms']:.2f} ms")
    print(f"  Max:      {stats['max_ms']:.2f} ms")
    print(f"  Median:   {stats['median_ms']:.2f} ms")
    print()
    print(f"  Threshold: ≤ 1000 ms (1 second)")
    passed = stats["avg_ms"] <= 1000
    print(f"  Result:   {'✓ PASS' if passed else '✗ FAIL'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
