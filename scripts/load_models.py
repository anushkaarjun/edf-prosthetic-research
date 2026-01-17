#!/usr/bin/env python3
"""
Script to load trained models into the API server.
"""
import requests
import json
import os

API_URL = "http://localhost:8000"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_models():
    """Load both CSP+SVM and EEGNet models into the API server."""
    
    print("Loading trained models into API server...")
    print(f"API URL: {API_URL}\n")
    
    # Check if API server is running
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        print("✓ API server is running")
    except requests.exceptions.RequestException:
        print("✗ API server is not running. Please start it first:")
        print(f"  cd {BASE_DIR}")
        print("  python3 eeg_api_server.py")
        return
    
    # Load CSP+SVM model
    print("\n1. Loading CSP+SVM model...")
    try:
        response = requests.post(
            f"{API_URL}/load_model",
            json={
                "model_type_param": "csp_svm",
                "model_path": f"{BASE_DIR}/csp_svm_model.pkl"
            },
            timeout=10
        )
        response.raise_for_status()
        result = response.json()
        print(f"  ✓ {result.get('status', 'Model loaded')}")
    except Exception as e:
        print(f"  ✗ Error loading CSP+SVM: {e}")
    
    # Load EEGNet model
    print("\n2. Loading EEGNet model...")
    try:
        response = requests.post(
            f"{API_URL}/load_model",
            json={
                "model_type_param": "eegnet",
                "model_path": f"{BASE_DIR}/eegnet_trained.pth"
            },
            timeout=10
        )
        response.raise_for_status()
        result = response.json()
        print(f"  ✓ {result.get('status', 'Model loaded')}")
    except Exception as e:
        print(f"  ✗ Error loading EEGNet: {e}")
    
    # Check health to see loaded models
    print("\n3. Checking API status...")
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        health = response.json()
        print(f"  Model loaded: {health.get('model_loaded', False)}")
        print(f"  Model type: {health.get('model_type', 'None')}")
        print(f"  Classes: {health.get('classes', 'None')}")
    except Exception as e:
        print(f"  ✗ Error checking health: {e}")
    
    print("\nDone! Models are loaded and ready to use.")

if __name__ == "__main__":
    load_models()
