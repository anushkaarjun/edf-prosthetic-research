#!/usr/bin/env python3
"""
Load the improved CNN-LSTM model into the API server.
Only loads if accuracy is higher than baseline (CNN-LSTM at 51.94%).
"""
import requests
import os
import sys
import re

API_URL = "http://localhost:8000"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_ACC = 51.94  # Original CNN-LSTM current best accuracy

def get_improved_cnn_lstm_accuracy():
    """Extract accuracy from results file."""
    results_file = os.path.join(BASE_DIR, "results", "improved_cnn_lstm_results.txt")
    
    # Also check for generic results file
    if not os.path.exists(results_file):
        results_file = os.path.join(BASE_DIR, "results", "training_output.txt")
    
    if not os.path.exists(results_file):
        return None
    
    try:
        with open(results_file, 'r') as f:
            content = f.read()
            # Look for validation accuracy patterns
            patterns = [
                r'Validation Accuracy:\s*0?\.?(\d+)',  # "Validation Accuracy: 0.5508"
                r'Val Acc.*?:\s*0?\.?(\d+)',           # "Val Acc: 0.5508"
                r'(\d+\.\d+)%.*accuracy',              # "55.08% accuracy"
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    # Convert to percentage
                    if '.' in matches[-1]:
                        return float(matches[-1])
                    else:
                        # Assume decimal if < 1
                        val = float(f"0.{matches[-1]}")
                        return val * 100
            
            # Try to find percentage directly
            pct_match = re.search(r'(\d+\.\d+)%', content)
            if pct_match:
                return float(pct_match.group(1))
                
    except Exception as e:
        print(f"⚠️  Error reading results file: {e}")
    
    return None


def load_improved_cnn_lstm_model():
    """Load the improved CNN-LSTM model into the API server."""
    model_path = os.path.join(BASE_DIR, "models", "best_improved_cnn_lstm.pth")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("💡 Make sure training has completed and model is saved.")
        print("   Train with: make train-improved-cnn-lstm")
        return False
    
    # Check accuracy before loading
    print("📊 Checking improved CNN-LSTM model accuracy...")
    improved_acc = get_improved_cnn_lstm_accuracy()
    
    if improved_acc is None:
        print("⚠️  Could not determine improved model accuracy")
        print("💡 Cannot verify if model should be loaded.")
        response = input("   Load anyway? (yes/no): ")
        if response.lower() != 'yes':
            print("❌ Aborting - accuracy check required")
            return False
    else:
        print(f"📊 Improved CNN-LSTM accuracy: {improved_acc:.2f}%")
        print(f"📊 Baseline (CNN-LSTM) accuracy: {BASELINE_ACC}%")
        
        if improved_acc <= BASELINE_ACC:
            diff = BASELINE_ACC - improved_acc
            print(f"⚠️  Improved model accuracy ({improved_acc:.2f}%) is LOWER than baseline ({BASELINE_ACC}%)")
            print(f"📉 Difference: -{diff:.2f}%")
            print("❌ Not loading improved model - keeping current best model active")
            return False
        else:
            diff = improved_acc - BASELINE_ACC
            print(f"✅ Improved model accuracy ({improved_acc:.2f}%) is HIGHER than baseline ({BASELINE_ACC}%)")
            print(f"📈 Improvement: +{diff:.2f}%")
    
    print(f"\n📦 Loading improved CNN-LSTM model from: {model_path}")
    
    try:
        response = requests.post(
            f"{API_URL}/load_model",
            json={
                "model_type_param": "improved_cnn_lstm",
                "model_path": model_path,
                "n_channels": 64
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ {data['status']}")
            print(f"📊 Model Type: {data['model_type']}")
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to API server at {API_URL}")
        print("💡 Make sure the API server is running: make api-server")
        return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Loading Improved CNN-LSTM model into API server...")
    print(f"📍 API URL: {API_URL}")
    print()
    
    success = load_improved_cnn_lstm_model()
    
    if success:
        print()
        print("✅ Improved CNN-LSTM model loaded successfully!")
        print("🎉 You can now use it in the React simulator.")
        print()
        print("Next steps:")
        print("1. Refresh the React simulator")
        print("2. Check /health endpoint to verify model is loaded")
        print("3. Test predictions with real EEG data")
    else:
        print()
        print("❌ Failed to load model. Check errors above.")
        sys.exit(1)
