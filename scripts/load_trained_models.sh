#!/bin/bash
# Script to load trained models into the API server

API_URL="http://localhost:8000"
BASE_DIR="/Users/anushkaarjun/synopsys/edf-prosthetic-research"

echo "Loading trained models into API server..."
echo ""

# Load CSP+SVM model
echo "1. Loading CSP+SVM model..."
curl -X POST "${API_URL}/load_model" \
  -H "Content-Type: application/json" \
  -d "{
    \"model_type_param\": \"csp_svm\",
    \"model_path\": \"${BASE_DIR}/csp_svm_model.pkl\"
  }"

echo ""
echo ""

# Load EEGNet model
echo "2. Loading EEGNet model..."
curl -X POST "${API_URL}/load_model" \
  -H "Content-Type: application/json" \
  -d "{
    \"model_type_param\": \"eegnet\",
    \"model_path\": \"${BASE_DIR}/eegnet_trained.pth\"
  }"

echo ""
echo ""

# Check health
echo "3. Checking API health..."
curl "${API_URL}/health"

echo ""
echo ""
echo "Done! Models should now be loaded."
