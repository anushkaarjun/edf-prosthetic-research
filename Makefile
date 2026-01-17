SHELL := /bin/bash
DATA_PATH := /Users/anushkaarjun/Desktop/Outside\ of\ School/Prosethic\ Research\ Data/files\ 2
API_PORT := 8000

# ============================================================================
# Environment Setup
# ============================================================================

init:  # ENV SETUP
	uv sync --all-groups
	uv run pre-commit install
	@echo "Environment initialized with uv."

update:
	uv sync --upgrade --all-groups
	uv run pre-commit autoupdate

update-deep:
	uv cache clean pypi
	make update

clean:
	rm -rf .venv
	rm -rf .pytest_cache
	rm -rf build/
	rm -rf dist/
	rm -rf junit-pytest.xml
	rm -rf logs/*
	find . -name ".coverage*" -delete
	find . -name "coverage.xml" -delete
	find . -name "__pycache__" -exec rm -r {} +

# ============================================================================
# Code Quality
# ============================================================================

test:
	uv run pytest --cov=src --cov-report=term-missing --no-cov-on-fail --cov-report=xml --cov-fail-under=90
	rm .coverage

lint:
	uv run ruff format
	uv run ruff check --fix

typecheck:
	uv run pyright src

format:
	make lint
	make typecheck

# ============================================================================
# Model Training
# ============================================================================

train-csp-svm:
	@echo "Training CSP+SVM model..."
	python3 scripts/train_on_validation_data.py --data-path "$(DATA_PATH)" --max-subjects 5 --no-eegnet --csp-svm

train-eegnet:
	@echo "Training EEGNet model..."
	python3 scripts/train_on_validation_data.py --data-path "$(DATA_PATH)" --max-subjects 5 --eegnet --no-csp-svm

train-cnn-lstm:
	@echo "Training CNN-LSTM model..."
	python3 scripts/train_cnn_lstm.py --data-path "$(DATA_PATH)" --max-subjects 5

train-improved:
	@echo "Training improved neural network model..."
	python3 scripts/train_improved_model.py --data-path "$(DATA_PATH)" --max-subjects 5 --epochs 50

train-all:
	@echo "Training all models..."
	python3 scripts/train_on_validation_data.py --data-path "$(DATA_PATH)" --max-subjects 5 --eegnet --csp-svm
	python3 scripts/train_cnn_lstm.py --data-path "$(DATA_PATH)" --max-subjects 5
	python3 scripts/train_improved_model.py --data-path "$(DATA_PATH)" --max-subjects 5 --epochs 50

# ============================================================================
# API Server
# ============================================================================

api-server:
	@echo "Starting EEG API server on port $(API_PORT)..."
	python3 scripts/eeg_api_server.py

api-server-kill:
	@echo "Stopping API server on port $(API_PORT)..."
	-lsof -ti:$(API_PORT) | xargs kill -9 2>/dev/null || echo "No process found on port $(API_PORT)"

load-models:
	@echo "Loading trained models into API server..."
	python3 scripts/load_models.py

load-csp-svm:
	@echo "Loading CSP+SVM model..."
	curl -X POST "http://localhost:$(API_PORT)/load_model" \
		-H "Content-Type: application/json" \
		-d '{"model_type_param": "csp_svm", "model_path": "$(shell pwd)/models/csp_svm_model.pkl"}'

load-eegnet:
	@echo "Loading EEGNet model..."
	curl -X POST "http://localhost:$(API_PORT)/load_model" \
		-H "Content-Type: application/json" \
		-d '{"model_type_param": "eegnet", "model_path": "$(shell pwd)/models/eegnet_trained.pth"}'

load-cnn-lstm:
	@echo "Loading CNN-LSTM model..."
	curl -X POST "http://localhost:$(API_PORT)/load_model" \
		-H "Content-Type: application/json" \
		-d '{"model_type_param": "cnn_lstm", "model_path": "$(shell pwd)/models/best_model.pth", "n_channels": 64}'

load-improved:
	@echo "Loading ImprovedEEGNet model..."
	python3 scripts/load_improved_model.py

api-health:
	@echo "Checking API health..."
	curl -s http://localhost:$(API_PORT)/health | python3 -m json.tool

# ============================================================================
# React Simulator
# ============================================================================

react-simulator:
	@echo "To start React simulator:"
	@echo "  cd ../eeg-simulator-ui-2"
	@echo "  npm start"

# ============================================================================
# Documentation
# ============================================================================

tree:
	uv run python repo_tree.py --update-readme

app:
	uv run python -m edf_ml_model
