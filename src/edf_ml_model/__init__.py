"""EDF ML Model package for reading and processing EDF brain scan files."""

from edf_ml_model.data_utils import annotation_to_motion, get_run_number
from edf_ml_model.edf_parser import (
    EDFAnnotation,
    EDFHeader,
    EDFParser,
    read_edf,
)
from edf_ml_model.inference import InferenceBuffer, predict_with_confidence
from edf_ml_model.model import (
    EEGMotorImageryNet,
    train_model,
    train_model_with_hyperparams,
    tune_hyperparameters,
)
from edf_ml_model.preprocessing import (
    EPOCH_WINDOW,
    FREQ_HIGH,
    FREQ_LOW,
    INFERENCE_BUFFER_SECONDS,
    TARGET_SFREQ,
    clean_data,
    compute_spectral_analysis,
    create_half_second_epochs,
    detect_bad_data,
    normalize_signal,
    preprocess_raw,
)

__all__ = [
    "EPOCH_WINDOW",
    "FREQ_HIGH",
    "FREQ_LOW",
    "INFERENCE_BUFFER_SECONDS",
    "TARGET_SFREQ",
    "EDFAnnotation",
    "EDFHeader",
    "EDFParser",
    "EEGMotorImageryNet",
    "InferenceBuffer",
    "annotation_to_motion",
    "clean_data",
    "compute_spectral_analysis",
    "create_half_second_epochs",
    "detect_bad_data",
    "get_run_number",
    "normalize_signal",
    "predict_with_confidence",
    "preprocess_raw",
    "read_edf",
    "train_model",
    "train_model_with_hyperparams",
    "tune_hyperparameters",
]
