"""EDF ML Model package for reading and processing EDF brain scan files."""

from edf_ml_model.edf_parser import (
    EDFAnnotation,
    EDFHeader,
    EDFParser,
    read_edf,
)
from edf_ml_model.data_utils import get_run_number, annotation_to_motion
from edf_ml_model.preprocessing import (
    normalize_signal,
    compute_spectral_analysis,
    detect_bad_data,
    clean_data,
    preprocess_raw,
    create_half_second_epochs,
    TARGET_SFREQ,
    FREQ_LOW,
    FREQ_HIGH,
    EPOCH_WINDOW,
    INFERENCE_BUFFER_SECONDS,
)
from edf_ml_model.model import EEGMotorImageryNet, train_model
from edf_ml_model.inference import InferenceBuffer, predict_with_confidence

__all__ = [
    "EDFAnnotation",
    "EDFHeader",
    "EDFParser",
    "read_edf",
    "get_run_number",
    "annotation_to_motion",
    "normalize_signal",
    "compute_spectral_analysis",
    "detect_bad_data",
    "clean_data",
    "preprocess_raw",
    "create_half_second_epochs",
    "EEGMotorImageryNet",
    "train_model",
    "InferenceBuffer",
    "predict_with_confidence",
    "TARGET_SFREQ",
    "FREQ_LOW",
    "FREQ_HIGH",
    "EPOCH_WINDOW",
    "INFERENCE_BUFFER_SECONDS",
]
