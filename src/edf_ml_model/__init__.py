"""EDF ML Model package for reading and processing EDF brain scan files."""

from edf_ml_model.edf_parser import (
    EDFAnnotation,
    EDFHeader,
    EDFParser,
    read_edf,
)

__all__ = [
    "EDFAnnotation",
    "EDFHeader",
    "EDFParser",
    "read_edf",
]
