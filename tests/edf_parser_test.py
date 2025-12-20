"""Tests for the EDF parser module."""

from datetime import datetime
from pathlib import Path

import pytest

from edf_ml_model.edf_parser import EDFHeader, EDFParser


def test_edf_parser_initialization(tmp_path: Path) -> None:
    """Test EDF parser initialization with non-existent file."""
    fake_file = tmp_path / "nonexistent.edf"
    with pytest.raises(FileNotFoundError):
        EDFParser(fake_file)


def test_edf_header_properties() -> None:
    """Test EDFHeader properties."""

    header = EDFHeader(
        version="0",
        patient_id="Test Patient",
        recording_id="Test Recording",
        start_date="01.01.23",
        start_time="12.00.00",
        header_bytes=256,
        reserved="",
        num_records=10,
        record_duration=1.0,
        num_signals=2,
        signal_labels=["EEG1", "EEG2"],
        transducer_types=["", ""],
        physical_dimensions=["uV", "uV"],
        physical_min=[-100.0, -100.0],
        physical_max=[100.0, 100.0],
        digital_min=[-32768, -32768],
        digital_max=[32767, 32767],
        prefiltering=["", ""],
        num_samples_per_record=[256, 256],
        reserved_signals=["", ""],
    )

    # Test sample frequencies
    sample_freqs = header.sample_frequencies
    assert len(sample_freqs) == 2
    assert sample_freqs[0] == 256.0
    assert sample_freqs[1] == 256.0

    # Test datetime parsing
    dt = header.start_datetime
    assert isinstance(dt, datetime)


def test_read_edf_convenience_function(tmp_path: Path) -> None:
    """Test the read_edf convenience function."""
    # This is a placeholder test - would need actual EDF file for full test
    from edf_ml_model.edf_parser import read_edf

    fake_file = tmp_path / "test.edf"
    with pytest.raises(FileNotFoundError):
        read_edf(fake_file)

