"""EDF file parser for reading European Data Format files.

This module provides functionality to read and parse EDF (European Data Format)
and EDF+ files, which are commonly used for storing EEG, EMG, and other
physiological signal data.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger


@dataclass
class EDFHeader:
    """EDF file header information."""

    version: str
    patient_id: str
    recording_id: str
    start_date: str
    start_time: str
    header_bytes: int
    reserved: str
    num_records: int
    record_duration: float
    num_signals: int
    signal_labels: list[str]
    transducer_types: list[str]
    physical_dimensions: list[str]
    physical_min: list[float]
    physical_max: list[float]
    digital_min: list[int]
    digital_max: list[int]
    prefiltering: list[str]
    num_samples_per_record: list[int]
    reserved_signals: list[str]

    @property
    def sample_frequencies(self) -> list[float]:
        """Calculate sample frequencies for each signal.

        :return: List of sample frequencies in Hz.
        """
        return [ns / self.record_duration for ns in self.num_samples_per_record]

    @property
    def start_datetime(self) -> datetime:
        """Parse start date and time into datetime object.

        :return: Start datetime object.
        """
        date_str = self.start_date.strip()
        time_str = self.start_time.strip()

        # EDF date format: DD.MM.YY, time format: HH.MM.SS
        try:
            dt = datetime.strptime(f"{date_str} {time_str}", "%d.%m.%y %H.%M.%S")
            return dt
        except ValueError:
            logger.warning(f"Could not parse datetime from {date_str} {time_str}")
            return datetime.now()


@dataclass
class EDFAnnotation:
    """EDF annotation/event information."""

    onset: float  # Time in seconds from start of recording
    duration: float  # Duration in seconds
    description: str  # Annotation text/event code


class EDFParser:
    """Parser for EDF and EDF+ files.

    Supports reading header information, signal data, and annotations
    from European Data Format files commonly used for physiological data.
    """

    def __init__(self, filepath: str | Path, verbose: bool = False) -> None:
        """Initialize EDF parser.

        :param filepath: Path to the EDF file.
        :param verbose: Whether to print verbose output.
        """
        self.filepath = Path(filepath)
        self.verbose = verbose

        if not self.filepath.exists():
            raise FileNotFoundError(f"EDF file not found: {self.filepath}")

        self.header: EDFHeader | None = None
        self._file_handle = None

    def read_header(self) -> EDFHeader:
        """Read and parse EDF file header.

        :return: EDFHeader object containing header information.
        """
        with open(self.filepath, "rb") as f:
            # Version (8 bytes)
            version = f.read(8).strip().decode("ascii", errors="ignore")

            # Patient ID (80 bytes)
            patient_id = f.read(80).strip().decode("ascii", errors="ignore")

            # Recording ID (80 bytes)
            recording_id = f.read(80).strip().decode("ascii", errors="ignore")

            # Start date (8 bytes) - format: DD.MM.YY
            start_date = f.read(8).strip().decode("ascii", errors="ignore")

            # Start time (8 bytes) - format: HH.MM.SS
            start_time = f.read(8).strip().decode("ascii", errors="ignore")

            # Header bytes (8 bytes)
            header_bytes = int(f.read(8).strip().decode("ascii"))

            # Reserved (44 bytes)
            reserved = f.read(44).strip().decode("ascii", errors="ignore")

            # Number of data records (8 bytes)
            num_records = int(f.read(8).strip().decode("ascii"))

            # Duration of data record in seconds (8 bytes)
            record_duration = float(f.read(8).strip().decode("ascii"))

            # Number of signals (4 bytes)
            num_signals = int(f.read(4).strip().decode("ascii"))

            # Signal information (16 bytes each for various fields)
            signal_labels = [
                f.read(16).strip().decode("ascii", errors="ignore")
                for _ in range(num_signals)
            ]
            transducer_types = [
                f.read(80).strip().decode("ascii", errors="ignore")
                for _ in range(num_signals)
            ]
            physical_dimensions = [
                f.read(8).strip().decode("ascii", errors="ignore")
                for _ in range(num_signals)
            ]
            physical_min = [
                float(f.read(8).strip().decode("ascii")) for _ in range(num_signals)
            ]
            physical_max = [
                float(f.read(8).strip().decode("ascii")) for _ in range(num_signals)
            ]
            digital_min = [
                int(f.read(8).strip().decode("ascii")) for _ in range(num_signals)
            ]
            digital_max = [
                int(f.read(8).strip().decode("ascii")) for _ in range(num_signals)
            ]
            prefiltering = [
                f.read(80).strip().decode("ascii", errors="ignore")
                for _ in range(num_signals)
            ]
            num_samples_per_record = [
                int(f.read(8).strip().decode("ascii")) for _ in range(num_signals)
            ]
            reserved_signals = [
                f.read(32).strip().decode("ascii", errors="ignore")
                for _ in range(num_signals)
            ]

        header = EDFHeader(
            version=version,
            patient_id=patient_id,
            recording_id=recording_id,
            start_date=start_date,
            start_time=start_time,
            header_bytes=header_bytes,
            reserved=reserved,
            num_records=num_records,
            record_duration=record_duration,
            num_signals=num_signals,
            signal_labels=signal_labels,
            transducer_types=transducer_types,
            physical_dimensions=physical_dimensions,
            physical_min=physical_min,
            physical_max=physical_max,
            digital_min=digital_min,
            digital_max=digital_max,
            prefiltering=prefiltering,
            num_samples_per_record=num_samples_per_record,
            reserved_signals=reserved_signals,
        )

        self.header = header

        if self.verbose:
            logger.info(f"Read EDF header from {self.filepath}")
            logger.info(f"  Signals: {header.num_signals}")
            logger.info(f"  Records: {header.num_records}")
            logger.info(
                f"  Duration: {header.record_duration * header.num_records:.2f} seconds"
            )

        return header

    def read_signal(
        self,
        signal_index: int | None = None,
        signal_label: str | None = None,
        start_record: int = 0,
        num_records: int | None = None,
    ) -> tuple[np.ndarray, float]:
        """Read signal data from EDF file.

        :param signal_index: Index of the signal to read (0-based).
        :param signal_label: Label of the signal to read (alternative to signal_index).
        :param start_record: Starting data record (0-based).
        :param num_records: Number of records to read (None = all remaining).
        :return: Tuple of (signal_data, sample_frequency).
        :raises ValueError: If signal_index or signal_label is invalid.
        """
        if self.header is None:
            self.read_header()

        # Find signal index
        if signal_label is not None:
            try:
                signal_index = self.header.signal_labels.index(signal_label.strip())
            except ValueError as e:
                raise ValueError(
                    f"Signal label '{signal_label}' not found. Available labels: {self.header.signal_labels}"
                ) from e
        elif signal_index is None:
            raise ValueError("Either signal_index or signal_label must be provided")

        if signal_index < 0 or signal_index >= self.header.num_signals:
            raise ValueError(
                f"Signal index {signal_index} out of range [0, {self.header.num_signals})"
            )

        # Determine number of records to read
        if num_records is None:
            num_records = self.header.num_records - start_record

        if start_record + num_records > self.header.num_records:
            raise ValueError(
                f"Requested records [{start_record}, {start_record + num_records}) "
                f"exceed total records {self.header.num_records}"
            )

        # Calculate offsets
        samples_per_record = self.header.num_samples_per_record[signal_index]
        bytes_per_sample = 2  # EDF uses 16-bit integers

        # Calculate byte positions
        record_size_bytes = sum(
            ns * bytes_per_sample for ns in self.header.num_samples_per_record
        )

        # Offset to start of data records
        data_start = self.header.header_bytes

        # Offset to specific signal within each record
        signal_offset_in_record = sum(
            self.header.num_samples_per_record[i] * bytes_per_sample
            for i in range(signal_index)
        )

        # Read signal data
        signal_data = []
        with open(self.filepath, "rb") as f:
            for record_idx in range(start_record, start_record + num_records):
                # Position at start of this record
                record_start = data_start + (record_idx * record_size_bytes)

                # Position at start of this signal within the record
                signal_start = record_start + signal_offset_in_record

                f.seek(signal_start)
                samples_bytes = f.read(samples_per_record * bytes_per_sample)

                # Convert bytes to numpy array (little-endian 16-bit signed integers)
                samples = np.frombuffer(samples_bytes, dtype="<i2")

                signal_data.append(samples)

        # Concatenate all records
        signal_array = np.concatenate(signal_data)

        # Convert from digital to physical units
        dig_min = self.header.digital_min[signal_index]
        dig_max = self.header.digital_max[signal_index]
        phys_min = self.header.physical_min[signal_index]
        phys_max = self.header.physical_max[signal_index]

        # Handle division by zero
        if dig_max - dig_min == 0:
            logger.warning(
                f"Digital range is zero for signal {signal_index}, using raw values"
            )
            signal_array = signal_array.astype(np.float32)
        else:
            # Linear scaling: physical = (digital - dig_min) * (phys_max - phys_min) / (dig_max - dig_min) + phys_min
            signal_array = (signal_array - dig_min) * (phys_max - phys_min) / (
                dig_max - dig_min
            ) + phys_min

        sample_freq = self.header.sample_frequencies[signal_index]

        if self.verbose:
            logger.info(
                f"Read signal {signal_index} ({self.header.signal_labels[signal_index]}): "
                f"{len(signal_array)} samples at {sample_freq:.2f} Hz"
            )

        return signal_array, sample_freq

    def read_all_signals(
        self, start_record: int = 0, num_records: int | None = None
    ) -> tuple[np.ndarray, list[float]]:
        """Read all signals from EDF file.

        :param start_record: Starting data record (0-based).
        :param num_records: Number of records to read (None = all remaining).
        :return: Tuple of (signal_data array of shape (num_signals, num_samples), sample_frequencies).
        """
        if self.header is None:
            self.read_header()

        all_signals = []
        sample_freqs = []

        for signal_idx in range(self.header.num_signals):
            signal_data, sample_freq = self.read_signal(
                signal_index=signal_idx,
                start_record=start_record,
                num_records=num_records,
            )
            all_signals.append(signal_data)
            sample_freqs.append(sample_freq)

        # Stack signals into array (num_signals, num_samples)
        signals_array = np.array(all_signals, dtype=np.float32)

        return signals_array, sample_freqs

    def read_annotations(self) -> list[EDFAnnotation]:
        """Read annotations from EDF+ file.

        EDF+ annotations are stored in the first signal if the file contains
        a signal labeled "EDF Annotations".

        :return: List of EDFAnnotation objects.
        """
        if self.header is None:
            self.read_header()

        annotations = []

        # Check if there's an annotations signal (EDF+ format)
        try:
            ann_idx = self.header.signal_labels.index("EDF Annotations")
        except ValueError:
            if self.verbose:
                logger.info("No EDF Annotations signal found (may not be EDF+ format)")
            return annotations

        # Read the annotation signal
        ann_signal, _ = self.read_signal(signal_index=ann_idx)

        # Parse annotations (EDF+ annotation format is TAL - Time-stamped Annotation List)
        # Format: +<onset>21<duration>21<description>20
        # where 21 is 0x15 (DC1), 20 is 0x14 (DC4)
        try:
            # Convert signal to bytes (assuming it's already in the right format)
            # This is simplified - full TAL parsing is more complex
            ann_bytes = ann_signal.astype(np.uint8).tobytes()

            # Find annotation markers and parse
            # This is a simplified parser - full EDF+ TAL format is more complex
            current_onset = 0.0
            current_duration = 0.0
            current_desc = ""

            i = 0
            while i < len(ann_bytes):
                if ann_bytes[i] == 0x15:  # DC1 - separator
                    if current_desc:
                        annotations.append(
                            EDFAnnotation(
                                onset=current_onset,
                                duration=current_duration,
                                description=current_desc,
                            )
                        )
                    current_duration = 0.0
                    current_desc = ""
                    i += 1
                elif ann_bytes[i] == 0x14:  # DC4 - end marker
                    if current_desc:
                        annotations.append(
                            EDFAnnotation(
                                onset=current_onset,
                                duration=current_duration,
                                description=current_desc,
                            )
                        )
                    current_onset = 0.0
                    current_duration = 0.0
                    current_desc = ""
                    i += 1
                else:
                    # Collect description text
                    current_desc += chr(ann_bytes[i])
                    i += 1

        except Exception as e:
            logger.warning(f"Could not parse annotations: {e}")

        if self.verbose:
            logger.info(f"Read {len(annotations)} annotations")

        return annotations

    def get_info(self) -> dict[str, Any]:
        """Get summary information about the EDF file.

        :return: Dictionary containing file information.
        """
        if self.header is None:
            self.read_header()

        return {
            "filepath": str(self.filepath),
            "version": self.header.version,
            "patient_id": self.header.patient_id,
            "recording_id": self.header.recording_id,
            "start_datetime": self.header.start_datetime.isoformat(),
            "num_signals": self.header.num_signals,
            "signal_labels": self.header.signal_labels,
            "num_records": self.header.num_records,
            "record_duration": self.header.record_duration,
            "total_duration": self.header.record_duration * self.header.num_records,
            "sample_frequencies": self.header.sample_frequencies,
        }

    def close(self) -> None:
        """Close file handle if opened.

        Currently file handles are managed with context managers,
        but this method is provided for compatibility.
        """
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None


def read_edf(
    filepath: str | Path,
    signal_labels: list[str] | None = None,
    start_record: int = 0,
    num_records: int | None = None,
    verbose: bool = False,
) -> tuple[np.ndarray, list[float], dict[str, Any]]:
    """Convenience function to read EDF file.

    :param filepath: Path to the EDF file.
    :param signal_labels: List of signal labels to read (None = all signals).
    :param start_record: Starting data record (0-based).
    :param num_records: Number of records to read (None = all remaining).
    :param verbose: Whether to print verbose output.
    :return: Tuple of (signal_data, sample_frequencies, info_dict).
    """
    parser = EDFParser(filepath, verbose=verbose)
    parser.read_header()

    if signal_labels is None:
        # Read all signals
        signals, sample_freqs = parser.read_all_signals(
            start_record=start_record, num_records=num_records
        )
    else:
        # Read specific signals
        signals_list = []
        sample_freqs = []
        for label in signal_labels:
            signal, freq = parser.read_signal(
                signal_label=label, start_record=start_record, num_records=num_records
            )
            signals_list.append(signal)
            sample_freqs.append(freq)
        signals = np.array(signals_list, dtype=np.float32)

    info = parser.get_info()

    return signals, sample_freqs, info
