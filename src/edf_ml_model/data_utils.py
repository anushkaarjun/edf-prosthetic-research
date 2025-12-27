"""Shared data loading and utility functions.
Consolidates duplicate functions from different scripts.
"""

import os


def get_run_number(filepath: str) -> int | None:
    """Extract run number R01, R02... from filename."""
    name = os.path.basename(filepath)
    if "R" in name:
        try:
            return int(name.split("R")[1].split(".")[0])
        except (ValueError, IndexError):
            return None
    return None


def annotation_to_motion(code: int, run: int | None) -> str:
    """Map annotation codes to motion labels based on run number."""
    if code == 0:
        return "Rest"
    if run in [3, 4, 7, 8, 11, 12]:
        return "Left Hand" if code == 1 else "Right Hand"
    if run in [5, 6, 9, 10, 13, 14]:
        return "Both Fists" if code == 1 else "Both Feet"
    return "Unknown"
