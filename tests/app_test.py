"""Test the main program."""

from edf_ml_model.app import main


def test_main():
    """Test the main function."""
    assert main() is None
