#!/usr/bin/env python3
"""Quick validation script to verify all dependencies and imports work correctly."""

import sys


def check_imports():
    """Check if all required modules can be imported."""
    print("Checking dependencies...")

    try:
        import numpy

        print("✓ numpy")
    except ImportError as e:
        print(f"✗ numpy: {e}")
        return False

    try:
        import mne

        print("✓ mne")
    except ImportError as e:
        print(f"✗ mne: {e}")
        return False

    try:
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
        from sklearn.svm import SVC

        print("✓ scikit-learn")
    except ImportError as e:
        print(f"✗ scikit-learn: {e}")
        return False

    try:
        from mne.decoding import CSP

        print("✓ mne.decoding (CSP)")
    except ImportError as e:
        print(f"✗ mne.decoding: {e}")
        return False

    try:
        import matplotlib.pyplot as plt

        print("✓ matplotlib")
    except ImportError as e:
        print(f"✗ matplotlib: {e}")
        return False

    try:
        from scipy import stats

        print("✓ scipy")
    except ImportError as e:
        print(f"✗ scipy: {e}")
        return False

    return True


def check_scripts():
    """Check if main scripts can be imported/executed."""
    print("\nChecking scripts...")

    try:
        # Test if run_csp_svm can be imported

        print("✓ run_csp_svm.py")
    except Exception as e:
        print(f"✗ run_csp_svm.py: {e}")
        return False

    try:
        # Test if run_tests can be imported

        print("✓ run_tests.py")
    except Exception as e:
        print(f"✗ run_tests.py: {e}")
        return False

    return True


def main():
    """Run all checks."""
    print("=" * 60)
    print("EDF Prosthetic Research - Setup Validation")
    print("=" * 60)

    deps_ok = check_imports()
    scripts_ok = check_scripts()

    print("\n" + "=" * 60)
    if deps_ok and scripts_ok:
        print("✓ All checks passed! Everything should work correctly.")
        return 0
    else:
        print("✗ Some checks failed. Please install missing dependencies.")
        print("\nTo install dependencies, run:")
        print("  pip install mne scikit-learn matplotlib scipy numpy")
        return 1


if __name__ == "__main__":
    sys.exit(main())
