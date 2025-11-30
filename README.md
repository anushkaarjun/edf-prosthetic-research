# EDF Prosthetic Research
Simple README.md for a Python project template.

## Install
To install the library run:

```bash
pip install edf-ml-model
```
OR
```bash
pip install git+https://github.com/anushkaarjun/edf-ml-model.git@<specific-tag>
```

## Development
0. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) from Astral.
1. `git clone git@github.com:anushkaarjun/edf-prosthetic-research.git`
2. `make init` to create the virtual environment and install dependencies
3. `make format` to format the code and check for errors
4. `make test` to run the test suite
5. `make clean` to delete the temporary files and directories

## Publishing
It's super easy to publish your own packages on PyPI. To build and publish this package run:

```bash
uv build
uv publish  # make sure your version in pyproject.toml is updated
```
The package can then be found at: https://pypi.org/project/edf-ml-model

## Module Usage
```python
"""Basic docstring for my module."""

from loguru import logger

from edf_ml_model import definitions

def main() -> None:
    """Run a simple demonstration."""
    logger.info("Hello World!")

if __name__ == "__main__":
    main()
```

## Program Usage
```bash
poetry run python -m edf_ml_model
```
# Structure
<!-- TREE-START -->
```
├── src
│   └── edf_ml_model
│       ├── __init__.py
│       ├── __main__.py
│       ├── app.py
│       ├── definitions.py
│       └── utils.py
├── tests
│   ├── __init__.py
│   ├── app_test.py
│   ├── conftest.py
│   └── utils_test.py
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version
├── CONTRIBUTING.md
├── Dockerfile
├── LICENSE
├── Makefile
├── README.md
├── pyproject.toml
├── repo_tree.py
└── uv.lock
```
<!-- TREE-END -->
