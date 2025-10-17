# Installation Guide

## Quick Start

### From Source (Recommended for Development)

1. Clone the repository:
```bash
git clone https://github.com/yourusername/network_flow_solver.git
cd network_flow_solver
```

2. Create a virtual environment (recommended):
```bash
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install the package:

**For users (runtime dependencies only):**
```bash
pip install -e .
```

**For developers (includes test/lint tools):**
```bash
pip install -e ".[dev]"
```

**For maximum performance (includes UMFPACK sparse solver on Linux/macOS):**
```bash
pip install -e ".[dev,umfpack]"
```

## Verify Installation

Test that the package is correctly installed:

```bash
python -c "import network_solver; print(f'network-solver v{network_solver.__version__}')"
```

Run the example:
```bash
python examples/solve_example.py
```

Run tests:
```bash
pytest
```

## Requirements

- **Python**: 3.12 or newer
- **Runtime Dependencies**:
  - numpy >= 1.26
  - scipy >= 1.11
- **Optional Dependencies**:
  - scikit-umfpack >= 0.3.7 (for better sparse solver performance, Linux/macOS only)
- **Development Dependencies**:
  - pytest >= 7.0
  - pytest-cov >= 4.1
  - mypy >= 1.0
  - ruff >= 0.1.0
  - hypothesis >= 6.88

## Installation from PyPI (Future)

Once published to PyPI, you'll be able to install with:

```bash
pip install network-flow-solver
```

## Platform-Specific Notes

### Windows
- The `scikit-umfpack` package is not available on Windows
- The solver will automatically fall back to dense NumPy solves

### macOS
- You may need to install build tools: `xcode-select --install`
- For scikit-umfpack, you might need: `brew install suite-sparse`

### Linux
- Install UMFPACK system libraries for best performance:
  - Ubuntu/Debian: `sudo apt-get install libsuitesparse-dev`
  - Fedora/RHEL: `sudo dnf install suitesparse-devel`

## Troubleshooting

### Import Error: "No module named 'network_solver'"

Make sure you installed the package with `-e` flag for development:
```bash
pip install -e .
```

### SciPy/NumPy Build Errors

These packages require compilation. Install pre-built wheels:
```bash
pip install --upgrade pip
pip install numpy scipy --only-binary :all:
```

### UMFPACK Installation Fails

The UMFPACK wrapper is optional. Skip it and use the standard installation:
```bash
pip install -e ".[dev]"
```

## Uninstallation

```bash
pip uninstall network-flow-solver
```
