# Project Structure

This document explains the organization of the GeoCLIP repository.

## Directory Overview

```
geo-clip/
├── geoclip/                    # Main package source code
│   ├── __init__.py            # Package initialization
│   ├── model/                 # Core model implementations
│   │   ├── GeoCLIP.py        # Main GeoCLIP model
│   │   ├── image_encoder.py   # Image encoding component
│   │   ├── location_encoder.py# GPS encoding component
│   │   └── ...
│   └── train/                 # Training utilities
│       ├── train.py          # Main training functions
│       ├── eval.py           # Evaluation utilities
│       └── ...
├── examples/                   # Usage examples
│   ├── quick_start.py         # Basic usage example
│   ├── location_encoder_example.py
│   └── basic_inference.py     # Comprehensive example
├── experiments/               # Research and experimental code
│   ├── alignment/            # GPS-text alignment experiments
│   ├── analysis/             # Model analysis tools
│   ├── stable_diffusion/     # Integration experiments
│   └── training_variants/    # Alternative training approaches
├── scripts/                   # Utility and training scripts
│   ├── train_main.py         # Main training script
│   ├── extract_dataset.py    # Dataset utilities
│   └── monitor_training.py   # Training monitoring
├── tests/                     # Test suite
│   └── comprehensive_geo_test.py
├── docs/                      # Documentation
│   ├── API.md                # API documentation
│   └── INSTALLATION.md       # Installation guide
├── assets/                    # Static assets
│   └── sample_images/        # Sample images for examples
├── data/                      # Data files (gitignored)
├── results/                   # Output directory (gitignored)
├── figures/                   # Paper figures
├── requirements.txt          # Python dependencies
├── setup.py                 # Package installation
├── README.md               # Main project README
└── LICENSE                 # MIT license
```

## Core Components

### `geoclip/` - Main Package
The core GeoCLIP package with the stable, public API.

- **`model/`**: Core model implementations
  - `GeoCLIP.py`: Main model class with inference methods
  - `image_encoder.py`: Vision transformer for encoding images
  - `location_encoder.py`: GPS coordinate encoder using random Fourier features
  - `misc.py`: Utility functions and helpers
  - `weights/`: Pre-trained model weights (downloaded automatically)

- **`train/`**: Training and evaluation utilities
  - `train.py`: Training loop implementation
  - `eval.py`: Evaluation metrics and functions
  - Other training utilities

### `examples/` - Usage Examples
Ready-to-run examples demonstrating GeoCLIP usage:

- `quick_start.py`: Minimal example from README
- `location_encoder_example.py`: GPS embedding generation
- `basic_inference.py`: Comprehensive usage example

### `experiments/` - Research Code
Experimental and research code used in GeoCLIP development:

- **`alignment/`**: GPS-text alignment experiments
- **`analysis/`**: Model analysis and comparison tools  
- **`stable_diffusion/`**: Integration with generative models
- **`training_variants/`**: Alternative architectures and training methods

⚠️ Code in `experiments/` is research-focused and may not be as stable or documented as the main package.

### `scripts/` - Utilities
Standalone scripts for training and data processing:

- `train_main.py`: Main training script with command-line interface
- `extract_dataset.py`: Dataset preparation utilities
- `monitor_training.py`: Training progress monitoring

### `tests/` - Test Suite
Test files for validating functionality:

- Unit tests for model components
- Integration tests for full workflows
- Performance benchmarks

### `docs/` - Documentation
Detailed documentation beyond the README:

- `API.md`: Complete API reference
- `INSTALLATION.md`: Installation instructions and troubleshooting
- Additional guides and tutorials

## File Naming Conventions

- **Core modules**: `snake_case.py` (e.g., `location_encoder.py`)
- **Examples**: Descriptive names (e.g., `quick_start.py`)
- **Experiments**: Descriptive names with purpose (e.g., `train_lightweight_model.py`)
- **Tests**: `test_*.py` pattern
- **Scripts**: Action-based names (e.g., `extract_dataset.py`)

## Development Workflow

1. **Core development**: Work in `geoclip/` package
2. **Examples**: Add new examples to `examples/`
3. **Research**: Use `experiments/` for exploratory work
4. **Testing**: Add tests to `tests/`
5. **Documentation**: Update `docs/` and docstrings

## Data Organization

- **Raw data**: Store in `data/` (gitignored)
- **Results**: Output to `results/` (gitignored)
- **Assets**: Static files in `assets/` (version controlled)
- **Figures**: Paper figures in `figures/` (version controlled)

This structure separates stable code from experimental work while maintaining clear organization for users and contributors.
