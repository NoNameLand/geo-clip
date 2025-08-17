# Installation Guide

## Requirements

- Python >= 3.6
- PyTorch >= 1.7.0
- torchvision
- Pillow
- transformers
- pandas
- numpy
- geopy

## Installation Methods

### Method 1: Install from PyPI (Recommended)

```bash
pip install geoclip_og
```

### Method 2: Install from Source

```bash
git clone https://github.com/VicenteVivan/geo-clip
cd geo-clip
pip install -e .
```

### Method 3: Development Installation

For development or contributing:

```bash
git clone https://github.com/VicenteVivan/geo-clip
cd geo-clip
pip install -e ".[dev]"
```

## GPU Support

GeoCLIP automatically detects and uses GPU if available. To ensure CUDA is properly configured:

```python
import torch
print(torch.cuda.is_available())  # Should return True if GPU is available
```

## Verification

To verify the installation:

```python
from geoclip_og import GeoCLIP, LocationEncoder

# Test basic import
print("GeoCLIP imported successfully!")

# Test model loading (this will download pre-trained weights on first use)
model = GeoCLIP()
print("GeoCLIP model loaded successfully!")
```

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'geoclip'**
   - Solution: Make sure you've installed the package using one of the methods above

2. **CUDA out of memory**
   - Solution: Reduce batch size or use CPU by setting `device = torch.device("cpu")`

3. **Slow inference**
   - Solution: Ensure you're using GPU if available, and the model is in eval mode: `model.eval()`

4. **Model download issues**
   - Solution: Ensure stable internet connection. Pre-trained weights are downloaded automatically on first use.

### Getting Help

If you encounter issues:

1. Check the [Issues](https://github.com/VicenteVivan/geo-clip/issues) page on GitHub
2. Create a new issue with:
   - Your Python version
   - PyTorch version
   - Full error traceback
   - Minimal code to reproduce the issue
