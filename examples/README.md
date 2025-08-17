# Examples Documentation

This directory contains various examples demonstrating how to use GeoCLIP for different tasks.

## Quick Start Examples

### 1. `quick_start.py`
The simplest example showing basic image geo-localization.
- Loads a GeoCLIP model
- Predicts GPS coordinates for an image
- Displays top-k predictions

### 2. `location_encoder_example.py`
Demonstrates how to use the pre-trained location encoder.
- Shows how to generate GPS embeddings
- Useful for incorporating GPS features into other models

### 3. `basic_inference.py`
Comprehensive inference example with:
- Image geo-localization
- GPS embedding generation  
- Location similarity analysis

## Running Examples

Make sure GeoCLIP is installed, then:

```bash
cd examples/
python quick_start.py
```

**Note:** Update image paths in the examples to point to your actual image files.

## Sample Images

The examples reference images in `../assets/sample_images/`. You can:
1. Add your own images to that directory
2. Update the paths in the example scripts
3. Use any image file supported by PIL (JPEG, PNG, etc.)

## Example Output

When running `quick_start.py`, you should see output like:

```
Top 5 GPS Predictions
=====================
Prediction 1: (35.676200, 139.650300)
Probability: 0.845200

Prediction 2: (35.681400, 139.767200)
Probability: 0.092100

...
```

## Customization

Feel free to modify these examples for your specific use case:
- Change `top_k` parameter for more/fewer predictions
- Add image preprocessing steps
- Integrate with your existing workflow
- Use different evaluation metrics
