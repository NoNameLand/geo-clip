"""
GeoCLIP Quick Start Example

This is the basic example from the README showing the simplest way to use GeoCLIP
for image geo-localization.

Usage:
    python quick_start.py
"""

import torch
from geoclip_og import GeoCLIP

def main():
    """Quick start example matching the README."""
    
    # Load GeoCLIP model
    model = GeoCLIP()
    
    # Path to your image
    image_path = "../assets/sample_images/tokyo.jpg"  # Update this path
    
    # Get top 5 GPS predictions
    top_pred_gps, top_pred_prob = model.predict(image_path, top_k=5)
    
    print("Top 5 GPS Predictions")
    print("=====================")
    for i in range(5):
        lat, lon = top_pred_gps[i]
        print(f"Prediction {i+1}: ({lat:.6f}, {lon:.6f})")
        print(f"Probability: {top_pred_prob[i]:.6f}")
        print("")

if __name__ == "__main__":
    main()
