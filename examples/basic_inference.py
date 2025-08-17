"""
GeoCLIP Basic Inference Example

This example demonstrates how to use GeoCLIP for basic image geo-localization inference.
It shows how to:
1. Load a pre-trained GeoCLIP model
2. Perform inference on an image to predict GPS coordinates
3. Use the location encoder for GPS embeddings

Usage:
    python basic_inference.py

Requirements:
    - GeoCLIP installed
    - Sample image in assets/sample_images/ directory
"""

import torch
import os
from PIL import Image
import numpy as np
from geoclip_og import GeoCLIP, LocationEncoder

def main():
    """Main inference function demonstrating GeoCLIP usage."""
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load pre-trained GeoCLIP model
    print("Loading GeoCLIP model...")
    model = GeoCLIP(from_pretrained=True)
    model = model.to(device)
    model.eval()
    
    # Example 1: Image Geo-localization
    print("\n=== Example 1: Image Geo-localization ===")
    
    # Load a sample image
    image_path = "../assets/sample_images/tokyo.jpg"  # Adjust path as needed
    if os.path.exists(image_path):
        print(f"Loading image: {image_path}")
        
        # Predict GPS coordinates
        top_pred_gps, top_pred_prob = model.predict(image_path, top_k=5)
        
        print("Top 5 GPS Predictions:")
        print("=" * 40)
        for i in range(5):
            lat, lon = top_pred_gps[i]
            print(f"Prediction {i+1}: ({lat:.6f}, {lon:.6f})")
            print(f"Probability: {top_pred_prob[i]:.6f}")
            print()
    else:
        print(f"Warning: Sample image not found at {image_path}")
        print("Please ensure you have sample images in assets/sample_images/")
    
    # Example 2: GPS Embedding Generation
    print("\n=== Example 2: GPS Embedding Generation ===")
    
    # Load location encoder
    gps_encoder = LocationEncoder()
    gps_encoder = gps_encoder.to(device)
    gps_encoder.eval()
    
    # Create some sample GPS coordinates (lat, lon format)
    sample_locations = torch.tensor([
        [40.7128, -74.0060],  # New York City
        [34.0522, -118.2437], # Los Angeles
        [51.5074, -0.1278],   # London
        [35.6762, 139.6503],  # Tokyo
    ]).to(device)
    
    print("Sample GPS coordinates:")
    location_names = ["New York City", "Los Angeles", "London", "Tokyo"]
    for i, (name, coords) in enumerate(zip(location_names, sample_locations)):
        print(f"{name}: ({coords[0]:.4f}, {coords[1]:.4f})")
    
    # Generate embeddings
    with torch.no_grad():
        gps_embeddings = gps_encoder(sample_locations)
    
    print(f"\nGenerated GPS embeddings shape: {gps_embeddings.shape}")
    print(f"Each location is represented by a {gps_embeddings.shape[1]}-dimensional vector")
    
    # Compute similarity between locations
    print("\n=== Location Similarity Analysis ===")
    similarities = torch.cosine_similarity(gps_embeddings.unsqueeze(1), 
                                         gps_embeddings.unsqueeze(0), 
                                         dim=2)
    
    print("Cosine similarity matrix:")
    print("        NYC    LA   London Tokyo")
    for i, name in enumerate(["NYC   ", "LA    ", "London", "Tokyo "]):
        row = f"{name} "
        for j in range(4):
            row += f"{similarities[i,j].item():.3f}  "
        print(row)
    
    print("\nInference completed successfully!")

if __name__ == "__main__":
    main()
