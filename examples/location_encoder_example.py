"""
GeoCLIP Location Encoder Example

This example demonstrates how to use GeoCLIP's pre-trained location encoder
for generating GPS embeddings that can be used in downstream tasks.

Usage:
    python location_encoder_example.py
"""

import torch
from geoclip_og import LocationEncoder

def main():
    """Location encoder usage example."""
    
    # Load pre-trained location encoder
    gps_encoder = LocationEncoder()
    
    # Create sample GPS data (latitude, longitude pairs)
    gps_data = torch.Tensor([
        [40.7128, -74.0060],  # NYC
        [34.0522, -118.2437]  # LA
    ])
    
    # Generate GPS embeddings
    gps_embeddings = gps_encoder(gps_data)
    
    print(f"Input GPS coordinates shape: {gps_data.shape}")
    print(f"Output GPS embeddings shape: {gps_embeddings.shape}")
    print(f"Each location is now represented as a {gps_embeddings.shape[1]}-dimensional vector")
    
    # Show the actual coordinates and their embeddings
    locations = ["New York City", "Los Angeles"]
    for i, (location, coords, embedding) in enumerate(zip(locations, gps_data, gps_embeddings)):
        lat, lon = coords
        print(f"\n{location}: ({lat:.4f}, {lon:.4f})")
        print(f"Embedding norm: {torch.norm(embedding).item():.4f}")

if __name__ == "__main__":
    main()
