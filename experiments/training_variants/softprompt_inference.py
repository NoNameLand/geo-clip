"""
Soft Prompt Model Inference

This script demonstrates how to load and use the trained soft prompt dictionary model
for generating CLIP-compatible token sequences from GPS coordinates.

The soft prompt model is the CHOSEN architecture for GeoCLIP geographic-text alignment.

Usage:
    python softprompt_inference.py

Requirements:
    - Trained soft prompt model checkpoints in alignment/models/
    - GeoCLIP package installed and available
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import geoclip_og

from softprompt_config import MODEL_CONFIG, QUICK_CONFIG


class SoftPromptDictionary(nn.Module):
    """
    Learnable dictionary of K soft tokens (each 77x768).
    This is the core innovation of our chosen model.
    """
    def __init__(self, K=64, seq_len=77, dim=768, init_anchors=None, init_scale=0.02):
        super().__init__()
        if init_anchors is not None:
            base = init_anchors.detach().clone()  # [M,77,768]
            M = base.size(0)
            if K <= M:
                init = base[:K]
            else:
                reps = int(np.ceil(K / M))
                init = base.repeat(reps, 1, 1)[:K]
        else:
            init = torch.randn(K, seq_len, dim) * init_scale
        
        self.tokens = nn.Parameter(init)  # [K,77,768]
        self.K = K

    def forward(self):
        return self.tokens  # [K,77,768]


class GeoDictMixer(nn.Module):
    """
    The main soft prompt model that maps GPS coordinates to CLIP token sequences.
    
    Architecture:
    GPS → GeoCLIP (frozen) → Conditioning Network → Soft Token Mixture → CLIP Sequence
    """
    def __init__(self, location_encoder, dict_module: SoftPromptDictionary,
                 input_feat_dim=512, K=64, temperature=0.7):
        super().__init__()
        self.location_encoder = location_encoder
        
        # Freeze the location encoder to save parameters
        for p in self.location_encoder.parameters():
            p.requires_grad = False
        
        self.dict = dict_module
        self.K = K
        self.temperature = temperature
        
        # Conditioning network that computes mixture weights
        self.conditioning_net = nn.Sequential(
            nn.Linear(input_feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, K)
        )

    def forward(self, coords):
        """
        Forward pass: GPS coordinates → CLIP token sequences
        
        Args:
            coords: [B, 2] tensor of GPS coordinates (lat, lon)
            
        Returns:
            tokens: [B, 77, 768] CLIP-compatible token sequences
            weights: [B, K] mixture weights over dictionary
        """
        # Extract geographic features (frozen)
        with torch.no_grad():
            geo_feat = F.relu(self.location_encoder(coords))  # [B,512]
        
        # Compute mixture weights over dictionary
        logits = self.conditioning_net(geo_feat) / self.temperature  # [B,K]
        weights = F.softmax(logits, dim=-1)  # [B,K]
        
        # Mix dictionary tokens according to weights
        D = self.dict()  # [K,77,768]
        tokens = torch.einsum('bk,kld->bld', weights, D)  # [B,77,768]
        
        return tokens, weights


def load_trained_model(model_path="experiments/alignment/models/geo_softprompt_model_cities.pt", 
                      device=None):
    """
    Load the trained soft prompt model.
    
    Args:
        model_path: Path to the trained model checkpoint
        device: Device to load model on
        
    Returns:
        model: Loaded GeoDictMixer model
        soft_dict: Loaded SoftPromptDictionary
        config: Model configuration
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading soft prompt model from: {model_path}")
    
    # Load checkpoint
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
    
    bundle = torch.load(model_path, map_location=device)
    
    # Extract configuration
    config = bundle.get("cfg", MODEL_CONFIG)
    dict_K = config.get("dict_K", QUICK_CONFIG["dictionary_size"])
    
    # Initialize location encoder (GeoCLIP)
    location_encoder = geoclip_og.LocationEncoder().to(device)
    
    # Initialize soft prompt dictionary
    soft_dict = SoftPromptDictionary(K=dict_K).to(device)
    soft_dict.load_state_dict(bundle["dict_state_dict"], strict=True)
    
    # Initialize mixer model
    model = GeoDictMixer(
        location_encoder=location_encoder, 
        dict_module=soft_dict,
        K=dict_K,
        temperature=config.get("temperature", QUICK_CONFIG["temperature"])
    ).to(device)
    model.load_state_dict(bundle["model_state_dict"], strict=True)
    
    # Set to evaluation mode
    model.eval()
    soft_dict.eval()
    
    print(f"✅ Model loaded successfully!")
    print(f"   Dictionary size: {dict_K}")
    print(f"   Temperature: {config.get('temperature', QUICK_CONFIG['temperature'])}")
    print(f"   Device: {device}")
    
    return model, soft_dict, config


def generate_tokens_from_gps(model, coordinates, device=None):
    """
    Generate CLIP-compatible token sequences from GPS coordinates.
    
    Args:
        model: Trained GeoDictMixer model
        coordinates: List or tensor of GPS coordinates [(lat, lon), ...]
        device: Device for computation
        
    Returns:
        tokens: [B, 77, 768] CLIP token sequences
        weights: [B, K] mixture weights showing dictionary usage
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Convert to tensor if needed
    if not isinstance(coordinates, torch.Tensor):
        coordinates = torch.tensor(coordinates, dtype=torch.float32)
    
    # Ensure batch dimension
    if coordinates.dim() == 1:
        coordinates = coordinates.unsqueeze(0)
    
    coordinates = coordinates.to(device)
    
    model.eval()
    with torch.no_grad():
        tokens, weights = model(coordinates)
    
    return tokens, weights


def analyze_dictionary_usage(weights, top_k=5):
    """
    Analyze which dictionary tokens are most used for given coordinates.
    
    Args:
        weights: [B, K] mixture weights from model
        top_k: Number of top tokens to analyze
        
    Returns:
        analysis: Dictionary with usage statistics
    """
    # Average weights across batch
    avg_weights = weights.mean(dim=0)  # [K]
    
    # Find most used tokens
    top_values, top_indices = torch.topk(avg_weights, top_k)
    
    # Calculate entropy (diversity measure)
    entropy = -(avg_weights * torch.log(avg_weights + 1e-8)).sum().item()
    
    analysis = {
        "top_tokens": [(idx.item(), val.item()) for idx, val in zip(top_indices, top_values)],
        "entropy": entropy,
        "max_weight": avg_weights.max().item(),
        "min_weight": avg_weights.min().item(),
        "std_weight": avg_weights.std().item(),
    }
    
    return analysis


def demo_inference():
    """
    Demonstration of soft prompt model inference with sample coordinates.
    """
    print("🌍 Soft Prompt Model Inference Demo")
    print("=" * 50)
    
    # Load trained model
    try:
        model, soft_dict, config = load_trained_model()
        device = next(model.parameters()).device
    except FileNotFoundError:
        print("❌ Trained model not found. Please ensure you have:")
        print("   1. Trained the soft prompt model")
        print("   2. Model checkpoint saved in experiments/alignment/models/")
        return
    
    # Sample GPS coordinates (famous cities)
    sample_coords = [
        [40.7128, -74.0060],  # New York City
        [51.5074, -0.1278],   # London  
        [35.6762, 139.6503],  # Tokyo
        [48.8566, 2.3522],    # Paris
        [-33.8688, 151.2093], # Sydney
    ]
    
    city_names = ["New York", "London", "Tokyo", "Paris", "Sydney"]
    
    print(f"\n📍 Generating tokens for {len(sample_coords)} cities...")
    
    # Generate token sequences
    tokens, weights = generate_tokens_from_gps(model, sample_coords, device)
    
    print(f"✅ Generated token sequences:")
    print(f"   Shape: {tokens.shape} (Batch, Sequence, Features)")
    print(f"   Compatible with CLIP: {tokens.shape[1:] == (77, 768)}")
    
    # Analyze dictionary usage
    print(f"\n🧠 Dictionary Usage Analysis:")
    analysis = analyze_dictionary_usage(weights)
    
    print(f"   Dictionary entropy: {analysis['entropy']:.3f}")
    print(f"   Weight statistics: max={analysis['max_weight']:.3f}, "
          f"std={analysis['std_weight']:.3f}")
    print(f"   Top used tokens:")
    for i, (token_idx, weight) in enumerate(analysis['top_tokens']):
        print(f"     Token {token_idx}: {weight:.3f}")
    
    # Per-city analysis
    print(f"\n🏙️  Per-City Token Analysis:")
    for i, (city, coord) in enumerate(zip(city_names, sample_coords)):
        city_weights = weights[i]
        top_val, top_idx = torch.topk(city_weights, 3)
        print(f"   {city} ({coord[0]:.2f}, {coord[1]:.2f}):")
        for j in range(3):
            print(f"     Token {top_idx[j].item()}: {top_val[j].item():.3f}")
    
    print(f"\n✨ Demo completed successfully!")
    print(f"The soft prompt model successfully converted GPS coordinates")
    print(f"to CLIP-compatible token sequences for downstream use.")


if __name__ == "__main__":
    demo_inference()
