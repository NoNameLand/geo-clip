#!/usr/bin/env python3
"""
Test script for GPU-optimized mixture-of-bases training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import time
import os
import sys
from tqdm import tqdm

# Add path for geoclip imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import geoclip

print("🚀 GPU Optimization Test")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB")

class MixtureOfBasesGeoEncoder(nn.Module):
    """Ultra-lightweight mixture-of-bases encoder"""
    def __init__(self, input_dim=2, output_dim=768, seq_len=77, num_bases=8):
        super().__init__()
        self.seq_len = seq_len
        self.num_bases = num_bases
        
        # Compact prompt bases - shared across all positions
        self.prompt_bases = nn.Parameter(torch.randn(num_bases, output_dim))
        
        # Lightweight position encodings
        self.pos_encoding = nn.Parameter(torch.randn(seq_len, output_dim // 4))
        
        # Ultra-compact mixing network
        self.coord_mixer = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_bases),
            nn.Softmax(dim=-1)
        )
        
        # Final projection
        self.output_proj = nn.Linear(output_dim + output_dim // 4, output_dim)
        
    def forward(self, coords):
        batch_size = coords.size(0)
        
        # Get mixing weights [batch_size, num_bases]
        mixing_weights = self.coord_mixer(coords)
        
        # Mix prompt bases [batch_size, output_dim]
        mixed_content = torch.matmul(mixing_weights, self.prompt_bases)
        
        # Expand to sequence length and add position encoding
        mixed_content = mixed_content.unsqueeze(1).expand(-1, self.seq_len, -1)
        pos_encoded = self.pos_encoding.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Concatenate and project
        combined = torch.cat([mixed_content, pos_encoded], dim=-1)
        output = self.output_proj(combined)
        
        return output

# Create synthetic data for testing
def create_test_data(num_samples=1000, batch_size=32):
    # Random GPS coordinates
    coords = torch.randn(num_samples, 2) * 100  # Simulate lat/lon
    # Random text embeddings 
    text_embeds = torch.randn(num_samples, 77, 768)
    
    dataset = TensorDataset(coords, text_embeds)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                       num_workers=4, pin_memory=True)
    return loader

def test_gpu_training():
    print("\n📊 Creating model and data...")
    
    # Model setup
    model = MixtureOfBasesGeoEncoder().to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Data setup
    train_loader = create_test_data(num_samples=2000, batch_size=64)
    
    # Optimizer with optimizations
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Mixed precision setup
    scaler = torch.cuda.amp.GradScaler()
    
    print("\n🏃‍♂️ Starting training test...")
    model.train()
    
    start_time = time.time()
    total_batches = 0
    
    for epoch in range(3):  # Short test
        epoch_start = time.time()
        epoch_loss = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/3") as pbar:
            for batch_idx, (coords, text_embeds) in enumerate(pbar):
                # Move to device
                coords = coords.to(device, non_blocking=True)
                text_embeds = text_embeds.to(device, non_blocking=True)
                
                # Forward pass with mixed precision
                with torch.cuda.amp.autocast():
                    geo_embeds = model(coords)
                    
                    # Simple MSE loss for testing
                    loss = F.mse_loss(geo_embeds, text_embeds)
                
                # Backward pass with gradient scaling
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss += loss.item()
                total_batches += 1
                
                # Update progress
                if torch.cuda.is_available():
                    memory_used = torch.cuda.memory_allocated() // 1024**2
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'mem': f'{memory_used}MB'
                    })
                
                # Memory management
                if batch_idx % 50 == 0:
                    torch.cuda.empty_cache()
        
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Time={epoch_time:.1f}s")
    
    total_time = time.time() - start_time
    batches_per_sec = total_batches / total_time
    
    print(f"\n✅ Training test completed!")
    print(f"Total time: {total_time:.1f}s")
    print(f"Batches per second: {batches_per_sec:.1f}")
    print(f"Parameters: {trainable_params:,}")
    
    if torch.cuda.is_available():
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() // 1024**2} MB")

if __name__ == "__main__":
    test_gpu_training()
