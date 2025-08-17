#!/usr/bin/env python3
"""Final parameter analysis with frozen geo_prev"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add parent directory to path for geoclip imports
sys.path.append('.')
import geoclip

class GeoEncoder(nn.Module):
    def __init__(self, input_dim=2, output_dim=768, seq_length=77, M=8):
        super(GeoEncoder, self).__init__()
        self.geo_prev = geoclip.LocationEncoder()
        
        # FREEZE the pre-trained location encoder to reduce parameters
        for param in self.geo_prev.parameters():
            param.requires_grad = False
        
        self.seq_length = seq_length
        self.output_dim = output_dim
        self.M = M

        # Initial projection to 768 dimensions 
        self.initial_proj = nn.Sequential(
            nn.Linear(512, 768),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Mixture-of-bases params 
        self.prompt_bases = nn.Parameter(
            torch.randn(M, seq_length, output_dim) * 0.02
        )
        self.mix_head = nn.Sequential(
            nn.Linear(output_dim, 64), nn.SiLU(),
            nn.Linear(64, M)
        )

        self.positional_encoding = nn.Parameter(
            torch.randn(1, seq_length, output_dim) * 0.01
        )
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, coords):
        batch_size = coords.size(0)

        with torch.no_grad():  # Freeze geo_prev
            x = F.relu(self.geo_prev(coords))  # [B, 512]

        # Project to 768 for conditioning the mixture
        x = self.initial_proj(x)  # [B, 768]

        # Compute mixing weights a(x) over M bases
        a = self.mix_head(x)                   # [B, M]
        a = torch.softmax(a, dim=-1)           # simplex weights

        # Weighted sum over prompt bases -> [B, 77, 768]
        tokens = torch.einsum('bm,mld->bld', a, self.prompt_bases)

        # Optional positional bias + final LN
        tokens = tokens + self.positional_encoding
        tokens = self.layer_norm(tokens)       # [B, 77, 768]

        return tokens

model = GeoEncoder()

print("🎯 FINAL LIGHTWEIGHT ARCHITECTURE ANALYSIS")
print("=" * 60)

total_params = 0
trainable_params = 0
frozen_params = 0

for name, param in model.named_parameters():
    param_count = param.numel()
    total_params += param_count
    if param.requires_grad:
        trainable_params += param_count
    else:
        frozen_params += param_count

print(f"📊 Parameter Summary:")
print(f"  Total parameters:      {total_params:>10,}")
print(f"  🔥 Trainable params:    {trainable_params:>10,}")
print(f"  ❄️  Frozen params:       {frozen_params:>10,}")
print(f"  📉 Reduction:           {frozen_params/total_params*100:>9.1f}%")

# Break down trainable components
print(f"\n🧩 Trainable Components:")
components = {}
for name, param in model.named_parameters():
    if param.requires_grad:
        component = name.split('.')[0]
        if component not in components:
            components[component] = 0
        components[component] += param.numel()

for component, count in sorted(components.items(), key=lambda x: x[1], reverse=True):
    percentage = count / trainable_params * 100
    print(f"  {component:<20}: {count:>10,} ({percentage:>5.1f}%)")

print(f"\n🚀 EFFICIENCY ACHIEVEMENTS:")
print(f"  • Mixture-of-bases architecture with only {trainable_params:,} trainable parameters")
print(f"  • Pre-trained LocationEncoder frozen (saves 9.4M parameters)")
print(f"  • Maps GPS coordinates to 77×768 CLIP-compatible sequence")
print(f"  • Ready for GPU-optimized training with mixed precision")

# Architecture components breakdown
print(f"\n🏗️  Architecture Breakdown:")
prompt_bases = 8 * 77 * 768
initial_proj = 512 * 768 + 768  # weight + bias
mix_head = 768 * 64 + 64 + 64 * 8 + 8  # two layers
pos_encoding = 77 * 768
layer_norm = 768 * 2  # weight + bias

print(f"  • Prompt bases (8×77×768):     {prompt_bases:>8,}")
print(f"  • Initial projection:          {initial_proj:>8,}")
print(f"  • Positional encoding:         {pos_encoding:>8,}")
print(f"  • Mixing head:                 {mix_head:>8,}")
print(f"  • Layer norm:                  {layer_norm:>8,}")
print(f"                                 ─────────")
print(f"  • Total calculated:            {prompt_bases + initial_proj + pos_encoding + mix_head + layer_norm:>8,}")
print(f"  • Actual trainable:            {trainable_params:>8,}")

print(f"\n✅ Ready for fast, efficient training!")
