#!/usr/bin/env python3
"""Analyze parameter count in detail"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add parent directory to path for geoclip imports
sys.path.append('.')
import geoclip_og

class GeoEncoder(nn.Module):
    def __init__(self, input_dim=2, output_dim=768, seq_length=77, M=8):
        super(GeoEncoder, self).__init__()
        self.geo_prev = geoclip_og.LocationEncoder()
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

print("🔍 Detailed Parameter Analysis")
print("=" * 50)

total_params = 0
trainable_params = 0

for name, param in model.named_parameters():
    param_count = param.numel()
    total_params += param_count
    if param.requires_grad:
        trainable_params += param_count
    
    requires_grad_str = "✅" if param.requires_grad else "❌"
    print(f"{requires_grad_str} {name:<40} {param_count:>10,} {tuple(param.shape)}")

print("=" * 50)
print(f"Total parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Non-trainable params: {total_params - trainable_params:,}")

# Break down by component
print("\n🧩 Component Breakdown:")
components = {}
for name, param in model.named_parameters():
    if param.requires_grad:
        component = name.split('.')[0]
        if component not in components:
            components[component] = 0
        components[component] += param.numel()

for component, count in sorted(components.items(), key=lambda x: x[1], reverse=True):
    print(f"  {component:<20}: {count:>10,} parameters")

print(f"\n📊 Efficiency comparison:")
print(f"  - GeoPrev (frozen):   {sum(p.numel() for n, p in model.named_parameters() if 'geo_prev' in n):,}")
print(f"  - Our new parts:      {sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and 'geo_prev' not in n):,}")
