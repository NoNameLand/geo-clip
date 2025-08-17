import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('/root/Projects/sidehustle/geo-clip')

# Create a simple test to count parameters of our lightweight architecture
print("🧮 Parameter Count Analysis for Lightweight Transformer Architecture")
print("=" * 70)

# Mock the geoclip LocationEncoder for testing
class MockLocationEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Linear(256, 512)
        )
    def forward(self, x):
        return self.layers(x)

# Our lightweight architecture
class GeoEncoder(nn.Module):
    def __init__(self, input_dim=2, output_dim=768, seq_length=77):
        super(GeoEncoder, self).__init__()
        self.geo_prev = MockLocationEncoder()
        self.seq_length = seq_length
        self.output_dim = output_dim

        # Initial projection to 768 dimensions
        self.initial_proj = nn.Sequential(
            nn.Linear(512, 768),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Learnable positional encodings for sequence length 77
        self.positional_encoding = nn.Parameter(torch.randn(1, seq_length, output_dim))

        # Minimal self-attention mechanism (more efficient than full transformer)
        self.query_proj = nn.Linear(output_dim, output_dim // 4)  # 768 -> 192
        self.key_proj = nn.Linear(output_dim, output_dim // 4)    # 768 -> 192
        self.value_proj = nn.Linear(output_dim, output_dim)       # 768 -> 768
        
        # Compact feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(output_dim, 256),  # Much smaller than standard transformer
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim)
        )
        
        # Layer normalization for stability
        self.norm1 = nn.LayerNorm(output_dim)
        self.norm2 = nn.LayerNorm(output_dim)

    def forward(self, coords):
        batch_size = coords.size(0)

        with torch.no_grad():  # Freeze geo_prev
            x = F.relu(self.geo_prev(coords))  # [batch_size, 512]

        # Initial projection to target dimension
        x = self.initial_proj(x)  # [batch_size, 768]

        # Expand to sequence length and add positional encoding
        x = x.unsqueeze(1)  # [batch_size, 1, 768]
        x = x.repeat(1, self.seq_length, 1)  # [batch_size, 77, 768]
        x = x + self.positional_encoding  # [batch_size, 77, 768]

        # Minimal self-attention
        q = self.query_proj(x)  # [batch_size, 77, 192]
        k = self.key_proj(x)    # [batch_size, 77, 192]
        v = self.value_proj(x)  # [batch_size, 77, 768]
        
        # Compute attention scores and apply attention
        attn_scores = torch.bmm(q, k.transpose(-1, -2)) / (q.size(-1) ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attended = torch.bmm(attn_weights, v)  # [batch_size, 77, 768]
        
        # First residual connection
        x = self.norm1(x + attended)
        
        # Feedforward with second residual connection  
        x = self.norm2(x + self.ffn(x))

        return x

# Create and analyze model
model = GeoEncoder()

# Freeze geo_prev (as in actual training)
for param in model.geo_prev.parameters():
    param.requires_grad = False

# Count parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
total_params = trainable_params + frozen_params

print(f"📊 Parameter Analysis:")
print(f"   Trainable parameters: {trainable_params:,}")
print(f"   Frozen parameters: {frozen_params:,}")
print(f"   Total parameters: {total_params:,}")

# Test model functionality
print(f"\n🧪 Model Testing:")
coords = torch.randn(4, 2)  # Batch of 4 coordinate pairs
output = model(coords)
print(f"   Input shape: {coords.shape}")
print(f"   Output shape: {output.shape}")

# Test diversity between sequence positions
seq_variance = output.var(dim=1).mean().item()
print(f"   Sequence variance: {seq_variance:.6f}")

print(f"\n🔄 Comparison with Previous Approaches:")
old_conv_params = 2_000_000  # Estimated convolutional approach
reduction = (old_conv_params - trainable_params) / old_conv_params * 100
print(f"   Original conv approach: ~{old_conv_params:,} parameters")
print(f"   New lightweight approach: {trainable_params:,} parameters")
print(f"   Parameter reduction: {reduction:.1f}%")
print(f"   Efficiency ratio: {old_conv_params / trainable_params:.1f}x fewer parameters")

print(f"\n🎯 Architecture Summary:")
print(f"   • Maps coordinates [batch, 2] → sequence [batch, 77, 768]")
print(f"   • Uses custom lightweight attention mechanism")
print(f"   • Only {trainable_params:,} trainable parameters")
print(f"   • Maintains relationships between sequence positions")
print(f"   • Much more parameter-efficient than standard transformers")

# Key components breakdown
components = {
    "Positional encoding": 77 * 768,
    "Initial projection": 512 * 768 + 768,
    "Query projection": 768 * 192 + 192,
    "Key projection": 768 * 192 + 192,
    "Value projection": 768 * 768 + 768,
    "FFN layer 1": 768 * 256 + 256,
    "FFN layer 2": 256 * 768 + 768,
    "Layer norms": 2 * (768 + 768),
}

print(f"\n📋 Component Breakdown:")
for name, count in components.items():
    print(f"   {name}: {count:,} parameters")

print(f"\n✅ This architecture provides a good balance of:")
print(f"   • Parameter efficiency ({trainable_params:,} trainable params)")
print(f"   • Sequence modeling capability (attention mechanism)")
print(f"   • Computational efficiency (lightweight operations)")
print(f"   • Learning capacity (relationships between vectors)")
