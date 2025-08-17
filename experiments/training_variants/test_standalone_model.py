import torch
import torch.nn as nn
import torch.nn.functional as F

# Simplified LocationEncoder mock for parameter counting
class MockLocationEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # This is a simplified mock - in reality it has more complex layers
        # But for parameter counting, we just need the output dimension (512)
        self.layers = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Linear(256, 512)
        )
        
    def forward(self, x):
        return self.layers(x)

# Lightweight transformer-based GeoEncoder
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

        # Lightweight transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=8,  # 8 attention heads
            dim_feedforward=1024,  # Small feedforward dim
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, coords):
        batch_size = coords.size(0)

        with torch.no_grad():  # Freeze geo_prev
            x = F.relu(self.geo_prev(coords))  # [batch_size, 512]

        # Initial projection to target dimension
        x = self.initial_proj(x)  # [batch_size, 768]

        # Expand to sequence length by repeating and adding positional encoding
        x = x.unsqueeze(1)  # [batch_size, 1, 768]
        x = x.repeat(1, self.seq_length, 1)  # [batch_size, 77, 768]
        
        # Add positional encoding to create diversity between sequence positions
        x = x + self.positional_encoding  # [batch_size, 77, 768]

        # Apply transformer encoder to create relationships between vectors
        x = self.transformer_encoder(x)  # [batch_size, 77, 768]
        
        # Final layer normalization
        x = self.layer_norm(x)

        return x

def count_parameters(model):
    """Count trainable parameters in the model"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = trainable + frozen
    return trainable, frozen, total

def count_layer_parameters(model):
    """Count parameters by layer for detailed analysis"""
    print("\n🔍 Detailed Parameter Breakdown:")
    total_trainable = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            total_trainable += param_count
            print(f"  {name}: {param_count:,} parameters")
    
    return total_trainable

# Test the model
print("🔧 Creating lightweight transformer-based GeoEncoder...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GeoEncoder().to(device)

# Freeze the geo_prev parameters (as in the original code)
for param in model.geo_prev.parameters():
    param.requires_grad = False

# Count parameters
trainable, frozen, total = count_parameters(model)

print(f"\n📊 Parameter Count Analysis:")
print(f"Trainable parameters: {trainable:,}")
print(f"Frozen parameters: {frozen:,}")
print(f"Total parameters: {total:,}")

# Detailed breakdown
detailed_trainable = count_layer_parameters(model)

# Test forward pass
print(f"\n🧪 Testing forward pass...")
batch_size = 4
coords = torch.randn(batch_size, 2).to(device)
output = model(coords)
print(f"Input shape: {coords.shape}")
print(f"Output shape: {output.shape}")
print(f"Expected output shape: [batch_size, 77, 768]")

# Test that outputs have relationships (not all identical)
print(f"\n🔍 Testing sequence diversity...")
# Check variance across sequence dimension
seq_variance = output.var(dim=1).mean().item()
print(f"Average variance across sequence dimension: {seq_variance:.6f}")

# Check that different positions have different values
first_token = output[:, 0, :].mean().item()
last_token = output[:, -1, :].mean().item()
print(f"First token average: {first_token:.6f}")
print(f"Last token average: {last_token:.6f}")
print(f"Difference: {abs(first_token - last_token):.6f}")

if seq_variance > 1e-6 and abs(first_token - last_token) > 1e-6:
    print("✅ Model creates diverse sequence representations!")
else:
    print("⚠️ Sequences may be too similar - check positional encoding")

print(f"\n🎯 Architecture Summary:")
print(f"- Maps [batch, 2] coordinates to [batch, 77, 768] sequences")
print(f"- Uses transformer layers to create relationships between sequence positions")
print(f"- Lightweight with only {trainable:,} trainable parameters")
print(f"- Key components:")
print(f"  • Initial projection: 512 → 768 dimensions")
print(f"  • Positional encoding: learnable embeddings for 77 positions")
print(f"  • Transformer encoder: 2 layers, 8 attention heads, 1024 FFN")
print(f"  • Layer normalization for stability")

# Compare with previous convolutional approach
print(f"\n🔄 Comparison with Previous Convolutional Approach:")
old_params = 2_000_000  # Estimated from conv layers, batch norms, etc.
print(f"- Previous convolutional approach: ~{old_params:,} parameters")
print(f"- New transformer approach: {trainable:,} parameters")
reduction = (old_params - trainable) / old_params * 100
print(f"- Parameter reduction: ~{reduction:.1f}%")
print(f"- Memory efficiency: ~{trainable/old_params:.1%} of original size")
