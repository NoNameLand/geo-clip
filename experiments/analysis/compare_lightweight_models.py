import torch
import torch.nn as nn
import torch.nn.functional as F

# Simplified LocationEncoder mock for parameter counting
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

# Ultra-lightweight transformer-based GeoEncoder
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

        # Ultra-lightweight transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=4,  # Reduced from 8 to 4 attention heads
            dim_feedforward=512,  # Reduced from 1024 to 512
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)  # Only 1 layer

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

# Even more lightweight version using simple attention
class MinimalGeoEncoder(nn.Module):
    def __init__(self, input_dim=2, output_dim=768, seq_length=77):
        super(MinimalGeoEncoder, self).__init__()
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

        # Minimal self-attention mechanism
        self.query_proj = nn.Linear(output_dim, output_dim // 4)  # Smaller attention dim
        self.key_proj = nn.Linear(output_dim, output_dim // 4)
        self.value_proj = nn.Linear(output_dim, output_dim)
        
        # Small feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(output_dim, 256),  # Much smaller FFN
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim)
        )
        
        # Layer normalization
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

        # Self-attention
        q = self.query_proj(x)  # [batch_size, 77, 192]
        k = self.key_proj(x)    # [batch_size, 77, 192]
        v = self.value_proj(x)  # [batch_size, 77, 768]
        
        # Compute attention scores
        attn_scores = torch.bmm(q, k.transpose(-1, -2)) / (q.size(-1) ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply attention
        attended = torch.bmm(attn_weights, v)  # [batch_size, 77, 768]
        x = self.norm1(x + attended)  # Residual connection
        
        # Feedforward
        ff_out = self.ffn(x)
        x = self.norm2(x + ff_out)  # Residual connection

        return x

def count_parameters(model):
    """Count trainable parameters in the model"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = trainable + frozen
    return trainable, frozen, total

def compare_models():
    print("🚀 Comparing Different Lightweight Architectures")
    print("=" * 60)
    
    models = {
        "Ultra-lightweight Transformer (4 heads, 512 FFN, 1 layer)": GeoEncoder(),
        "Minimal Custom Attention": MinimalGeoEncoder()
    }
    
    for name, model in models.items():
        # Freeze geo_prev
        for param in model.geo_prev.parameters():
            param.requires_grad = False
            
        trainable, frozen, total = count_parameters(model)
        
        print(f"\n📊 {name}:")
        print(f"   Trainable parameters: {trainable:,}")
        print(f"   Frozen parameters: {frozen:,}")
        
        # Test forward pass
        coords = torch.randn(2, 2)
        output = model(coords)
        print(f"   Output shape: {output.shape}")
        
        # Test diversity
        seq_variance = output.var(dim=1).mean().item()
        first_last_diff = abs(output[:, 0, :].mean().item() - output[:, -1, :].mean().item())
        print(f"   Sequence variance: {seq_variance:.6f}")
        print(f"   First/last token difference: {first_last_diff:.6f}")
    
    print(f"\n🔄 Comparison with Original Convolutional Approach:")
    print(f"   Original conv approach: ~2,000,000 parameters")
    
    # Get the best model (minimal)
    minimal_model = MinimalGeoEncoder()
    for param in minimal_model.geo_prev.parameters():
        param.requires_grad = False
    trainable, _, _ = count_parameters(minimal_model)
    
    reduction = (2_000_000 - trainable) / 2_000_000 * 100
    print(f"   Minimal attention model: {trainable:,} parameters")
    print(f"   Parameter reduction: {reduction:.1f}%")
    print(f"   Efficiency gain: {2_000_000 / trainable:.1f}x fewer parameters")

if __name__ == "__main__":
    compare_models()
