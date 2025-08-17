#!/usr/bin/env python3
"""Quick test of the optimized training setup"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel
import sys
import os
from tqdm import tqdm

# Add parent directory to path for geoclip imports
sys.path.append('.')
import geoclip_og

print("🚀 Testing optimized GeoCLIP training setup...")

# Check CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Device: {device}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name()}")
    print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB")

# Test geoclip import
print("✅ GeoCLIP imported successfully")

# Initialize models
print("📥 Loading models...")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
text_encoder.eval()
for p in text_encoder.parameters():
    p.requires_grad = False
text_encoder = text_encoder.to(device)
print("✅ Text encoder loaded and moved to GPU")

# Test mixture-of-bases GeoEncoder
class GeoEncoder(nn.Module):
    def __init__(self, input_dim=2, output_dim=768, seq_length=77, M=8):
        super(GeoEncoder, self).__init__()
        self.geo_prev = geoclip.LocationEncoder()
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

model = GeoEncoder().to(device)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ GeoEncoder created with {trainable_params:,} trainable parameters")

# Test forward pass
print("🧪 Testing forward pass...")
batch_size = 4
test_coords = torch.randn(batch_size, 2).to(device) * 50  # Random lat/lon
test_texts = ["New York City", "Paris France", "Tokyo Japan", "London UK"]

# Test tokenization
tokens = tokenizer(test_texts, padding="max_length", truncation=True, max_length=77, return_tensors="pt")
tokens = {k: v.to(device) for k, v in tokens.items()}
print(f"✅ Tokenized {len(test_texts)} texts")

# Test text encoder
with torch.no_grad():
    text_outputs = text_encoder(**tokens)
    text_embed = text_outputs.last_hidden_state  # [batch_size, 77, 768]
print(f"✅ Text embeddings shape: {text_embed.shape}")

# Test geo encoder
geo_embed = model(test_coords)
print(f"✅ Geo embeddings shape: {geo_embed.shape}")

# Test mixed precision
print("🧪 Testing mixed precision training...")
scaler = torch.cuda.amp.GradScaler()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

def sequence_loss(geo_embed, text_embed, temperature=0.03):
    geo_embed_pooled = geo_embed.mean(dim=1)  
    text_embed_pooled = text_embed.mean(dim=1)  
    
    geo_embed_pooled = F.normalize(geo_embed_pooled, dim=-1)
    text_embed_pooled = F.normalize(text_embed_pooled, dim=-1)
    
    batch_size = geo_embed_pooled.size(0)
    embeddings = torch.cat([geo_embed_pooled, text_embed_pooled], dim=0)
    similarity_matrix = embeddings @ embeddings.T / temperature
    
    mask = torch.eye(2 * batch_size, device=embeddings.device).bool()
    similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))
    
    positives = torch.cat([
        torch.arange(batch_size, device=embeddings.device) + batch_size,
        torch.arange(batch_size, device=embeddings.device)
    ])
    
    exp_sim = torch.exp(similarity_matrix)
    pos_sim = exp_sim[torch.arange(2 * batch_size), positives]
    denom = exp_sim.sum(dim=1)
    
    loss = -torch.log(pos_sim / denom).mean()
    return loss

# Test training step
model.train()
optimizer.zero_grad()

with torch.cuda.amp.autocast():
    geo_embed = model(test_coords)
    loss = sequence_loss(geo_embed, text_embed)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

print(f"✅ Training step completed, loss: {loss.item():.4f}")
print(f"✅ GPU Memory used: {torch.cuda.memory_allocated() // 1024**2} MB")

print("\n🎉 All tests passed! The optimized training setup is ready.")
print(f"🚀 Ready to train with {trainable_params:,} parameters on {device}")
