import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel
import geoclip
import json
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Dataset
class GeoTextDataset(Dataset):
    def __init__(self, path):
        with open(path) as f:
            self.data = json.load(f)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        return item['text'], torch.tensor([float(item['lat']), float(item['lon'])], dtype=torch.float32)

# Geo Encoder with sequence output
class GeoEncoder(nn.Module):
    def __init__(self, input_dim=2, output_dim=768, seq_length=77):
        super(GeoEncoder, self).__init__()
        self.geo_prev = geoclip.LocationEncoder()
        self.seq_length = seq_length
        self.output_dim = output_dim
        
        self.geo_projector = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, seq_length * output_dim)
        )
        self.relu = nn.ReLU()

    def forward(self, coords):
        with torch.no_grad():
            x = self.relu(self.geo_prev(coords))
        x = self.geo_projector(x)
        x = x.view(-1, self.seq_length, self.output_dim)
        return x

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
text_encoder.eval()

# Load trained model
model_path = "alignment/models/geo_seq_model_cities.pt"
model = GeoEncoder().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Load test data
test_path = "alignment/dataset/val_split.csv"
import pandas as pd
df = pd.read_csv(test_path)
texts = df['text'].tolist()[:32]  # Take first 32 samples
coords = torch.tensor([[row['lat'], row['lon']] for _, row in df.head(32).iterrows()], dtype=torch.float32).to(device)

# Get embeddings
with torch.no_grad():
    # Text embeddings with proper tokenization
    tokens = tokenizer(texts, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
    text_outputs = text_encoder(**tokens)
    text_embed_seq = text_outputs.last_hidden_state  # [32, 77, 768]
    text_embed_pooled = text_outputs.pooler_output   # [32, 768]
    
    # Geo embeddings
    geo_embed_seq = model(coords)  # [32, 77, 768]
    geo_embed_pooled = geo_embed_seq.mean(dim=1)  # [32, 768]

# Normalize embeddings
text_embed_seq = F.normalize(text_embed_seq, dim=-1)
geo_embed_seq = F.normalize(geo_embed_seq, dim=-1)
text_embed_pooled = F.normalize(text_embed_pooled, dim=-1)
geo_embed_pooled = F.normalize(geo_embed_pooled, dim=-1)

# Compute similarity matrices
# 1. Sequence-level similarity (average over sequence)
seq_similarity = torch.sum(geo_embed_seq * text_embed_seq, dim=-1).mean(dim=-1)  # [32]
seq_sim_matrix = geo_embed_pooled @ text_embed_pooled.T  # [32, 32]

# 2. Pooled similarity
pooled_sim_matrix = geo_embed_pooled @ text_embed_pooled.T  # [32, 32]

# Create visualizations
os.makedirs("alignment/results", exist_ok=True)

# Plot comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Sequence similarity matrix (pooled from sequence)
axes[0, 0].imshow(seq_sim_matrix.cpu().numpy(), cmap='viridis', vmin=-1, vmax=1)
axes[0, 0].set_title("Geo Sequence → Text Pooled Similarity")
axes[0, 0].set_xlabel("Text Index")
axes[0, 0].set_ylabel("Geo Index")

# Pooled similarity matrix
axes[0, 1].imshow(pooled_sim_matrix.cpu().numpy(), cmap='viridis', vmin=-1, vmax=1)
axes[0, 1].set_title("Geo Pooled → Text Pooled Similarity")
axes[0, 1].set_xlabel("Text Index")
axes[0, 1].set_ylabel("Geo Index")

# Diagonal values comparison
diag_seq = torch.diag(seq_sim_matrix).cpu().numpy()
diag_pooled = torch.diag(pooled_sim_matrix).cpu().numpy()

axes[1, 0].plot(diag_seq, label='Sequence-based', marker='o')
axes[1, 0].plot(diag_pooled, label='Pooled-based', marker='s')
axes[1, 0].set_title("Diagonal Similarity Values")
axes[1, 0].set_xlabel("Sample Index")
axes[1, 0].set_ylabel("Similarity")
axes[1, 0].legend()
axes[1, 0].grid(True)

# Per-position sequence similarity
pos_similarities = torch.sum(geo_embed_seq * text_embed_seq, dim=-1)  # [32, 77]
avg_pos_sim = pos_similarities.mean(dim=0).cpu().numpy()  # [77]

axes[1, 1].plot(avg_pos_sim)
axes[1, 1].set_title("Average Similarity per Token Position")
axes[1, 1].set_xlabel("Token Position")
axes[1, 1].set_ylabel("Average Similarity")
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig("alignment/results/sequence_comparison.png", dpi=300, bbox_inches='tight')
print("✅ Saved sequence comparison plot")

# Print statistics
print(f"Sequence-based diagonal mean: {diag_seq.mean():.4f} ± {diag_seq.std():.4f}")
print(f"Pooled-based diagonal mean: {diag_pooled.mean():.4f} ± {diag_pooled.std():.4f}")
print(f"Sequence per-position mean: {avg_pos_sim.mean():.4f} ± {avg_pos_sim.std():.4f}")

# Top-k accuracy
def top_k_accuracy(sim_matrix, k=5):
    _, top_indices = torch.topk(sim_matrix, k, dim=1)
    correct = (top_indices == torch.arange(sim_matrix.size(0), device=sim_matrix.device).unsqueeze(1)).any(dim=1)
    return correct.float().mean().item()

print(f"Sequence-based Top-1 accuracy: {top_k_accuracy(seq_sim_matrix, 1):.4f}")
print(f"Sequence-based Top-5 accuracy: {top_k_accuracy(seq_sim_matrix, 5):.4f}")
print(f"Pooled-based Top-1 accuracy: {top_k_accuracy(pooled_sim_matrix, 1):.4f}")
print(f"Pooled-based Top-5 accuracy: {top_k_accuracy(pooled_sim_matrix, 5):.4f}")
