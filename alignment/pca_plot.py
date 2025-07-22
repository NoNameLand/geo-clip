import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel
import geoclip
import json
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import PCA
import csv
from mpl_toolkits.mplot3d import Axes3D
# ...existing dataset and model code...
# Dataset
class GeoTextDataset(Dataset):
    def __init__(self, path):
        with open(path) as f:
            if path.endswith('.csv'):
                self.data = []
                with open(path, 'r') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        self.data.append({'text': row['text'], 'lat': row['lat'], 'lon': row['lon']})
            else:
                self.data = json.load(f)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        return item['text'], torch.tensor([float(item['lat']), float(item['lon'])], dtype=torch.float32)

# Model
class GeoEncoder(nn.Module):
    def __init__(self, input_dim=2, output_dim=768):
        super(GeoEncoder, self).__init__()
        self.geo_prev = geoclip.LocationEncoder()
        self.projector = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, output_dim)
        )
        self.relu = nn.ReLU()

    def forward(self, coords):
        with torch.no_grad():  # Freeze geo_prev
            x = self.relu(self.geo_prev(coords))
        x = self.projector(x)
        return x

# Paths
test_path = "alignment/dataset/cities.json" #generated_dataset/val_split.csv"
model_path = "alignment/models/dummy_model_cities.pt"

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
test_loader = DataLoader(GeoTextDataset(test_path), batch_size=32)

# Load test data
test_loader = DataLoader(GeoTextDataset(test_path), batch_size=256, shuffle=True)
texts, coords = next(iter(test_loader))
coords = coords.to(device)

# Tokenize and encode text
tokens = tokenizer(list(texts), padding=True, return_tensors="pt", truncation=True).to(device)
with torch.no_grad():
    text_embed = text_encoder(**tokens).pooler_output  # [B, 768]
    text_embed = F.normalize(text_embed, dim=-1)

# Trained geo encoder
model_trained = GeoEncoder().to(device)
model_trained.load_state_dict(torch.load(model_path))
model_trained.eval()
with torch.no_grad():
    geo_embed_trained = model_trained(coords)
    geo_embed_trained = F.normalize(geo_embed_trained, dim=-1)

# Untrained geo encoder
model_untrained = GeoEncoder().to(device)
model_untrained.eval()
with torch.no_grad():
    geo_embed_untrained = model_untrained(coords)
    geo_embed_untrained = F.normalize(geo_embed_untrained, dim=-1)

# PCA to 2D
pca = PCA(n_components=2)
geo_trained_2d = pca.fit_transform(geo_embed_trained.cpu().numpy())
geo_untrained_2d = pca.fit_transform(geo_embed_untrained.cpu().numpy())
text_2d = pca.fit_transform(text_embed.cpu().numpy())

# Plot
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.scatter(geo_untrained_2d[:, 0], geo_untrained_2d[:, 1], c='tab:blue', alpha=0.7)
plt.title("Untrained Geo Encoder (PCA 2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.subplot(1, 3, 2)
plt.scatter(geo_trained_2d[:, 0], geo_trained_2d[:, 1], c='tab:green', alpha=0.7)
plt.title("Trained Geo Encoder (PCA 2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.subplot(1, 3, 3)
plt.scatter(text_2d[:, 0], text_2d[:, 1], c='tab:orange', alpha=0.7)
plt.title("Text Encoder (PCA 2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.tight_layout()
plt.savefig("alignment/results/pca_geo_text_comparison.png")
print("✅ Saved PCA comparison plot to alignment/results/pca_geo_text_comparison.png")

# Stack embeddings for joint PCA
all_embeds = torch.cat([geo_embed_trained, geo_embed_untrained, text_embed], dim=0).cpu().numpy()
pca = PCA(n_components=2)
all_embeds_2d = pca.fit_transform(all_embeds)

B = geo_embed_trained.shape[0]
geo_trained_2d = all_embeds_2d[:B]
geo_untrained_2d = all_embeds_2d[B:2*B]
text_2d = all_embeds_2d[2*B:]

plt.figure(figsize=(8, 8))
plt.scatter(text_2d[:, 0], text_2d[:, 1], c='tab:orange', label='Text Encoder', alpha=0.7)
plt.scatter(geo_trained_2d[:, 0], geo_trained_2d[:, 1], c='tab:green', label='Trained Geo Encoder', alpha=0.7)
plt.scatter(geo_untrained_2d[:, 0], geo_untrained_2d[:, 1], c='tab:blue', label='Untrained Geo Encoder', alpha=0.7)

# Connect corresponding points (geo <-> text)
for i in range(B):
    plt.plot([geo_trained_2d[i, 0], text_2d[i, 0]], [geo_trained_2d[i, 1], text_2d[i, 1]], c='gray', alpha=0.3)
    plt.plot([geo_untrained_2d[i, 0], text_2d[i, 0]], [geo_untrained_2d[i, 1], text_2d[i, 1]], c='lightblue', alpha=0.2)

plt.legend()
plt.title("PCA 2D: Text vs. Geo Embeddings\n(lines connect corresponding points)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig("alignment/results/pca_geo_text_lines.png")
print("✅ Saved PCA plot with lines to alignment/results/pca_geo_text_lines.png")

# Stack for joint PCA
all_embeds = torch.cat([geo_embed_trained, geo_embed_untrained, text_embed], dim=0).cpu().numpy()
pca = PCA(n_components=3)
all_embeds_3d = pca.fit_transform(all_embeds)

B = geo_embed_trained.shape[0]
geo_trained_3d = all_embeds_3d[:B]
geo_untrained_3d = all_embeds_3d[B:2*B]
text_3d = all_embeds_3d[2*B:]

# Untrained Geo Encoder vs Text Encoder
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(text_3d[:, 0], text_3d[:, 1], text_3d[:, 2], c='tab:orange', label='Text Encoder', alpha=0.7)
ax.scatter(geo_untrained_3d[:, 0], geo_untrained_3d[:, 1], geo_untrained_3d[:, 2], c='tab:blue', label='Untrained Geo Encoder', alpha=0.7)
for i in range(B):
    ax.plot([geo_untrained_3d[i, 0], text_3d[i, 0]],
            [geo_untrained_3d[i, 1], text_3d[i, 1]],
            [geo_untrained_3d[i, 2], text_3d[i, 2]],
            c='lightblue', alpha=0.3)
ax.set_title("3D PCA: Untrained Geo Encoder vs Text Encoder")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.legend()
plt.tight_layout()
plt.savefig("alignment/results/pca_3d_untrained_geo_text.png")
plt.close(fig)
print("✅ Saved 3D PCA plot for untrained geo encoder to alignment/results/pca_3d_untrained_geo_text.png")

# Trained Geo Encoder vs Text Encoder
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(text_3d[:, 0], text_3d[:, 1], text_3d[:, 2], c='tab:orange', label='Text Encoder', alpha=0.7)
ax.scatter(geo_trained_3d[:, 0], geo_trained_3d[:, 1], geo_trained_3d[:, 2], c='tab:green', label='Trained Geo Encoder', alpha=0.7)
for i in range(B):
    ax.plot([geo_trained_3d[i, 0], text_3d[i, 0]],
            [geo_trained_3d[i, 1], text_3d[i, 1]],
            [geo_trained_3d[i, 2], text_3d[i, 2]],
            c='gray', alpha=0.3)
ax.set_title("3D PCA: Trained Geo Encoder vs Text Encoder")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.legend()
plt.tight_layout()
plt.savefig("alignment/results/pca_3d_trained_geo_text.png")
plt.close(fig)
print("✅ Saved 3D PCA plot for trained geo encoder to alignment/results/pca_3d_trained_geo_text.png")