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
test_path = "alignment/dataset/cities.json"# generated_dataset/geo_clip_with_text.json"
model_path = "alignment/models/dummy_model_cities.pt"
os.makedirs("alignment/results", exist_ok=True)

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()

# Load test data
test_loader = DataLoader(GeoTextDataset(test_path), batch_size=256, shuffle=True)
texts, coords = next(iter(test_loader))
coords = coords.to(device)

# Tokenize and encode text
tokens = tokenizer(list(texts), padding=True, return_tensors="pt", truncation=True).to(device)
with torch.no_grad():
    text_embed = text_encoder(**tokens).pooler_output  # [B, 768]
    text_embed = F.normalize(text_embed, dim=-1)

# Load and run trained model
model_trained = GeoEncoder().to(device)
model_trained.load_state_dict(torch.load(model_path))
model_trained.eval()
with torch.no_grad():
    geo_embed_trained = model_trained(coords)
    geo_embed_trained = F.normalize(geo_embed_trained, dim=-1)

# Load and run untrained model (random weights)
model_untrained = GeoEncoder().to(device)
model_untrained.eval()
with torch.no_grad():
    geo_embed_untrained = model_untrained(coords)
    geo_embed_untrained = F.normalize(geo_embed_untrained, dim=-1)


geo_embed_trained = F.normalize(geo_embed_trained, dim=-1)
geo_embed_untrained = F.normalize(geo_embed_untrained, dim=-1)
text_embed = F.normalize(text_embed, dim=-1)

# Compute similarity matrix
similarity_matrix_trained = geo_embed_trained @ text_embed.T  # [B, B]
similarity_matrix_untrained = geo_embed_untrained @ text_embed.T  # [B, B]

# Plot heatmap
# Plot side by side
plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
sns.heatmap(similarity_matrix_untrained.cpu().numpy(), cmap="viridis", square=True,
            cbar_kws={"label": "Cosine Similarity"}, xticklabels=False, yticklabels=False,
            vmin=-1, vmax=1)
plt.title("Untrained Geo Encoder\nCosine Similarity Matrix")
plt.xlabel("Text Embedding Index")
plt.ylabel("Geo Embedding Index")

plt.subplot(1, 2, 2)
sns.heatmap(similarity_matrix_trained.cpu().numpy(), cmap="viridis", square=True,
            cbar_kws={"label": "Cosine Similarity"}, xticklabels=False, yticklabels=False, 
            vmin=-1, vmax=1)
plt.title("Trained Geo Encoder\nCosine Similarity Matrix")
plt.xlabel("Text Embedding Index")
plt.ylabel("Geo Embedding Index")

plt.tight_layout()
plt.savefig("alignment/results/geo_encoder_similarity_matrix_comparison.png")
print("✅ Saved side-by-side heatmap to alignment/results/geo_encoder_similarity_matrix_comparison.png")
