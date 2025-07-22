import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel
import geoclip
import json
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import csv
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
test_path = "alignment/dataset/generated_dataset/val_split.csv"
model_path = "alignment/models/dummy_model_cities.pt"

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
test_loader = DataLoader(GeoTextDataset(test_path), batch_size=32)

# Evaluation
def get_similarities(model):
    model.eval()
    model.to(device)
    sims = []
    with torch.no_grad():
        for texts, coords in test_loader:
            coords = coords.to(device)
            tokens = tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True).to(device)
            text_embed = text_encoder(**tokens).pooler_output
            geo_embed = model(coords)
            sim = F.cosine_similarity(geo_embed, text_embed, dim=-1)
            sims.extend(sim.cpu().numpy())
    return sims

# Compare before and after training
model_untrained = GeoEncoder()
model_trained = GeoEncoder()
model_trained.load_state_dict(torch.load(model_path))

before = get_similarities(model_untrained)
after = get_similarities(model_trained)

# Plot
plt.figure(figsize=(10, 5))
plt.hist(before, bins=20, alpha=0.5, label="Before Training", density=True)
plt.hist(after, bins=20, alpha=0.5, label="After Training", density=True )
plt.xlabel("Cosine Similarity")
plt.ylabel("Density")
plt.title("Geo Encoder vs. Text Embedding (Before vs. After Training)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("alignment/results/geo_encoder_comparison.png")