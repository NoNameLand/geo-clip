import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel
import geoclip
from torch.utils.data import Dataset, DataLoader
import os 
import json
import matplotlib.pyplot as plt
import csv
# Dataset
class GeoTextDataset(Dataset):
    def __init__(self, path):
        with open(path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item['text'], torch.tensor([float(item['lat']), float(item['lon'])], dtype=torch.float32)

# Paths
data_path = "alignment/dataset/cities.json"
save_path = "alignment/models/dummy_model_cities.pt"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Load dataset
full_dataset = GeoTextDataset(data_path)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_set, val_set = torch.utils.data.random_split(full_dataset, [train_size, val_size])
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=True)

def save_split(dataset, filename):
    with open(filename, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["text", "lat", "lon"])
        for text, coords in dataset:
            writer.writerow([text, coords[0].item(), coords[1].item()])

save_split(train_set, "alignment/dataset/train_split.csv")
save_split(val_set, "alignment/dataset/val_split.csv")
print("✅ Saved train and val splits to CSV.")


# Load SD text encoder (CLIP ViT-L/14)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
text_encoder.eval()  # Freeze
for p in text_encoder.parameters():
    p.requires_grad = False

# Geo Encoder: MLP that maps (lat, lon) -> 768-dim embedding
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

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GeoEncoder().to(device)
text_encoder = text_encoder.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-4)
epochs = 30
train_losses = []
val_losses = []
# Loss function: SimCLR-style contrastive loss
def simclr_loss(geo_embed, text_embed, temperature=0.03):
    geo_embed = F.normalize(geo_embed, dim=-1)
    text_embed = F.normalize(text_embed, dim=-1)
    batch_size = geo_embed.size(0)

    # Concatenate embeddings
    embeddings = torch.cat([geo_embed, text_embed], dim=0)  # [2*B, D]
    similarity_matrix = embeddings @ embeddings.T / temperature  # [2*B, 2*B]

    # Remove self-similarity
    mask = torch.eye(2 * batch_size, device=embeddings.device).bool()
    similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))

    # For each anchor, the positive is at index: i <-> i+batch_size and i+batch_size <-> i
    positives = torch.cat([
        torch.arange(batch_size, device=embeddings.device) + batch_size,
        torch.arange(batch_size, device=embeddings.device)
    ])

    # Compute numerator and denominator for each anchor
    exp_sim = torch.exp(similarity_matrix)
    pos_sim = exp_sim[torch.arange(2 * batch_size), positives]
    denom = exp_sim.sum(dim=1)

    # SimCLR loss
    loss = -torch.log(pos_sim / denom).mean()
    return loss

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for texts, coords in train_loader:
        coords = coords.to(device)
        tokens = tokenizer(list(texts), padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            text_embed = text_encoder(**tokens).pooler_output.to(device)

        geo_embed = model(coords)
        loss = simclr_loss(geo_embed, text_embed)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} - Train Loss: {total_loss/len(train_loader):.4f}")
    train_losses.append(total_loss / len(train_loader))

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for texts, coords in val_loader:
            coords = coords.to(device)
            tokens = tokenizer(list(texts), padding=True, return_tensors="pt").to(device)
            text_embed = text_encoder(**tokens).pooler_output.to(device)
            geo_embed = model(coords)
            val_loss += simclr_loss(geo_embed, text_embed).item()

    val_losses.append(val_loss / len(val_loader))
    print(f"Epoch {epoch+1} - Val Loss: {val_loss/len(val_loader):.4f}")

# Save model
torch.save(model.state_dict(), save_path)
print(f"✅ Saved model to {save_path}")

# Save Losses Plot
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig("alignment/models/loss_plot.png")
print("✅ Saved loss plot to alignment/models/loss_plot.png")
