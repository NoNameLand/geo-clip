import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel
import geoclip
# Load SD text encoder (CLIP ViT-L/14)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
text_encoder.eval()  # Freeze
for p in text_encoder.parameters():
    p.requires_grad = False

# Geo Encoder: MLP that maps (lat, lon) -> 768-dim embeddin
class GeoEncoder(nn.Module):
    def __init__(self, input_dim=2, output_dim=768):
        super(GeoEncoder, self).__init__()
        self.geo_prev = geoclip.LocationEncoder()
        self.projector = nn.Linear(512, output_dim)
        self.relu = nn.ReLU()

    def forward(self, coords):
        x = self.relu(self.geo_prev(coords))
        x = self.projector(x)
        return x

geo_encoder = GeoEncoder()

optimizer = torch.optim.Adam(geo_encoder.parameters(), lr=1e-4)

# Dummy batch
batch = [
    {"text": "desert",        "location": [33.6, -111.8]},
    {"text": "tropical forest", "location": [-3.0, -60.0]},
    {"text": "mountain village", "location": [46.6, 10.7]},
]

texts = [item["text"] for item in batch]
coords = torch.tensor([item["location"] for item in batch], dtype=torch.float32)

# Get target text embeddings from SD encoder
tokens = tokenizer(texts, padding=True, return_tensors="pt")
with torch.no_grad():
    text_embeds = text_encoder(**tokens).last_hidden_state[:, 0]  # CLS token, shape: [B, 768]

# Train one step
geo_embeds = geo_encoder(coords)  # shape: [B, 768]

# Loss: cosine or MSE
loss = F.mse_loss(geo_embeds, text_embeds)
# loss = 1 - F.cosine_similarity(geo_embeds, text_embeds).mean()

loss.backward()
optimizer.step()

print("Loss:", loss.item())
