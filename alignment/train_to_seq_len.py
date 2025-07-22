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
save_path = "alignment/models/geo_seq_model_cities.pt"
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

# Geo Encoder: MLP that maps (lat, lon) -> [seq_len, output_dim] embedding
class GeoEncoder(nn.Module):
    def __init__(self, input_dim=2, output_dim=768, seq_length=77):
        super(GeoEncoder, self).__init__()
        self.geo_prev = geoclip.LocationEncoder()
        self.seq_length = seq_length
        self.output_dim = output_dim
        
        # Project geo features to sequence dimension
        self.geo_projector = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, seq_length * output_dim)
        )
        self.relu = nn.ReLU()

    def forward(self, coords):
        with torch.no_grad():  # Freeze geo_prev
            x = self.relu(self.geo_prev(coords))  # [batch_size, 512]
        
        x = self.geo_projector(x)  # [batch_size, seq_length * output_dim]
        x = x.view(-1, self.seq_length, self.output_dim)  # [batch_size, seq_length, output_dim]
        return x

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GeoEncoder().to(device)
text_encoder = text_encoder.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-4)
epochs = 15
train_losses = []
val_losses = []
# Loss function: SimCLR-style contrastive loss for sequence embeddings
def simclr_loss(geo_embed, text_embed, temperature=0.03):
    # Average over sequence dimension to get [batch_size, output_dim]
    geo_embed_pooled = geo_embed.mean(dim=1)  # [batch_size, output_dim]
    
    geo_embed_pooled = F.normalize(geo_embed_pooled, dim=-1)
    text_embed = F.normalize(text_embed, dim=-1)
    batch_size = geo_embed_pooled.size(0)

    # Concatenate embeddings
    embeddings = torch.cat([geo_embed_pooled, text_embed], dim=0)  # [2*B, D]
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

# Loss function for sequence-to-sequence comparison (SimCLR-style)
def sequence_loss(geo_embed, text_embed, temperature=0.03):
    # geo_embed: [batch_size, seq_len, output_dim]
    # text_embed: [batch_size, seq_len, output_dim]
    
    # Pool sequences to get [batch_size, output_dim]
    geo_embed_pooled = geo_embed.mean(dim=1)  # [batch_size, output_dim]
    text_embed_pooled = text_embed.mean(dim=1)  # [batch_size, output_dim]
    
    # Normalize embeddings
    geo_embed_pooled = F.normalize(geo_embed_pooled, dim=-1)
    text_embed_pooled = F.normalize(text_embed_pooled, dim=-1)
    
    batch_size = geo_embed_pooled.size(0)

    # Concatenate embeddings
    embeddings = torch.cat([geo_embed_pooled, text_embed_pooled], dim=0)  # [2*B, D]
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
        tokens = tokenizer(list(texts), padding="max_length", truncation=True, max_length=77, return_tensors="pt").to(device)

        # Ensure tokens are padded to length 77
        tokens['input_ids'] = F.pad(tokens['input_ids'], (0, 77 - tokens['input_ids'].size(1)), value=tokenizer.pad_token_id)
        tokens['attention_mask'] = F.pad(tokens['attention_mask'], (0, 77 - tokens['attention_mask'].size(1)), value=0)

        with torch.no_grad():
            # Get full token embeddings [batch_size, seq_len, hidden_size]
            text_outputs = text_encoder(**tokens)
            text_embed = text_outputs.last_hidden_state  # [batch_size, 77, 768]

        geo_embed = model(coords)  # [batch_size, 77, 768]
        
        # Use sequence-to-sequence loss
        loss = sequence_loss(geo_embed, text_embed)

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
            tokens = tokenizer(list(texts), padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
            # Ensure tokens are padded to length 77
            tokens['input_ids'] = F.pad(tokens['input_ids'], (0, 77 - tokens['input_ids'].size(1)), value=tokenizer.pad_token_id)
            tokens['attention_mask'] = F.pad(tokens['attention_mask'], (0, 77 - tokens['attention_mask'].size(1)), value=0)
            text_outputs = text_encoder(**tokens)
            text_embed = text_outputs.last_hidden_state  # [batch_size, 77, 768]
            geo_embed = model(coords)  # [batch_size, 77, 768]
            val_loss += sequence_loss(geo_embed, text_embed).item()

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

# Quick test to verify the model learns
print("\n🧪 Running quick learning test...")
model.eval()
with torch.no_grad():
    # Test with first batch
    test_batch = next(iter(val_loader))
    texts, coords = test_batch
    coords = coords.to(device)
    tokens = tokenizer(list(texts), padding="max_length", truncation=True, max_length=77, return_tensors="pt").to(device)
    
    # Ensure tokens are padded to length 77
    tokens['input_ids'] = F.pad(tokens['input_ids'], (0, 77 - tokens['input_ids'].size(1)), value=tokenizer.pad_token_id)
    tokens['attention_mask'] = F.pad(tokens['attention_mask'], (0, 77 - tokens['attention_mask'].size(1)), value=0)
    
    text_outputs = text_encoder(**tokens)
    text_embed = text_outputs.last_hidden_state  # [batch_size, 77, 768]
    geo_embed = model(coords)  # [batch_size, 77, 768]
    
    # Pool embeddings for similarity comparison
    geo_pooled = geo_embed.mean(dim=1)  # [batch_size, 768]
    text_pooled = text_embed.mean(dim=1)  # [batch_size, 768]
    
    # Normalize and compute similarities
    geo_pooled = F.normalize(geo_pooled, dim=-1)
    text_pooled = F.normalize(text_pooled, dim=-1)
    
    # Compute cosine similarity matrix
    similarity_matrix = geo_pooled @ text_pooled.T
    
    # Check diagonal (correct pairs) vs off-diagonal (incorrect pairs)
    diagonal_sim = torch.diag(similarity_matrix).mean().item()
    off_diagonal_sim = (similarity_matrix.sum() - torch.diag(similarity_matrix).sum()) / (similarity_matrix.numel() - len(similarity_matrix))
    
    print(f"✅ Average diagonal similarity (correct pairs): {diagonal_sim:.4f}")
    print(f"✅ Average off-diagonal similarity (incorrect pairs): {off_diagonal_sim:.4f}")
    print(f"✅ Learning ratio (diagonal/off-diagonal): {diagonal_sim/off_diagonal_sim:.2f}")
    
    if diagonal_sim > off_diagonal_sim:
        print("🎉 Model is learning! Correct pairs are more similar than incorrect pairs.")
    else:
        print("⚠️ Model needs more training. Incorrect pairs are still more similar.")

print("\n✅ Training and testing complete!")
