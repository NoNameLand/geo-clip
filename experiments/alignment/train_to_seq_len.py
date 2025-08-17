import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel
from torch.utils.data import Dataset, DataLoader
import os 
import json
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
import sys
import numpy as np

# Add parent directory to path for geoclip imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import geoclip_og
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

# Load dataset with GPU optimizations
full_dataset = GeoTextDataset(data_path)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_set, val_set = torch.utils.data.random_split(full_dataset, [train_size, val_size])

# Optimized DataLoader settings for GPU
batch_size = 64 if torch.cuda.is_available() else 32
num_workers = 4 if torch.cuda.is_available() else 0

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, 
                         num_workers=num_workers, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, 
                       num_workers=num_workers, pin_memory=True, persistent_workers=True)

def save_split(dataset, filename):
    with open(filename, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["text", "lat", "lon"])
        for text, coords in dataset:
            writer.writerow([text, coords[0].item(), coords[1].item()])

save_split(train_set, "alignment/dataset/train_split.csv")
save_split(val_set, "alignment/dataset/val_split.csv")
print("✅ Saved train and val splits to CSV.")


# Load SD text encoder (CLIP ViT-L/14) with proper device handling
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🚀 Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    # Enable GPU optimizations for faster training
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

# Initialize text encoder (device assignment will be handled in training loop)
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
text_encoder.eval()  # Freeze
for p in text_encoder.parameters():
    p.requires_grad = False

# Move text encoder to device after model initialization
text_encoder = text_encoder.to(device)

def build_clip_token_anchors(prompts, device="cuda", dtype=torch.float32):
    """Build anchor token embeddings from text prompts using CLIP"""
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder.eval()
    if isinstance(device, str):
        device = torch.device(device)
    text_encoder = text_encoder.to(device)
    
    anchors = []
    with torch.no_grad():
        for prompt in prompts:
            tokens = tokenizer([prompt], padding="max_length", truncation=True, 
                             max_length=77, return_tensors="pt").to(device)
            text_outputs = text_encoder(**tokens)
            anchor = text_outputs.last_hidden_state  # [1, 77, 768]
            anchors.append(anchor.squeeze(0))  # [77, 768]
    
    # Stack to [M, 77, 768]
    anchors = torch.stack(anchors, dim=0).to(dtype)
    return anchors

class GeoPrompt77(nn.Module):
    """Geo-conditioned prompt generator using frozen anchor embeddings"""
    def __init__(self, location_encoder, input_feat_dim=512, output_dim=768, 
                 seq_length=77, anchors=None, M=8, r=8):
        super(GeoPrompt77, self).__init__()
        
        # Freeze the location encoder
        self.location_encoder = location_encoder
        for param in self.location_encoder.parameters():
            param.requires_grad = False
        
        self.input_feat_dim = input_feat_dim
        self.output_dim = output_dim
        self.seq_length = seq_length
        self.M = M
        self.r = r
        
        # Store frozen anchor embeddings [M, 77, 768]
        if anchors is not None:
            self.register_buffer('anchors', anchors)  # frozen
        else:
            # Fallback random anchors if not provided
            self.register_buffer('anchors', torch.randn(M, seq_length, output_dim) * 0.02)
        
        # Conditioning network: geo features -> mixing weights
        self.conditioning_net = nn.Sequential(
            nn.Linear(input_feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, M)  # output M mixing weights
        )
        
        # Optional: low-rank adaptation for fine-tuning anchors
        if r > 0:
            self.lora_A = nn.Parameter(torch.randn(M, seq_length, r) * 0.02)
            self.lora_B = nn.Parameter(torch.randn(M, r, output_dim) * 0.02)
        else:
            self.lora_A = None
            self.lora_B = None
        
        # Final layer norm
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, coords):
        batch_size = coords.size(0)
        
        # Get geo features (frozen)
        with torch.no_grad():
            geo_features = F.relu(self.location_encoder(coords))  # [B, 512]
        
        # Compute mixing weights
        mixing_weights = self.conditioning_net(geo_features)  # [B, M]
        mixing_weights = F.softmax(mixing_weights, dim=-1)     # [B, M]
        
        # Get base anchors
        base_tokens = self.anchors  # [M, 77, 768]
        
        # Apply LoRA adaptation if enabled
        anchors_tensor = self.get_buffer('anchors')  # Get tensor from buffer
        if self.lora_A is not None and self.lora_B is not None:
            # Low-rank adaptation: anchors + A @ B
            lora_delta = torch.matmul(self.lora_A, self.lora_B)  # [M, 77, 768]
            adapted_tokens = anchors_tensor + lora_delta
        else:
            adapted_tokens = anchors_tensor
        
        # Mix anchors based on geo conditioning
        # [B, M] @ [M, 77, 768] -> [B, 77, 768]
        output_tokens = torch.einsum('bm,mld->bld', mixing_weights, adapted_tokens)
        
        # Final normalization
        output_tokens = self.layer_norm(output_tokens)
        
        return output_tokens

# Build anchors from urban prompts
prompts = [
    "a photo of a city skyline, high detail",
    "wide aerial cityscape, daytime", 
    "aerial night city lights, long exposure",
    "street-level urban scene with pedestrians",
    "historic old town architecture, narrow streets",
    "modern glass skyscrapers downtown",
    "coastal city skyline with water reflection",
    "dense urban core, overcast"
]

# Setup with GPU optimizations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("🏗️ Building CLIP token anchors...")
anchors = build_clip_token_anchors(prompts, device=str(device), dtype=torch.float32)  # [M,77,768]
print(f"✅ Built {len(prompts)} anchor embeddings: {anchors.shape}")

model = GeoPrompt77(
    location_encoder=geoclip_og.LocationEncoder(),
    input_feat_dim=512,
    output_dim=768,
    seq_length=77,
    anchors=anchors,
    M=len(prompts),
    r=8  # LoRA rank
).to(device)

# Optimized optimizer for faster convergence
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4, 
                             eps=1e-6, betas=(0.9, 0.95))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

# Count parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"📊 Trainable parameters: {trainable_params:,}")

epochs = 15
train_losses = []
val_losses = []

print(f"🚀 Training with batch_size={batch_size}, device={device}")
import time
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

# Mixed precision training setup
scaler = torch.cuda.amp.GradScaler()

for epoch in range(epochs):
    model.train()
    total_loss = 0
    epoch_start = time.time()
    
    # Use tqdm for progress tracking
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
        for batch_idx, (texts, coords) in enumerate(pbar):
            # Clear cache periodically for memory management
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
            
            coords = coords.to(device, non_blocking=True)
            tokens = tokenizer(list(texts), padding="max_length", truncation=True, max_length=77, return_tensors="pt")
            tokens = {k: v.to(device, non_blocking=True) for k, v in tokens.items()}

            # Ensure tokens are padded to length 77
            tokens['input_ids'] = F.pad(tokens['input_ids'], (0, 77 - tokens['input_ids'].size(1)), value=tokenizer.pad_token_id)
            tokens['attention_mask'] = F.pad(tokens['attention_mask'], (0, 77 - tokens['attention_mask'].size(1)), value=0)

            with torch.no_grad(), torch.cuda.amp.autocast():
                # Get full token embeddings [batch_size, seq_len, hidden_size]
                text_outputs = text_encoder(**tokens)
                text_embed = text_outputs.last_hidden_state  # [batch_size, 77, 768]

            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                geo_embed = model(coords)  # [batch_size, 77, 768]
                
                # Use sequence-to-sequence loss
                loss = sequence_loss(geo_embed, text_embed)

            # Backward pass with gradient scaling
            optimizer.zero_grad(set_to_none=True)  # More memory efficient
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            
            # Update progress bar with GPU memory info
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() // 1024**2
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}', 
                    'mem': f'{memory_used}MB',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
                })

    epoch_time = time.time() - epoch_start
    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Time: {epoch_time:.1f}s")
    train_losses.append(avg_train_loss)

    # Validation with mixed precision
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for texts, coords in val_loader:
            coords = coords.to(device, non_blocking=True)
            tokens = tokenizer(list(texts), padding=True, truncation=True, max_length=77, return_tensors="pt")
            tokens = {k: v.to(device, non_blocking=True) for k, v in tokens.items()}
            
            # Ensure tokens are padded to length 77
            tokens['input_ids'] = F.pad(tokens['input_ids'], (0, 77 - tokens['input_ids'].size(1)), value=tokenizer.pad_token_id)
            tokens['attention_mask'] = F.pad(tokens['attention_mask'], (0, 77 - tokens['attention_mask'].size(1)), value=0)
            
            with torch.cuda.amp.autocast():
                text_outputs = text_encoder(**tokens)
                text_embed = text_outputs.last_hidden_state  # [batch_size, 77, 768]
                geo_embed = model(coords)  # [batch_size, 77, 768]
                val_loss += sequence_loss(geo_embed, text_embed).item()

    val_losses.append(val_loss / len(val_loader))
    print(f"Epoch {epoch+1} - Val Loss: {val_loss/len(val_loader):.4f}")
    
    # Step the scheduler
    scheduler.step()
    
    # Save checkpoint periodically
    if (epoch + 1) % 5 == 0:
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': val_loss / len(val_loader)
        }
        checkpoint_path = f'alignment/models/anchor_checkpoint_epoch_{epoch+1}.pth'
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Checkpoint saved to {checkpoint_path}")

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
    geo_embed = model(coords)  # [batch_size, 77, 768] - now using GeoPrompt77
    
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
