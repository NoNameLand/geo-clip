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
import time
from torch.cuda.amp import autocast, GradScaler

# Dataset with GPU optimization
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
save_path = "alignment/models/geo_seq_model_optimized.pt"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# GPU optimized setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

if torch.cuda.is_available():
    print(f"🔧 CUDA Device: {torch.cuda.get_device_name()}")
    print(f"🔧 CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    # Enable memory efficient settings
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Load dataset with optimized batch size for GPU
full_dataset = GeoTextDataset(data_path)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_set, val_set = torch.utils.data.random_split(full_dataset, [train_size, val_size])

# Optimized batch size and data loading for GPU
batch_size = 64 if torch.cuda.is_available() else 32
num_workers = 4 if torch.cuda.is_available() else 0

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, 
                         num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, 
                       num_workers=num_workers, pin_memory=True)

def save_split(dataset, filename):
    with open(filename, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["text", "lat", "lon"])
        for text, coords in dataset:
            writer.writerow([text, coords[0].item(), coords[1].item()])

save_split(train_set, "alignment/dataset/train_split.csv")
save_split(val_set, "alignment/dataset/val_split.csv")
print("✅ Saved train and val splits to CSV.")

# Load and optimize text encoder
print("🔧 Loading and optimizing text encoder...")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

# Properly move to device and optimize
text_encoder = text_encoder.to(device)
if torch.cuda.is_available():
    text_encoder = text_encoder.half()  # Use half precision for memory efficiency
text_encoder.eval()

# Freeze text encoder parameters
for p in text_encoder.parameters():
    p.requires_grad = False

print(f"✅ Text encoder loaded on {next(text_encoder.parameters()).device}")

# GPU-optimized GeoEncoder with mixture of bases
class GeoEncoder(nn.Module):
    def __init__(self, input_dim=2, output_dim=768, seq_length=77, M=8):
        super(GeoEncoder, self).__init__()
        self.geo_prev = geoclip.LocationEncoder()
        self.seq_length = seq_length
        self.output_dim = output_dim
        self.M = M

        # Initial projection with GPU-friendly operations
        self.initial_proj = nn.Sequential(
            nn.Linear(512, 768, bias=False),  # Remove bias for efficiency
            nn.GELU(),  # GELU is more GPU-friendly than ReLU
            nn.Dropout(0.1)  # Reduced dropout for faster training
        )

        # Mixture of bases - pre-allocated on GPU
        self.prompt_bases = nn.Parameter(
            torch.randn(M, seq_length, output_dim, device=device) * 0.02
        )
        
        # Compact mixing head
        self.mix_head = nn.Sequential(
            nn.Linear(output_dim, 32, bias=False),  # Smaller hidden size
            nn.GELU(),
            nn.Linear(32, M, bias=False)
        )

        # Smaller positional encoding
        self.positional_encoding = nn.Parameter(
            torch.randn(1, seq_length, output_dim, device=device) * 0.01
        )
        
        # Use RMSNorm for better GPU performance
        self.layer_norm = nn.RMSNorm(output_dim) if hasattr(nn, 'RMSNorm') else nn.LayerNorm(output_dim)

    def forward(self, coords):
        batch_size = coords.size(0)
        
        # Keep geo_prev operations on GPU
        with torch.no_grad():
            x = F.gelu(self.geo_prev(coords))  # [B, 512]

        # GPU-optimized operations
        x = self.initial_proj(x)  # [B, 768]

        # Efficient mixture computation using bmm for GPU
        a = self.mix_head(x)  # [B, M]
        a = F.softmax(a, dim=-1, dtype=torch.float32)  # Ensure float32 for stability
        
        # Use batch matrix multiplication for efficiency
        # Reshape prompt_bases for bmm: [M, L, D] -> [1, M, L*D] -> [B, M, L*D]
        bases_flat = self.prompt_bases.view(self.M, -1).unsqueeze(0).expand(batch_size, -1, -1)
        a_expanded = a.unsqueeze(-1)  # [B, M, 1]
        
        # Weighted sum using bmm
        tokens_flat = torch.bmm(a.unsqueeze(1), bases_flat).squeeze(1)  # [B, L*D]
        tokens = tokens_flat.view(batch_size, self.seq_length, self.output_dim)
        
        # Add positional encoding and normalize
        tokens = tokens + self.positional_encoding
        tokens = self.layer_norm(tokens)
        
        return tokens

# Create model with GPU optimization
print("🔧 Creating and optimizing model...")
model = GeoEncoder().to(device)

# Freeze geo_prev for efficiency
for param in model.geo_prev.parameters():
    param.requires_grad = False

# Count parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"📊 Model Parameters:")
print(f"   Trainable: {trainable_params:,}")
print(f"   Frozen: {frozen_params:,}")

# GPU-optimized optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4, 
                             eps=1e-6, betas=(0.9, 0.95))  # More aggressive learning
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

# Mixed precision training for GPU efficiency
scaler = GradScaler() if torch.cuda.is_available() else None
use_amp = torch.cuda.is_available()

epochs = 15
train_losses = []
val_losses = []

# GPU-optimized loss function with reduced memory allocation
def sequence_loss_optimized(geo_embed, text_embed, temperature=0.03):
    # Use more efficient pooling
    geo_pooled = geo_embed.mean(dim=1)
    text_pooled = text_embed.mean(dim=1)
    
    # Normalize in-place for memory efficiency
    geo_pooled = F.normalize(geo_pooled, dim=-1, p=2)
    text_pooled = F.normalize(text_pooled, dim=-1, p=2)
    
    batch_size = geo_pooled.size(0)
    
    # More efficient similarity computation
    geo_sim = geo_pooled @ geo_pooled.t()
    text_sim = text_pooled @ text_pooled.t()
    cross_sim = geo_pooled @ text_pooled.t()
    
    # Build similarity matrix more efficiently
    top_row = torch.cat([geo_sim, cross_sim], dim=1)
    bottom_row = torch.cat([cross_sim.t(), text_sim], dim=1)
    similarity_matrix = torch.cat([top_row, bottom_row], dim=0) / temperature
    
    # Remove diagonal efficiently
    mask = torch.eye(2 * batch_size, device=similarity_matrix.device, dtype=torch.bool)
    similarity_matrix.masked_fill_(mask, float('-inf'))
    
    # Compute loss efficiently
    positives = torch.cat([
        torch.arange(batch_size, batch_size * 2, device=similarity_matrix.device),
        torch.arange(0, batch_size, device=similarity_matrix.device)
    ])
    
    exp_sim = similarity_matrix.exp()
    pos_sim = exp_sim[torch.arange(2 * batch_size), positives]
    
    loss = -torch.log(pos_sim / exp_sim.sum(dim=1)).mean()
    return loss

print(f"🚀 Starting optimized training for {epochs} epochs...")
print(f"   Batch size: {batch_size}")
print(f"   Mixed precision: {use_amp}")
print(f"   Device: {device}")

start_time = time.time()

for epoch in range(epochs):
    model.train()
    total_loss = 0
    epoch_start = time.time()
    
    for batch_idx, (texts, coords) in enumerate(train_loader):
        coords = coords.to(device, non_blocking=True)
        
        # Pre-tokenize more efficiently
        tokens = tokenizer(texts, padding="max_length", truncation=True, 
                          max_length=77, return_tensors="pt")
        tokens = {k: v.to(device, non_blocking=True) for k, v in tokens.items()}

        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        
        if use_amp:
            with autocast():
                # Get text embeddings with mixed precision
                with torch.no_grad():
                    text_outputs = text_encoder(**tokens)
                    text_embed = text_outputs.last_hidden_state.float()  # Convert back to float32

                geo_embed = model(coords)
                loss = sequence_loss_optimized(geo_embed, text_embed)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            with torch.no_grad():
                text_outputs = text_encoder(**tokens)
                text_embed = text_outputs.last_hidden_state

            geo_embed = model(coords)
            loss = sequence_loss_optimized(geo_embed, text_embed)
            
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        
        # Clear cache periodically for memory efficiency
        if batch_idx % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    scheduler.step()
    
    train_loss = total_loss / len(train_loader)
    train_losses.append(train_loss)
    
    # Validation with optimization
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for texts, coords in val_loader:
            coords = coords.to(device, non_blocking=True)
            tokens = tokenizer(texts, padding="max_length", truncation=True, 
                              max_length=77, return_tensors="pt")
            tokens = {k: v.to(device, non_blocking=True) for k, v in tokens.items()}
            
            if use_amp:
                with autocast():
                    text_outputs = text_encoder(**tokens)
                    text_embed = text_outputs.last_hidden_state.float()
                    geo_embed = model(coords)
                    val_loss += sequence_loss_optimized(geo_embed, text_embed).item()
            else:
                text_outputs = text_encoder(**tokens)
                text_embed = text_outputs.last_hidden_state
                geo_embed = model(coords)
                val_loss += sequence_loss_optimized(geo_embed, text_embed).item()

    val_loss = val_loss / len(val_loader)
    val_losses.append(val_loss)
    
    epoch_time = time.time() - epoch_start
    print(f"Epoch {epoch+1:2d} - Train: {train_loss:.4f}, Val: {val_loss:.4f}, "
          f"Time: {epoch_time:.1f}s, LR: {scheduler.get_last_lr()[0]:.2e}")

total_time = time.time() - start_time
print(f"\n⚡ Training completed in {total_time:.1f}s ({total_time/epochs:.1f}s/epoch)")

# Save optimized model
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'train_losses': train_losses,
    'val_losses': val_losses,
    'trainable_params': trainable_params,
}, save_path)
print(f"✅ Saved optimized model to {save_path}")

# Save losses plot
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss', alpha=0.8)
plt.plot(val_losses, label='Val Loss', alpha=0.8)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss - GPU Optimized')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("alignment/models/optimized_loss_plot.png", dpi=150, bbox_inches='tight')
print("✅ Saved loss plot")

print(f"\n🎯 Optimization Summary:")
print(f"   ⚡ Mixed precision training: {use_amp}")
print(f"   🔧 Batch size: {batch_size}")
print(f"   📊 Total trainable parameters: {trainable_params:,}")
print(f"   ⏱️ Average time per epoch: {total_time/epochs:.1f}s")
print(f"   🚀 GPU memory optimizations applied")
print(f"   ✅ Training completed successfully!")

if torch.cuda.is_available():
    print(f"\n📈 GPU Utilization:")
    print(f"   Max memory allocated: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
    print(f"   Current memory allocated: {torch.cuda.memory_allocated()/1e9:.1f} GB")
