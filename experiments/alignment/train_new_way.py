# train_geo_softprompt.py
# Geo → CLIP token-sequence alignment via a learnable soft-prompt dictionary.
# Keeps your dataset + frozen CLIP text encoder. Emits [B,77,768] tokens ready for SD cross-attn.

import os
import sys
import json
import csv
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import CLIPTokenizer, CLIPTextModel
import matplotlib.pyplot as plt

# ----------------------------
# Imports for your geo encoder
# ----------------------------
# Add parent directory to path for geoclip imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import geoclip_og  # must provide geoclip_og.LocationEncoder()

# ----------------------------
# Dataset
# ----------------------------
class GeoTextDataset(Dataset):
    def __init__(self, path):
        with open(path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        coords = torch.tensor([float(item['lat']), float(item['lon'])], dtype=torch.float32)
        return text, coords

# ----------------------------
# Paths & setup
# ----------------------------
data_path = "alignment/dataset/cities.json"
save_dir = "alignment/models"
os.makedirs(save_dir, exist_ok=True)

save_model_path = os.path.join(save_dir, "geo_softprompt_model_cities.pt")
loss_plot_path = os.path.join(save_dir, "loss_plot_softprompt.png")

# ----------------------------
# Load dataset & splits
# ----------------------------
full_dataset = GeoTextDataset(data_path)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_set, val_set = torch.utils.data.random_split(full_dataset, [train_size, val_size])

batch_size = 64 if torch.cuda.is_available() else 32
num_workers = 4 if torch.cuda.is_available() else 0

train_loader = DataLoader(
    train_set, batch_size=batch_size, shuffle=True,
    num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
)
val_loader = DataLoader(
    val_set, batch_size=batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
)

def save_split(dataset, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["text", "lat", "lon"])
        for text, coords in dataset:
            writer.writerow([text, coords[0].item(), coords[1].item()])

save_split(train_set, "alignment/dataset/train_split.csv")
save_split(val_set, "alignment/dataset/val_split.csv")
print("✅ Saved train and val splits to CSV.")

# ----------------------------
# Device & CUDA knobs
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

# ----------------------------
# Tokenizer & frozen CLIP text encoder (ViT-L/14)
# ----------------------------
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
text_encoder.eval()
for p in text_encoder.parameters():
    p.requires_grad = False
text_encoder = text_encoder.to(device)

# ----------------------------
# Build CLIP token anchors (for dictionary init)
# ----------------------------
def build_clip_token_anchors(prompts, device="cuda", dtype=torch.float32):
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    enc = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    enc.eval().to(device)
    anchors = []
    with torch.no_grad():
        for prompt in prompts:
            toks = tok([prompt], padding="max_length", truncation=True,
                       max_length=77, return_tensors="pt").to(device)
            outs = enc(**toks)
            # last_hidden_state: [1,77,768]
            anchors.append(outs.last_hidden_state.squeeze(0))  # [77,768]
    return torch.stack(anchors, dim=0).to(dtype)  # [M,77,768]

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
print("🏗️ Building CLIP token anchors...")
anchors = build_clip_token_anchors(prompts, device=str(device), dtype=torch.float32)  # [M,77,768]
print(f"✅ Built {len(prompts)} anchor embeddings: {anchors.shape}")

# ----------------------------
# Soft-prompt dictionary & Geo mixer
# ----------------------------
class SoftPromptDictionary(nn.Module):
    """
    Learnable dictionary of K soft tokens (each 77x768).
    Initialized from CLIP anchors for stability, then trained.
    """
    def __init__(self, K=64, seq_len=77, dim=768, init_anchors=None, init_scale=0.02):
        super().__init__()
        if init_anchors is not None:
            base = init_anchors.detach().clone()  # [M,77,768]
            M = base.size(0)
            if K <= M:
                init = base[:K]
            else:
                reps = int(np.ceil(K / M))
                init = base.repeat(reps, 1, 1)[:K]
        else:
            init = torch.randn(K, seq_len, dim) * init_scale
        self.tokens = nn.Parameter(init)  # [K,77,768]

    def forward(self):
        return self.tokens  # [K,77,768]

class GeoDictMixer(nn.Module):
    """
    Maps GeoCLIP -> mixture over K soft tokens -> emits [B,77,768] token sequence.
    """
    def __init__(self, location_encoder, dict_module: SoftPromptDictionary,
                 input_feat_dim=512, K=64, temperature=0.7):
        super().__init__()
        self.location_encoder = location_encoder
        for p in self.location_encoder.parameters():
            p.requires_grad = False

        self.dict = dict_module
        self.K = K
        self.temperature = temperature

        self.conditioning_net = nn.Sequential(
            nn.Linear(input_feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, K)
        )

    def forward(self, coords):
        with torch.no_grad():
            geo_feat = F.relu(self.location_encoder(coords))  # [B,512]
        logits = self.conditioning_net(geo_feat) / self.temperature  # [B,K]
        weights = F.softmax(logits, dim=-1)  # [B,K]
        D = self.dict()  # [K,77,768]
        out = torch.einsum('bk,kld->bld', weights, D)  # [B,77,768]
        return out, weights

# Instantiate
dict_K = 64  # try 32–128
soft_dict = SoftPromptDictionary(K=dict_K, seq_len=77, dim=768, init_anchors=anchors).to(device)

model = GeoDictMixer(
    location_encoder=geoclip_og.LocationEncoder(),
    dict_module=soft_dict,
    input_feat_dim=512,
    K=dict_K,
    temperature=0.7
).to(device)

# ----------------------------
# Losses tailored for token sequences
# ----------------------------
def _ln(x):  # per-token LayerNorm to match SD stats
    return F.layer_norm(x, (x.size(-1),))

def token_cosine_loss(pred, tgt):
    # cosine per token, averaged; both [B,77,768]
    p = F.normalize(pred, dim=-1)
    t = F.normalize(tgt, dim=-1)
    cos = (p * t).sum(dim=-1)  # [B,77]
    return (1.0 - cos).mean()

def token_mse_loss(pred, tgt):
    # MSE after per-token LayerNorm for scale/offset invariance
    return F.mse_loss(_ln(pred), _ln(tgt))

def gram_loss(pred, tgt):
    # Match token-token correlations (what cross-attn "sees")
    # G = X X^T / d  with X = tokens [B,L,D]
    B, L, D = pred.shape
    p = _ln(pred).transpose(1, 2)  # [B,D,L]
    t = _ln(tgt).transpose(1, 2)   # [B,D,L]
    Gp = (p.transpose(1, 2) @ p) / D  # [B,L,L]
    Gt = (t.transpose(1, 2) @ t) / D  # [B,L,L]
    return F.mse_loss(Gp, Gt)

def sequence_contrastive_loss(pred, tgt, temperature=0.03):
    # Light contrastive term on pooled embeddings to keep pairing tight
    p = F.normalize(pred.mean(dim=1), dim=-1)  # [B,768]
    t = F.normalize(tgt.mean(dim=1), dim=-1)   # [B,768]
    embeds = torch.cat([p, t], dim=0)          # [2B,768]
    sim = embeds @ embeds.t() / temperature
    B = p.size(0)
    mask = torch.eye(2*B, device=embeds.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -float('inf'))
    pos_idx = torch.cat([torch.arange(B, device=embeds.device)+B,
                         torch.arange(B, device=embeds.device)])
    exp_sim = torch.exp(sim)
    pos = exp_sim[torch.arange(2*B, device=embeds.device), pos_idx]
    denom = exp_sim.sum(dim=1)
    return (-torch.log(pos/denom)).mean()

def dict_diversity_loss(D):
    # Encourage different dictionary atoms (avoid collapse)
    K = D.size(0)
    flat = F.normalize(D.flatten(1), dim=1)  # [K, L*D]
    G = flat @ flat.t()                      # [K,K]
    I = torch.eye(K, device=D.device)
    return ((G - I)**2).mean()

def mixture_entropy_penalty(weights):
    # Encourage mild sparsity in mixtures (penalize high entropy)
    w = weights.clamp_min(1e-8)
    ent = -(w * w.log()).sum(dim=-1).mean()
    return ent

# ----------------------------
# Optimizer / Scheduler / AMP
# ----------------------------
optimizer = torch.optim.AdamW(
    list(model.parameters()) + list(soft_dict.parameters()),
    lr=5e-4, weight_decay=1e-4, eps=1e-6, betas=(0.9, 0.95)
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

# ----------------------------
# Training
# ----------------------------
epochs = 15
train_losses, val_losses = [], []

print(f"🚀 Training with batch_size={batch_size}, device={device}")

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    epoch_start = time.time()

    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
        for batch_idx, (texts, coords) in enumerate(pbar):
            if torch.cuda.is_available() and (batch_idx % 50 == 0):
                torch.cuda.empty_cache()

            coords = coords.to(device, non_blocking=True)
            tokens = tokenizer(
                list(texts),
                padding="max_length", truncation=True, max_length=77,
                return_tensors="pt"
            )
            tokens = {k: v.to(device, non_blocking=True) for k, v in tokens.items()}

            # Forward teacher (frozen)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    text_outputs = text_encoder(**tokens)
                    text_embed = text_outputs.last_hidden_state  # [B,77,768]

            # Forward student
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                pred_tokens, mix_w = model(coords)  # [B,77,768], [B,K]

                # Composite loss
                L_mse  = token_mse_loss(pred_tokens, text_embed)
                L_cos  = token_cosine_loss(pred_tokens, text_embed)
                L_gram = gram_loss(pred_tokens, text_embed)
                L_ctr  = sequence_contrastive_loss(pred_tokens, text_embed)

                # regularizers
                D = soft_dict()  # [K,77,768]
                L_div = dict_diversity_loss(D)
                L_ent = mixture_entropy_penalty(mix_w)

                loss = (
                    1.00 * L_mse +
                    0.25 * L_cos +
                    0.10 * L_gram +
                    0.25 * L_ctr +
                    0.01 * L_div +
                    0.01 * L_ent
                )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(soft_dict.parameters()), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() // 1024**2
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'mem': f'{memory_used}MB',
                                  'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'})
            else:
                pbar.set_postfix({'loss': f'{loss.item():.4f}',
                                  'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'})

    epoch_time = round(time.time() - epoch_start, 1)
    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Time: {epoch_time}s")

    # Validation
    model.eval()
    val_loss = 0.0
    token_cos_snapshot = 0.0
    with torch.no_grad():
        for texts, coords in val_loader:
            coords = coords.to(device, non_blocking=True)
            tokens = tokenizer(
                list(texts),
                padding="max_length", truncation=True, max_length=77,
                return_tensors="pt"
            )
            tokens = {k: v.to(device, non_blocking=True) for k, v in tokens.items()}

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                text_outputs = text_encoder(**tokens)
                text_embed = text_outputs.last_hidden_state  # [B,77,768]
                pred_tokens, mix_w = model(coords)

                L_mse  = token_mse_loss(pred_tokens, text_embed)
                L_cos  = token_cosine_loss(pred_tokens, text_embed)
                L_gram = gram_loss(pred_tokens, text_embed)
                L_ctr  = sequence_contrastive_loss(pred_tokens, text_embed)
                # Typically skip reg terms in val, but it’s okay to keep small ones:
                D = soft_dict()
                L_div = dict_diversity_loss(D)
                L_ent = mixture_entropy_penalty(mix_w)

                loss = (
                    1.00 * L_mse +
                    0.25 * L_cos +
                    0.10 * L_gram +
                    0.25 * L_ctr +
                    0.01 * L_div +
                    0.01 * L_ent
                )

                val_loss += loss.item()
                token_cos_snapshot += (1.0 - L_cos).item()  # avg token cosine

    avg_val_loss = val_loss / len(val_loader)
    avg_token_cos = token_cos_snapshot / len(val_loader)
    val_losses.append(avg_val_loss)
    print(f"Epoch {epoch+1} - Val Loss: {avg_val_loss:.4f} | 🔎 Avg token cosine: {avg_token_cos:.3f}")

    scheduler.step()

    # Save checkpoint periodically
    if (epoch + 1) % 5 == 0:
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'dict_state_dict': soft_dict.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        }
        ckpt_path = os.path.join(save_dir, f'softprompt_checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, ckpt_path)
        print(f"💾 Checkpoint saved to {ckpt_path}")

# Save final model states
final_bundle = {
    'model_state_dict': model.state_dict(),
    'dict_state_dict': soft_dict.state_dict(),
    'cfg': {'dict_K': dict_K}
}
torch.save(final_bundle, save_model_path)
print(f"✅ Saved model to {save_model_path}")

# ----------------------------
# Save Loss Plot
# ----------------------------
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss (Soft-Prompt)')
plt.legend()
plt.tight_layout()
plt.savefig(loss_plot_path)
print(f"✅ Saved loss plot to {loss_plot_path}")

# ----------------------------
# Quick test to verify learning
# ----------------------------
print("\n🧪 Running quick learning test...")
model.eval()
with torch.no_grad():
    test_batch = next(iter(val_loader))
    texts, coords = test_batch
    coords = coords.to(device)
    tokens = tokenizer(list(texts), padding="max_length", truncation=True, max_length=77, return_tensors="pt").to(device)
    text_outputs = text_encoder(**tokens)
    text_embed = text_outputs.last_hidden_state  # [B,77,768]
    pred_tokens, mix_w = model(coords)          # [B,77,768], [B,K]

    # Pooled similarity just for sanity
    geo_pooled = F.normalize(pred_tokens.mean(dim=1), dim=-1)  # [B,768]
    text_pooled = F.normalize(text_embed.mean(dim=1), dim=-1)  # [B,768]
    S = geo_pooled @ text_pooled.T
    diagonal_sim = torch.diag(S).mean().item()
    off_diagonal_sim = (S.sum() - torch.diag(S).sum()) / (S.numel() - len(S))
    print(f"✅ Avg diagonal similarity (correct pairs): {diagonal_sim:.4f}")
    print(f"✅ Avg off-diagonal similarity (incorrect pairs): {off_diagonal_sim:.4f}")
    print(f"✅ Learning ratio (diag/off): {(diagonal_sim/off_diagonal_sim).item():.2f}")

    # Token-level cosine snapshot
    pt = F.normalize(pred_tokens, dim=-1)
    tt = F.normalize(text_embed, dim=-1)
    avg_token_cos = (pt * tt).sum(-1).mean().item()
    print(f"🔎 Token-level cosine (avg): {avg_token_cos:.3f}")

print("\n✅ Training and testing complete!")

# ----------------------------
# (Optional) How to wire into SD at inference:
# ----------------------------
# from diffusers import StableDiffusionPipeline
# pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)
# pipe.enable_model_cpu_offload()
# model.eval()
# with torch.no_grad():
#     coords = torch.tensor([[40.7128, -74.0060]], device=device)  # NYC
#     geo_tokens, _ = model(coords)             # [1,77,768]
#     pooled = geo_tokens.mean(dim=1)           # [1,768]
# image = pipe(
#     prompt_embeds=geo_tokens,                 # [B,77,768]
#     pooled_prompt_embeds=pooled,              # [B,768]
#     num_inference_steps=30,
#     guidance_scale=7.5
# ).images[0]
# image.save("nyc_geo_style.png")
