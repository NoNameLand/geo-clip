# eval_softprompt_alignment.py
import os, csv, json, math, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from transformers import CLIPTokenizer, CLIPTextModel

# ---------- Dataset ----------
class GeoTextDataset(Dataset):
    def __init__(self, path):
        if path.endswith(".csv"):
            self.data = []
            with open(path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.data.append({"text": row["text"], "lat": row["lat"], "lon": row["lon"]})
        else:
            with open(path, "r") as f:
                self.data = json.load(f)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        it = self.data[idx]
        return it["text"], torch.tensor([float(it["lat"]), float(it["lon"])], dtype=torch.float32)

# ---------- Model pieces (match training) ----------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import geoclip  # geoclip.LocationEncoder()

class SoftPromptDictionary(nn.Module):
    def __init__(self, K=64, seq_len=77, dim=768):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(K, seq_len, dim) * 0.02)
    def forward(self): return self.tokens

class GeoDictMixer(nn.Module):
    def __init__(self, location_encoder, dict_module: SoftPromptDictionary, input_feat_dim=512, K=64, temperature=0.7):
        super().__init__()
        self.location_encoder = location_encoder
        for p in self.location_encoder.parameters(): p.requires_grad = False
        self.dict = dict_module; self.K = K; self.temperature = temperature
        self.conditioning_net = nn.Sequential(
            nn.Linear(input_feat_dim, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, K)
        )
    def forward(self, coords):
        with torch.no_grad(): z = F.relu(self.location_encoder(coords))  # [B,512]
        logits = self.conditioning_net(z) / self.temperature
        w = F.softmax(logits, dim=-1)                 # [B,K]
        D = self.dict()                                # [K,77,768]
        tokens = torch.einsum('bk,kld->bld', w, D)     # [B,77,768]
        return tokens, w

# ---------- Paths ----------
bundle_path = "alignment/models/geo_softprompt_model_cities.pt"
test_path   = "alignment/dataset/generated_dataset/val_split.csv"
out_dir     = "alignment/results/softprompt_eval"
os.makedirs(out_dir, exist_ok=True)

# ---------- Device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---------- Teacher (frozen CLIP text) ----------
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_enc  = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()

# ---------- Data ----------
batch_size = 64 if torch.cuda.is_available() else 32
dataset = GeoTextDataset(test_path)
assert len(dataset) > 0, f"No rows in {test_path}"
loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# ---------- Load trained + untrained ----------
assert os.path.exists(bundle_path), f"Missing checkpoint: {bundle_path}"
bundle = torch.load(bundle_path, map_location=device)
dict_K = bundle.get("cfg", {}).get("dict_K", 64)

soft_tr = SoftPromptDictionary(K=dict_K).to(device)
geo_tr  = GeoDictMixer(geoclip.LocationEncoder(), soft_tr, K=dict_K, temperature=0.7).to(device)
soft_tr.load_state_dict(bundle["dict_state_dict"], strict=True)
geo_tr.load_state_dict(bundle["model_state_dict"], strict=True)
geo_tr.eval()

soft_un = SoftPromptDictionary(K=dict_K).to(device)
geo_un  = GeoDictMixer(geoclip.LocationEncoder(), soft_un, K=dict_K, temperature=0.7).to(device)
geo_un.eval()

# ---------- Helpers ----------
def layer_norm_tokens(x): return F.layer_norm(x, (x.size(-1),))

@torch.no_grad()
def teacher_tokens(texts):
    toks = tokenizer(list(texts), padding="max_length", truncation=True, max_length=77, return_tensors="pt").to(device)
    return text_enc(**toks).last_hidden_state  # [B,77,768]

def batch_metrics(pred_tokens, tgt_tokens):
    p = F.normalize(pred_tokens, dim=-1); t = F.normalize(tgt_tokens, dim=-1)
    token_cos  = (p * t).sum(dim=-1).mean(dim=-1)                       # [B]
    pooled_cos = (F.normalize(pred_tokens.mean(dim=1), dim=-1) *
                  F.normalize(tgt_tokens.mean(dim=1),   dim=-1)).sum(-1) # [B]
    p_ln = layer_norm_tokens(pred_tokens); t_ln = layer_norm_tokens(tgt_tokens); D = p_ln.size(-1)
    gram_err = ((p_ln @ p_ln.transpose(1,2) / D - t_ln @ t_ln.transpose(1,2) / D) ** 2).mean(dim=(1,2))
    return token_cos.cpu().detach().numpy(), pooled_cos.cpu().detach().numpy(), gram_err.cpu().detach().numpy()

@torch.no_grad()
def pooled_sim_matrix(pred_tokens, tgt_tokens):
    p = F.normalize(pred_tokens.mean(dim=1), dim=-1)
    t = F.normalize(tgt_tokens.mean(dim=1),  dim=-1)
    return (p @ t.T).cpu().numpy()

def np_entropy(W, eps=1e-12):  # W: [N,K]
    Wc = np.clip(W, eps, 1.0)
    return (- (Wc * np.log(Wc)).sum(axis=1))

# ---------- Accumulators ----------
token_cos_un, token_cos_tr = [], []
pooled_cos_un, pooled_cos_tr = [], []
gram_err_un, gram_err_tr = [], []
W_all_un, W_all_tr = [], []
S_un, S_tr = None, None
gram_triplet = None  # (teacher, pred, diff)

# ---------- Loop ----------
for bidx, (texts, coords) in enumerate(loader):
    coords = coords.to(device)
    H = teacher_tokens(texts)  # [B,77,768]

    Hu, Wu = geo_un(coords)
    tu, pu, gu = batch_metrics(Hu, H)
    token_cos_un.append(tu); pooled_cos_un.append(pu); gram_err_un.append(gu)
    W_all_un.append(Wu.cpu().detach().numpy())

    Ht, Wt = geo_tr(coords)
    tt, pt, gt = batch_metrics(Ht, H)
    token_cos_tr.append(tt); pooled_cos_tr.append(pt); gram_err_tr.append(gt)
    W_all_tr.append(Wt.cpu().detach().numpy())

    if S_un is None:
        S_un = pooled_sim_matrix(Hu, H)
        S_tr = pooled_sim_matrix(Ht, H)

    if gram_triplet is None and Ht.size(0) > 0:
        t_ln = layer_norm_tokens(H[0:1]); p_ln = layer_norm_tokens(Ht[0:1]); D = t_ln.size(-1)
        Gt = (t_ln @ t_ln.transpose(1,2) / D)[0].cpu().detach().numpy()
        Gp = (p_ln @ p_ln.transpose(1,2) / D)[0].cpu().detach().numpy()
        Gd = np.clip(np.abs(Gp - Gt), 0, 1.0)
        gram_triplet = (Gt, Gp, Gd)

# concat
token_cos_un = np.concatenate(token_cos_un); token_cos_tr = np.concatenate(token_cos_tr)
pooled_cos_un = np.concatenate(pooled_cos_un); pooled_cos_tr = np.concatenate(pooled_cos_tr)
gram_err_un = np.concatenate(gram_err_un); gram_err_tr = np.concatenate(gram_err_tr)
W_all_un = np.vstack(W_all_un); W_all_tr = np.vstack(W_all_tr)
ent_un = np_entropy(W_all_un); ent_tr = np_entropy(W_all_tr)

# ---------- Console summary ----------
def summary(name, tok, pool, gram, ent):
    print(f"\n[{name}]")
    print(f"  Token cosine    : {tok.mean():.3f} ± {tok.std():.3f}")
    print(f"  Pooled cosine   : {pool.mean():.3f} ± {pool.std():.3f}")
    print(f"  Gram error      : {gram.mean():.5f} (↓)")
    print(f"  Mixture entropy : {ent.mean():.3f} ± {ent.std():.3f}  (lower→sparser)")
summary("Untrained", token_cos_un, pooled_cos_un, gram_err_un, ent_un)
summary("Trained",   token_cos_tr, pooled_cos_tr, gram_err_tr, ent_tr)

def diag_off(S):
    B = S.shape[0]
    d = np.mean(np.diag(S))
    o = (np.sum(S) - np.sum(np.diag(S))) / (B*B - B)
    return d, o
if S_un is not None:
    d_un, o_un = diag_off(S_un); d_tr, o_tr = diag_off(S_tr)
    print(f"\n[Retrieval (pooled)]  diag/off — untrained: {d_un:.3f}/{o_un:.3f}   trained: {d_tr:.3f}/{o_tr:.3f}")

# ---------- Plot helpers ----------
def savefig(path): plt.savefig(path, dpi=220, bbox_inches="tight"); plt.close()

# 1) Token-level cosine
plt.figure(figsize=(8,4))
plt.hist(token_cos_un, bins=30, alpha=0.55, label="Before (untrained)", density=True)
plt.hist(token_cos_tr, bins=30, alpha=0.55, label="After (trained)",    density=True)
plt.xlabel("Average per-token cosine"); plt.ylabel("Density"); plt.title("Token-level Alignment")
plt.grid(True, alpha=0.3); plt.legend()
savefig(os.path.join(out_dir, "token_cosine_hist.png"))

# 2) Pooled cosine
plt.figure(figsize=(8,4))
plt.hist(pooled_cos_un, bins=30, alpha=0.55, label="Before (untrained)", density=True)
plt.hist(pooled_cos_tr, bins=30, alpha=0.55, label="After (trained)",    density=True)
plt.xlabel("Pooled cosine (mean over tokens)"); plt.ylabel("Density"); plt.title("Pooled Alignment")
plt.grid(True, alpha=0.3); plt.legend()
savefig(os.path.join(out_dir, "pooled_cosine_hist.png"))

# 3) Gram error
plt.figure(figsize=(8,4))
plt.hist(gram_err_un, bins=30, alpha=0.55, label="Before (untrained)", density=True)
plt.hist(gram_err_tr, bins=30, alpha=0.55, label="After (trained)",    density=True)
plt.xlabel("Gram error (LN tokens)"); plt.ylabel("Density"); plt.title("Token–Token Structure Match (lower is better)")
plt.grid(True, alpha=0.3); plt.legend()
savefig(os.path.join(out_dir, "gram_error_hist.png"))

# 4) Mixture entropy (fix scaling + visibility)
plt.figure(figsize=(8,4))

bins = np.linspace(0, 5, 50)

plt.hist(ent_un, bins=bins, alpha=0.55, label="Before (untrained)", density=True)
plt.hist(ent_tr, bins=bins, alpha=0.55, label="After (trained)", density=True)

plt.xlabel(r"Mixture entropy $H(\alpha)$")
plt.ylabel("Density")
plt.title("Mixture Sparsity (lower → sparser)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(0, 5)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "mixture_entropy_hist.png"))

# 5) Retrieval similarity matrices with clean colorbar
if S_un is not None:
    fig, ax = plt.subplots(1,2, figsize=(9.8,4), constrained_layout=True)
    im0 = ax[0].imshow(S_un, vmin=-1, vmax=1, cmap="coolwarm", interpolation="nearest")
    ax[0].set_title("Before"); ax[0].set_xlabel("Teacher index"); ax[0].set_ylabel("Geo index")
    im1 = ax[1].imshow(S_tr, vmin=-1, vmax=1, cmap="coolwarm", interpolation="nearest")
    ax[1].set_title("After");  ax[1].set_xlabel("Teacher index"); ax[1].set_ylabel("Geo index")
    # one shared colorbar to the right, not overlapping
    cbar = fig.colorbar(im1, ax=ax, location="right", shrink=0.88, pad=0.02)
    cbar.set_label("Cosine similarity")
    savefig(os.path.join(out_dir, "retrieval_sim_matrix.png"))

# 6) Gram heatmaps (teacher vs geo vs |diff|)
if gram_triplet is not None:
    Gt, Gp, Gd = gram_triplet
    fig, ax = plt.subplots(1,3, figsize=(12,3.8), constrained_layout=True)
    ax[0].imshow(Gt, cmap="viridis"); ax[0].set_title("Teacher Gram")
    ax[1].imshow(Gp, cmap="viridis"); ax[1].set_title("Geo Gram")
    ax[2].imshow(Gd, cmap="magma");  ax[2].set_title("|Diff| (clipped)")
    for a in ax: a.set_xticks([]); a.set_yticks([])
    savefig(os.path.join(out_dir, "gram_heatmaps_sample0.png"))

# 7) Average top-k mass curve
def topk_mass(W, Kmax=10):
    Wsort = -np.sort(-W, axis=1)
    ks = np.arange(1, min(Kmax, Wsort.shape[1])+1)
    ys = [Wsort[:, :k].sum(axis=1).mean() for k in ks]
    return ks, np.array(ys)

ks, y_un = topk_mass(W_all_un, Kmax=10)
_,  y_tr = topk_mass(W_all_tr, Kmax=10)
plt.figure(figsize=(6.2,4))
plt.plot(ks, y_un, "o--", label="Before (untrained)")
plt.plot(ks, y_tr, "o--", label="After (trained)")
plt.xlabel("Top-k atoms"); plt.ylabel("Avg cumulative mass")
plt.title("Mixture Concentration (higher is sparser)")
plt.ylim(0, 1.01); plt.grid(True, alpha=0.3); plt.legend()
savefig(os.path.join(out_dir, "mixture_topk_mass.png"))

print(f"\n✅ Saved evaluation plots to: {out_dir}")
