import os, csv, math, itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionImg2ImgPipeline
from transformers import CLIPTokenizer, CLIPTextModel, CLIPModel, CLIPProcessor

# ----------------------------
# Repro
# ----------------------------
SEED = 42
torch.manual_seed(SEED)

# ----------------------------
# Geo encoder + our model
# ----------------------------
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import geoclip_og  # exposes geoclip_og.LocationEncoder()

class SoftPromptDictionary(nn.Module):
    def __init__(self, K=64, seq_len=77, dim=768):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(K, seq_len, dim) * 0.02)
    def forward(self):
        return self.tokens  # [K,77,768]

class GeoDictMixer(nn.Module):
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
            nn.Linear(input_feat_dim, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, K)
        )
    def forward(self, coords):
        with torch.no_grad():
            geo_feat = F.relu(self.location_encoder(coords))  # [B,512]
        logits = self.conditioning_net(geo_feat) / self.temperature
        weights = F.softmax(logits, dim=-1)                  # [B,K]
        D = self.dict()                                      # [K,77,768]
        out = torch.einsum('bk,kld->bld', weights, D)        # [B,77,768]
        return out, weights

# ----------------------------
# Paths / IO helpers
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bundle_path = "alignment/models/geo_softprompt_model_cities.pt"  # produced by training
assert os.path.exists(bundle_path), f"Missing bundle at {bundle_path}"

out_root = "results/grid_experiments"
os.makedirs(out_root, exist_ok=True)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

# ----------------------------
# Load our trained bundle
# ----------------------------
bundle = torch.load(bundle_path, map_location=device)
dict_K = bundle.get("cfg", {}).get("dict_K", 64)
soft_dict = SoftPromptDictionary(K=dict_K).to(device)
geo_model = GeoDictMixer(
    location_encoder=geoclip_og.LocationEncoder(),
    dict_module=soft_dict,
    input_feat_dim=512,
    K=dict_K,
    temperature=0.7,
).to(device)
soft_dict.load_state_dict(bundle["dict_state_dict"], strict=True)
geo_model.load_state_dict(bundle["model_state_dict"], strict=True)
geo_model.eval()

# ----------------------------
# Build SD img2img pipeline
# ----------------------------
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device)
pipe.set_progress_bar_config(disable=False)
tok = pipe.tokenizer
txt_enc = pipe.text_encoder

@torch.no_grad()
def make_negative_embeds(batch_size: int):
    max_len = tok.model_max_length  # 77
    tokens = tok([""] * batch_size, padding="max_length", truncation=True,
                 max_length=max_len, return_tensors="pt").to(device)
    out = txt_enc(**tokens)
    neg_prompt = out.last_hidden_state.to(dtype=pipe.unet.dtype)  # [B,77,768]
    neg_pooled = out.pooler_output.to(dtype=pipe.unet.dtype)      # [B,768]
    return neg_prompt, neg_pooled

@torch.no_grad()
def teacher_text_tokens(text_list):
    max_len = tok.model_max_length
    tokens = tok(text_list, padding="max_length", truncation=True,
                 max_length=max_len, return_tensors="pt").to(device)
    out = txt_enc(**tokens)
    prompt_embeds = out.last_hidden_state.to(dtype=pipe.unet.dtype)
    pooled = out.pooler_output.to(dtype=pipe.unet.dtype)
    return prompt_embeds, pooled

@torch.no_grad()
def geo_tokens_from_coords(coords_tensor):  # [B,2] lat,lon
    geo_tokens, _ = geo_model(coords_tensor.to(device))           # fp32
    pooled = geo_tokens.mean(dim=1)
    return geo_tokens.to(dtype=pipe.unet.dtype), pooled.to(dtype=pipe.unet.dtype)

# ----------------------------
# Metrics: CLIPScore + image-image sim + optional LPIPS
# ----------------------------
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

try:
    import lpips
    lpips_model = lpips.LPIPS(net='vgg').to(device).eval()
except Exception:
    lpips_model = None

@torch.no_grad()
def clipscore_image_text(pil_img, text):
    inputs = clip_proc(text=[text], images=pil_img, return_tensors="pt", padding=True).to(device)
    # normalize features
    out = clip_model(**inputs)
    img = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
    txt = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
    return torch.sum(img * txt, dim=-1).item()  # cosine

@torch.no_grad()
def clip_image_sim(img_a, img_b):
    t = clip_proc(images=[img_a, img_b], return_tensors="pt").to(device)
    feats = clip_model.get_image_features(**t)  # [2,dim]
    feats = feats / feats.norm(dim=-1, keepdim=True)
    sim = torch.sum(feats[0] * feats[1]).item()
    return sim

@torch.no_grad()
def lpips_distance(img_a, img_b):
    if lpips_model is None:
        return None
    # To tensor in [-1,1], CHW
    def to_t(img):
        x = torch.tensor(torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))).float()
        x = torch.from_numpy(np.array(img)).permute(2,0,1).float()  # HWC->CHW
        x = x / 127.5 - 1.0
        return x.unsqueeze(0).to(device)
    import numpy as np
    a = to_t(img_a)
    b = to_t(img_b)
    d = lpips_model(a, b).item()
    return d

# ----------------------------
# Experiment config
# ----------------------------
sources = {
    "tel_aviv": "tel_aviv.jpg",
    "venice":   "venice.jpg",
}
targets = {
    "paris_france":    {"text": "Paris, France",    "coords": (48.8566,   2.3522)},
    "shanghai_china":  {"text": "Shanghai, China",  "coords": (31.2304, 121.4737)},
}
strength_list  = [0.25, 0.40, 0.55, 0.70]
guidance_list  = [2.0, 4.0, 6.0, 8.0]
methods = ["geo", "clip", "text"]  # ours, teacher tokens, plain text

# seed per run for determinism
def make_generator():
    g = torch.Generator(device=device)
    g.manual_seed(SEED)
    return g

# ----------------------------
# Core runner
# ----------------------------
def run_single(pipe, init_image, method, target_text, target_coords, strength, guidance):
    neg_e, neg_p = make_negative_embeds(1)
    generator = make_generator()
    if method == "geo":
        c = torch.tensor([list(target_coords)], dtype=torch.float32, device=device)  # [1,2]
        pe, pp = geo_tokens_from_coords(c)
        out = pipe(
            prompt_embeds=pe,
            pooled_prompt_embeds=pp,
            negative_prompt_embeds=neg_e,
            negative_pooled_prompt_embeds=neg_p,
            image=init_image,
            strength=strength,
            guidance_scale=guidance,
            num_inference_steps=30,
            generator=generator,
        )
    elif method == "clip":
        pe, pp = teacher_text_tokens([target_text])
        out = pipe(
            prompt_embeds=pe,
            pooled_prompt_embeds=pp,
            negative_prompt_embeds=neg_e,
            negative_pooled_prompt_embeds=neg_p,
            image=init_image,
            strength=strength,
            guidance_scale=guidance,
            num_inference_steps=30,
            generator=generator,
        )
    else:  # "text"
        out = pipe(
            prompt=target_text,
            image=init_image,
            strength=strength,
            guidance_scale=guidance,
            num_inference_steps=30,
            generator=generator,
        )
    return out.images[0]

# ----------------------------
# Compose a grid figure (rows=strength, cols=guidance)
# ----------------------------
def save_grid(grid_map, strengths, guidances, save_path, title=""):
    import matplotlib.pyplot as plt
    fig_w = 2.5 * len(guidances)
    fig_h = 2.5 * len(strengths) + (0.4 if title else 0.0)
    fig, axes = plt.subplots(len(strengths), len(guidances), figsize=(fig_w, fig_h))
    if len(strengths) == 1 and len(guidances) == 1:
        axes = [[axes]]
    elif len(strengths) == 1:
        axes = [axes]
    elif len(guidances) == 1:
        axes = [[ax] for ax in axes]

    for i, s in enumerate(strengths):
        for j, g in enumerate(guidances):
            ax = axes[i][j]
            ax.imshow(grid_map[(s,g)])
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(f"guidance={g}", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"strength={s}", fontsize=10)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

# ----------------------------
# Main sweep
# ----------------------------
def main():
    import pandas as pd
    results = []

    for src_key, src_path in sources.items():
        src_img_path = src_path
        assert os.path.exists(src_img_path), f"Missing source image {src_img_path}"
        src_img = Image.open(src_img_path).convert("RGB").resize((512,512))

        for tgt_key, tgt in targets.items():
            tgt_text = tgt["text"]
            tgt_coords = tgt["coords"]

            pair_dir = ensure_dir(os.path.join(out_root, f"{src_key}_to_{tgt_key}"))
            single_dir = ensure_dir(os.path.join(pair_dir, "singles"))
            grids_dir  = ensure_dir(os.path.join(pair_dir, "grids"))

            # we’ll keep per-method grids here
            per_method_grids = {}

            # For method-pair comparison (geo vs clip at same (s,g)), keep cache
            cache_geo  = {}
            cache_clip = {}

            for method in methods:
                grid_images = {}
                method_dir = ensure_dir(os.path.join(single_dir, method))
                pbar = tqdm(list(itertools.product(strength_list, guidance_list)),
                            desc=f"{src_key}->{tgt_key} [{method}]")
                for (s, g) in pbar:
                    img_out = run_single(pipe, src_img, method, tgt_text, tgt_coords, s, g)
                    grid_images[(s,g)] = img_out

                    # Save single image
                    fn = f"{src_key}_to_{tgt_key}_{method}_s{s:.2f}_g{g:.2f}".replace('.', 'p')
                    fn += ".png"
                    img_out.save(os.path.join(method_dir, fn))

                    # Metrics
                    clips = clipscore_image_text(img_out, tgt_text)                      # ↑ style match
                    cims  = clip_image_sim(src_img, img_out)                             # ↑ content preservation
                    lpd   = lpips_distance(src_img, img_out) if lpips_model else None    # ↓ perceptual diff (optional)

                    results.append({
                        "source": src_key,
                        "target": tgt_key,
                        "method": method,
                        "strength": s,
                        "guidance": g,
                        "clipscore_to_target": clips,
                        "clip_imgsim_to_source": cims,
                        "lpips_to_source": lpd,
                    })

                    if method == "geo":
                        cache_geo[(s,g)]  = img_out
                    elif method == "clip":
                        cache_clip[(s,g)] = img_out

                # Save grid figure for this method
                grid_path = os.path.join(grids_dir, f"grid_{method}_{src_key}_to_{tgt_key}.png")
                title = f"{src_key} → {tgt_key}  ({method})"
                save_grid(grid_images, strength_list, guidance_list, grid_path, title=title)
                per_method_grids[method] = grid_path

            # Geo↔CLIP pairwise image-sim at matched (s,g)
            for (s, g) in itertools.product(strength_list, guidance_list):
                if (s,g) in cache_geo and (s,g) in cache_clip:
                    sim_geo_clip = clip_image_sim(cache_geo[(s,g)], cache_clip[(s,g)])
                    results.append({
                        "source": src_key,
                        "target": tgt_key,
                        "method": "geo_vs_clip_pair",
                        "strength": s,
                        "guidance": g,
                        "clipscore_to_target": None,
                        "clip_imgsim_to_source": None,
                        "lpips_to_source": None,
                        "clip_imgsim_geo_vs_clip": sim_geo_clip
                    })

            # Optional: print where the grids are for this pair
            print(f"[Saved grids] {src_key}→{tgt_key}:")
            for m, p in per_method_grids.items():
                print(f"  - {m}: {p}")

    # Save metrics CSV + a LaTeX table snippet
    df = pd.DataFrame(results)
    csv_path = os.path.join(out_root, "metrics_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"[Saved metrics] {csv_path}")

    # Simple LaTeX table: average over settings per (source,target,method)
    agg = (df
           .groupby(["source","target","method"], dropna=False)
           .agg({
               "clipscore_to_target":"mean",
               "clip_imgsim_to_source":"mean",
               "lpips_to_source":"mean",
               "clip_imgsim_geo_vs_clip":"mean"
           })
           .reset_index())
    tex = agg.to_latex(index=False, float_format="%.3f", na_rep="--",
                       caption="Average metrics across strength/guidance grid.",
                       label="tab:grid_metrics")
    with open(os.path.join(out_root, "metrics_summary_table.tex"), "w") as f:
        f.write(tex)
    print(f"[Saved LaTeX table] {os.path.join(out_root, 'metrics_summary_table.tex')}")

if __name__ == "__main__":
    main()
