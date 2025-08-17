import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from transformers import CLIPTokenizer, CLIPTextModel
from tqdm import tqdm

# ----------------------------
# Geo encoder import
# ----------------------------
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import geoclip  # must expose geoclip.LocationEncoder()

# ----------------------------
# New model defs (must match training)
# ----------------------------
class SoftPromptDictionary(nn.Module):
    def __init__(self, K=64, seq_len=77, dim=768):
        super().__init__()
        # Will load weights from checkpoint; init small otherwise
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
            nn.Linear(input_feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, K)
        )
    def forward(self, coords):
        # coords: [B,2] as (lat, lon)
        with torch.no_grad():
            geo_feat = F.relu(self.location_encoder(coords))  # [B,512]
        logits = self.conditioning_net(geo_feat) / self.temperature  # [B,K]
        weights = F.softmax(logits, dim=-1)                           # [B,K]
        D = self.dict()                                               # [K,77,768]
        out = torch.einsum('bk,kld->bld', weights, D)                 # [B,77,768]
        return out, weights

# ----------------------------
# Load saved geo model
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bundle_path = "alignment/models/geo_softprompt_model_cities.pt"  # from training script
assert os.path.exists(bundle_path), f"Missing bundle at {bundle_path}"

bundle = torch.load(bundle_path, map_location=device)
dict_K = bundle.get("cfg", {}).get("dict_K", 64)

soft_dict = SoftPromptDictionary(K=dict_K, seq_len=77, dim=768).to(device)
geo_model = GeoDictMixer(
    location_encoder=geoclip.LocationEncoder(),
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

# convenience handles to the pipeline's tokenizer/encoder (matches the pipeline exactly)
tok = pipe.tokenizer
txt_enc = pipe.text_encoder

# helper: make negative (unconditional) embeddings that match shapes
@torch.no_grad()
def make_negative_embeds(batch_size: int):
    max_len = tok.model_max_length  # 77 for SD1.5
    tokens = tok([""] * batch_size, padding="max_length", truncation=True,
                 max_length=max_len, return_tensors="pt").to(device)
    out = txt_enc(**tokens)
    neg_prompt = out.last_hidden_state           # [B,77,768]
    neg_pooled = out.pooler_output               # [B,768]
    # cast to pipeline dtype
    neg_prompt = neg_prompt.to(dtype=pipe.unet.dtype)
    neg_pooled = neg_pooled.to(dtype=pipe.unet.dtype)
    return neg_prompt, neg_pooled

# helper: get CLIP teacher tokens for a text (baseline)
@torch.no_grad()
def teacher_text_tokens(text_list):
    max_len = tok.model_max_length
    tokens = tok(text_list, padding="max_length", truncation=True,
                 max_length=max_len, return_tensors="pt").to(device)
    out = txt_enc(**tokens)
    prompt_embeds = out.last_hidden_state         # [B,77,768]
    pooled = out.pooler_output                    # [B,768]
    return prompt_embeds.to(dtype=pipe.unet.dtype), pooled.to(dtype=pipe.unet.dtype)

# helper: geo tokens from coords
@torch.no_grad()
def geo_tokens_from_coords(coords_tensor):  # coords_tensor: [B,2] (lat, lon)
    geo_tokens, _ = geo_model(coords_tensor.to(device))  # [B,77,768] (fp32)
    pooled = geo_tokens.mean(dim=1)                      # [B,768] simple pooled
    # cast to pipeline dtype
    return geo_tokens.to(dtype=pipe.unet.dtype), pooled.to(dtype=pipe.unet.dtype)

# ----------------------------
# Inputs
# ----------------------------
coord_prompt = "New York City, USA"
coords = torch.tensor([[40.7128, -74.0060]], dtype=torch.float32, device=device)  # New York City

os.makedirs("results/diff_model_test", exist_ok=True)
init_image = Image.open("tel_aviv.jpg").convert("RGB").resize((512, 512))

# ----------------------------
# 1) Geo-conditioned img2img
# ----------------------------
geo_prompt_embeds, geo_pooled = geo_tokens_from_coords(coords)   # [1,77,768], [1,768]
neg_embeds, neg_pooled = make_negative_embeds(batch_size=geo_prompt_embeds.shape[0])

out_geo = pipe(
    prompt_embeds=geo_prompt_embeds,
    pooled_prompt_embeds=geo_pooled,
    negative_prompt_embeds=neg_embeds,
    negative_pooled_prompt_embeds=neg_pooled,
    image=init_image,
    strength=0.4,
    guidance_scale=4.5,
    num_inference_steps=30,
)
out_geo.images[0].save("results/diff_model_test/geo_embed_output.png")
print("✅ Saved image using GEO tokens as prompt embeds.")

# ----------------------------
# 2) CLIP teacher tokens baseline (correct tokens path)
# ----------------------------
clip_prompt_embeds, clip_pooled = teacher_text_tokens([coord_prompt])  # [1,77,768], [1,768]
neg_embeds2, neg_pooled2 = make_negative_embeds(batch_size=1)

out_clip = pipe(
    prompt_embeds=clip_prompt_embeds,
    pooled_prompt_embeds=clip_pooled,
    negative_prompt_embeds=neg_embeds2,
    negative_pooled_prompt_embeds=neg_pooled2,
    image=init_image,
    strength=0.40,
    guidance_scale=4.5,
    num_inference_steps=30,
)
out_clip.images[0].save("results/diff_model_test/clip_embed_output.png")
print("✅ Saved image using CLIP teacher tokens as prompt embeds.")

# ----------------------------
# 3) Plain text prompt baseline
# ----------------------------
out_text = pipe(
    prompt=coord_prompt,
    image=init_image,
    strength=0.40,
    guidance_scale=4.5,
    num_inference_steps=30,
)
out_text.images[0].save("results/diff_model_test/text_prompt_output.png")
print("✅ Saved image using plain text prompt.")
