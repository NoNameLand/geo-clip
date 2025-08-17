import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
import geoclip
import os
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel
import glob
from tqdm import tqdm
import json
# Load dummy geo encoder
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "alignment/models/mixture_checkpoint_epoch_10.pth"
# prompt_embed_geo = prompt_embed  # Store geo prompt embedding
geo_encoder = GeoPrompt77().to(device)
geo_encoder.load_state_dict(torch.load(model_path, map_location=device), strict=False)
geo_encoder.eval()

# Example coordinates (lat, lon)
coords = torch.tensor([[51.5072,  0.1276]], dtype=torch.float32).to(device)  # London
with torch.no_grad():
    prompt_embed = geo_encoder(coords)  # [1, 768]
if prompt_embed.dim() == 2:
    prompt_embed = prompt_embed.unsqueeze(1)  # Ensure shape is [1, 1, 768]
if prompt_embed.dim() == 3 and prompt_embed.shape[1] == 1:
    # If already in [1, 1, 768] shape, repeat to match sequence length
    # [1, 77, 768]  
    prompt_embed = prompt_embed.repeat(1, 77, 1)  # [1, 77, 768]

# Load Stable Diffusion pipeline
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to(device)

from transformers import CLIPTokenizer, CLIPTextModel
clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
clip_text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
coord_prompt = "London, England"
clip_tokens = clip_tokenizer([coord_prompt], return_tensors="pt", padding=True, truncation=True).to(device)
with torch.no_grad():
    prompt_embed_clip = clip_text_encoder(**clip_tokens).pooler_output  # [1, 768]
prompt_embed_clip = prompt_embed_clip.unsqueeze(1).repeat(1, 77, 1)  # [1, 77, 768]

# Load and preprocess input image
init_image = Image.open("tokyo.jpg").convert("RGB").resize((512, 512))

# Use the geo encoder output as the prompt embedding
output = pipe(
    prompt_embeds=prompt_embed,
    image=init_image,
    strength=0.45,
    guidance_scale=4
)

output_clip = pipe(
    prompt_embeds=prompt_embed_clip,
    image=init_image,
    strength=0.4,
    guidance_scale=4
)
# Save result
os.makedirs("results/diff_model_test", exist_ok=True)
output.images[0].save("results/diff_model_test/geo_embed_output.png")
print("✅ Saved image using geo encoder embedding as prompt.")

output_clip.images[0].save("results/diff_model_test/clip_embed_output.png")
print("✅ Saved image using CLIP encoder embedding as prompt.")

# --- Run with regular text prompt (no custom embedding) ---
output_text = pipe(
    prompt=coord_prompt,  # Use the same text as for CLIP
    image=init_image,
    strength=0.4,
    guidance_scale=4
)
output_text.images[0].save("results/diff_model_test/text_prompt_output.png")
print("✅ Saved image using regular text prompt.")
