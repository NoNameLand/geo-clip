import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
import geoclip
import os

# Load dummy geo encoder
class GeoEncoder(torch.nn.Module):
    def __init__(self, input_dim=2, output_dim=768):
        super().__init__()
        self.geo_prev = geoclip.LocationEncoder()
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, output_dim)
        )
        self.relu = torch.nn.ReLU()

    def forward(self, coords):
        with torch.no_grad():
            x = self.relu(self.geo_prev(coords))
        x = self.projector(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "alignment/models/dummy_model_cities.pt"
# prompt_embed_geo = prompt_embed  # Store geo prompt embedding
geo_encoder = GeoEncoder().to(device)
geo_encoder.load_state_dict(torch.load(model_path, map_location=device), strict=False)
geo_encoder.eval()

# Example coordinates (lat, lon)
coords = torch.tensor([[40.7128, 74.0060]], dtype=torch.float32).to(device)  # New york
with torch.no_grad():
    prompt_embed = geo_encoder(coords)  # [1, 768]
prompt_embed = prompt_embed.unsqueeze(1).repeat(1, 77, 1)  # [1, 77, 768]
# Load Stable Diffusion pipeline
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to(device)

from transformers import CLIPTokenizer, CLIPTextModel
clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
clip_text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
coord_prompt = "New York City, USA"
clip_tokens = clip_tokenizer([coord_prompt], return_tensors="pt", padding=True, truncation=True).to(device)
with torch.no_grad():
    prompt_embed_clip = clip_text_encoder(**clip_tokens).pooler_output  # [1, 768]
prompt_embed_clip = prompt_embed_clip.unsqueeze(1).repeat(1, 77, 1)  # [1, 77, 768]

# Load and preprocess input image
init_image = Image.open("venice.jpg").convert("RGB").resize((512, 512))

# Use the geo encoder output as the prompt embedding
output = pipe(
    prompt_embeds=prompt_embed,
    image=init_image,
    strength=0.4,
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
