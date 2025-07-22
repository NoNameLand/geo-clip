from diffusers import StableDiffusionImageVariationPipeline
import torch
from clip import clip
from PIL import Image

device = "cuda"
# 1. Load CLIP image embedder
clip_model, preprocess = clip.load("ViT-L/14", device)
img = preprocess(Image.open("example_input.jpg")).unsqueeze(0).to(device)
with torch.no_grad():
    img_embed = clip_model.encode_image(img) # R ^ 1x512

# 2. Use image-variation pipeline
pipe = StableDiffusionImageVariationPipeline.from_pretrained(
    "lambdalabs/sd-image-variations-diffusers", torch_dtype=torch.float16
).to(device)

output = pipe(image_embed=img_embed, guidance_scale=7.5)
output.images[0].save("variation.png")