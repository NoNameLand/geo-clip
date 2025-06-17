from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import torch

# Load pre-trained pipeline
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")

# Load and preprocess input image
init_image = Image.open("example_input.jpg").convert("RGB").resize((512, 512))

# Prompt to guide the transformation
prompt = "a fantasy version of this place"

# Strength controls how much to transform (0.0 = identical, 1.0 = new image)
strength = 0.75
guidance_scale = 7.5  # How strongly to follow the prompt

# Generate the image
output = pipe(prompt=prompt, image=init_image, strength=strength, guidance_scale=guidance_scale)

# Show or save
output.image[0].save("results/diff_model_test/stylized_output.png")
# output.image[0].show()