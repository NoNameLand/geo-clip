import json
import torch
from tqdm import tqdm
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import csv

# === CONFIG ===
DATASET_PATH = "data/data_0/shard_0.csv"       # Input Geo-CLIP dataset
OUTPUT_PATH = "alignment/dataset/generated_dataset/geo_clip_with_text.json"  # Output with captions
N = 10000                                  # How many samples to process
IMGS_PARENT_DIR = "data/data_0/"  # Directory where images are stored

# === Load BLIP model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# === Captioning function ===
def generate_caption(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Could not open image {image_path}: {e}")
        return None
    inputs = processor(image, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=50)
    return processor.decode(out[0], skip_special_tokens=True)

# === Load dataset ===
with open(DATASET_PATH) as f:
    reader = csv.DictReader(f)
    raw_data = [row for row in reader if row['IMG_FILE'] and row['LAT'] and row['LON']]

processed_data = []

# === Iterate and caption ===
for item in tqdm(raw_data[:N], desc="Generating captions"):
    lat = item["LAT"]
    lon = item["LON"]
    image_path = item["IMG_FILE"]

    caption = generate_caption(IMGS_PARENT_DIR + image_path)
    if caption:
        processed_data.append({
            "lat": lat,
            "lon": lon,
            "image_path": image_path,
            "text": caption
        })

# === Save to JSON ===
with open(OUTPUT_PATH, "w") as f:
    json.dump(processed_data, f, indent=2)

print(f"\n✅ Saved {len(processed_data)} entries to {OUTPUT_PATH}")
