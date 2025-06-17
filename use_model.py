# Imporst
import torch
from geoclip.model import GeoCLIP
from geoclip.train import train_decoder
from geoclip.train import dataloader
import os
import matplotlib.pyplot as plt

# Load model
model_path = "results/models/checkpoints/model_epoch_10.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load geo_clip model for gps_encoder
geo_clip = GeoCLIP(
    from_pretrained=True,
    queue_size=4096
).to(device)
geo_encoder = geo_clip.location_encoder

# Load trained decoder model
traindecoder = train_decoder.VaeDecoder().to(device)
traindecoder.load_state_dict(torch.load(model_path, map_location=device))

# Create dataloader
dataload = dataloader.GeoDataLoader(
    dataset_file="data/shard_0.csv",
    dataset_folder="data/images/",
    transform=dataloader.img_train_transform()
)
print("[INFO] Dataset loaded with {} images.".format(len(dataload)))

# Set model to evaluation mode
traindecoder.eval()
# Evaluate the model
with torch.no_grad():
    for i, (images, locations) in enumerate(dataload):
        images = images.to(device)
        locations = torch.tensor(locations).to(device)
        locations = locations.unsqueeze(1).T  # Ensure locations is of shape (batch_size, 2)
        
        # Forward pass through the decoder
        reconstructed_images = traindecoder(geo_encoder(locations))
        
        plt.imshow(reconstructed_images[0].cpu().permute(1, 2, 0).numpy())
        plt.title(f"Reconstructed Image {i + 1}")
        plt.savefig(f"results/reconstructed/reconstructed_image_{i + 1}.png")
        
        if i % 10 == 0:  # Print every 10 batches
            print(f"Processed batch {i + 1}/{len(dataload)}")
