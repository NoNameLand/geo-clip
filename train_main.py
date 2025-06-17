import sys
sys.path.append('geoclip/model/')

# Call dataloaders
from geoclip.train import dataloader
from geoclip.train import train_decoder
from geoclip.model import GeoCLIP
import torch
from torch import optim
from torch.utils.data import DataLoader
import os
import argparse
def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    traindecoder = train_decoder.VaeDecoder().to(device)
    model = traindecoder
    
    # Load feature extractor
    geo_clip = GeoCLIP(
        from_pretrained=True,
        queue_size=4096
    ).to(device)
    geo_encoder = geo_clip.location_encoder
    
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Create dataloaders
    dataload = dataloader.GeoDataLoader(
        dataset_file="data/shard_0.csv",
        dataset_folder="data/images/",
        transform=dataloader.img_train_transform()
    )
    print("[INFO] Dataset loaded with {} images.".format(len(dataload)))
    # Training loop
    for epoch in range(args.epochs):
        train_decoder.train_decoder(dataload,geo_encoder, model, optimizer, epoch, args.batch_size, device)
        
        # Save model checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"model_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model saved to {checkpoint_path}") 
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GeoCLIP model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.006, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer")
    parser.add_argument("--save_interval", type=int, default=5, help="Interval to save model checkpoints")
    parser.add_argument("--checkpoint_dir", type=str, default="results/models/checkpoints/", help="Directory to save model checkpoints")
    
    args = parser.parse_args()
    
    main(args)