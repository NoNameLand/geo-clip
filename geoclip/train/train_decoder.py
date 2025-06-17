import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from geoclip import LocationEncoder
import sys
sys.path.append('geoclip/model/') 
from image_decoder import VaeDecoder

def train_decoder(train_dataloader, constant_model, train_model, optimizer, epoch, batch_size, device, scheduler=None, criterion=nn.CrossEntropyLoss()):
    print("Starting Epoch", epoch)

    bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))

    targets_img_gps = torch.Tensor([i for i in range(batch_size)]).long().to(device)
    
    for i ,(imgs, gps) in bar:
        imgs = imgs.to(device) # Known images
        gps = torch.tensor(gps).to(device) # known GPS coordinates
        gps = gps.unsqueeze(1).T  # Ensure gps is of shape (batch_size, 2)
        optimizer.zero_grad()

        # GPS encoder to get longtitude lattitude
        gps_encoded = constant_model(gps)
        
        # GPS encoded to image
        gps_decoded = train_model(gps_encoded) #TODO: Size of gps_encoded should match the input size of VaeDecoder

        # Compute the loss
        criterion = F.binary_cross_entropy
        def resize_image_tensor(img_tensor, new_size):
            # img_tensor: [3, m, m]
            # Add batch dimension → [1, 3, m, m]
            img_tensor = img_tensor.unsqueeze(0)
            
            # Resize
            resized = F.interpolate(img_tensor, size=(new_size, new_size), mode='bilinear', align_corners=False)
            
            # Remove batch dimension → [3, n, n]
            return resized
        imgs = resize_image_tensor(imgs, gps_decoded.shape[2])
        imgs = F.sigmoid(imgs)  # Ensure images are in the range [0, 1] if using BCE loss
        # print("gps_decoded:", gps_decoded.shape, gps_decoded.min().item(), gps_decoded.max().item())
        # print("imgs:", imgs.shape, imgs.min().item(), imgs.max().item())
        loss = criterion(gps_decoded, imgs) #TODO: Size

        # Backpropagate
        loss.backward()
        optimizer.step()

        bar.set_description("Epoch {} loss: {:.5f}".format(epoch, loss.item()))

    if scheduler is not None:
        scheduler.step()
