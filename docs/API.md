# API Documentation

## GeoCLIP

The main GeoCLIP model for image geo-localization.

### Constructor

```python
GeoCLIP(from_pretrained=True, queue_size=4096)
```

**Parameters:**
- `from_pretrained` (bool): Whether to load pre-trained weights. Default: True
- `queue_size` (int): Size of the queue for contrastive learning. Default: 4096

### Methods

#### predict(image_path, top_k=5)

Predicts GPS coordinates for a given image.

**Parameters:**
- `image_path` (str): Path to the image file
- `top_k` (int): Number of top predictions to return. Default: 5

**Returns:**
- `top_pred_gps` (list): List of GPS coordinate tuples (latitude, longitude)
- `top_pred_prob` (list): List of corresponding probability scores

**Example:**
```python
from geoclip_og import GeoCLIP

model = GeoCLIP()
top_pred_gps, top_pred_prob = model.predict("image.jpg", top_k=5)

for i, (coords, prob) in enumerate(zip(top_pred_gps, top_pred_prob)):
    lat, lon = coords
    print(f"Prediction {i+1}: ({lat:.6f}, {lon:.6f}) - Probability: {prob:.6f}")
```

## LocationEncoder

Pre-trained GPS location encoder that can be used as a component in other geo-aware models.

### Constructor

```python
LocationEncoder()
```

### Methods

#### forward(gps_coords)

Encodes GPS coordinates into dense vector representations.

**Parameters:**
- `gps_coords` (torch.Tensor): Tensor of GPS coordinates with shape (batch_size, 2) where each coordinate is [latitude, longitude]

**Returns:**
- `embeddings` (torch.Tensor): Dense vector representations with shape (batch_size, 512)

**Example:**
```python
import torch
from geoclip_og import LocationEncoder

encoder = LocationEncoder()
gps_data = torch.Tensor([[40.7128, -74.0060], [34.0522, -118.2437]])  # NYC and LA
embeddings = encoder(gps_data)
print(embeddings.shape)  # torch.Size([2, 512])
```

## ImageEncoder

Image encoder component of GeoCLIP.

### Constructor

```python
ImageEncoder()
```

### Methods

#### forward(images)

Encodes images into dense vector representations.

**Parameters:**
- `images` (torch.Tensor): Batch of images with shape (batch_size, 3, H, W)

**Returns:**
- `embeddings` (torch.Tensor): Image embeddings with shape (batch_size, embedding_dim)

## Training

### train(train_dataloader, model, optimizer, epoch, batch_size, device, scheduler=None, criterion=nn.CrossEntropyLoss())

Main training function for GeoCLIP.

**Parameters:**
- `train_dataloader` (DataLoader): Training data loader
- `model` (GeoCLIP): GeoCLIP model to train
- `optimizer` (torch.optim.Optimizer): Optimizer for training
- `epoch` (int): Current epoch number
- `batch_size` (int): Batch size
- `device` (torch.device): Device to run training on
- `scheduler` (optional): Learning rate scheduler
- `criterion` (optional): Loss function. Default: CrossEntropyLoss

**Example:**
```python
from geoclip_og import GeoCLIP
from geoclip_og.train import train
import torch

model = GeoCLIP()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assuming you have a train_dataloader
train(train_dataloader, model, optimizer, epoch=0, batch_size=64, device=device)
```
