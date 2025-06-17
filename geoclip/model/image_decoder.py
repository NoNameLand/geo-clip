import torch
import torch.nn as nn
import torch.nn.functional as F

class VaeDecoder(nn.Module):
    def __init__(self, latent_dim=512, initial_channels=128, initial_spatial=4, img_channels=3):
        super().__init__()
        self.initial_channels = initial_channels
        self.initial_spatial = initial_spatial

        # MLP head: maps latent -> intermediate representation
        # Here: Dense 1024->768->512, then expand to spatial
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 768),
            nn.ReLU(inplace=True),
            nn.Linear(768, initial_channels * initial_spatial * initial_spatial)
        )

        # Decoder: ConvTranspose to upsample
        self.decoder = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(initial_channels, initial_channels // 2, 4, 2, 1),
            nn.GroupNorm(8, initial_channels // 2),  # more stable than BatchNorm for VAEs
            nn.LeakyReLU(0.2, inplace=True),

            # 8x8 -> 16x16
            nn.ConvTranspose2d(initial_channels // 2, initial_channels // 4, 4, 2, 1),
            nn.GroupNorm(8, initial_channels // 4),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16 -> 32x32
            nn.ConvTranspose2d(initial_channels // 4, initial_channels // 8, 4, 2, 1),
            nn.GroupNorm(8, initial_channels // 8),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32 -> 64x64
            nn.ConvTranspose2d(initial_channels // 8, img_channels, 4, 2, 1),
            nn.Sigmoid()
        )       

    def forward(self, z):
        # z: (B, 1024)
        x = self.mlp(z)  # -> (B, C0 * H0 * W0)
        B = z.size(0)
        x = x.view(B, self.initial_channels, self.initial_spatial, self.initial_spatial)
        img = self.decoder(x)
        return img