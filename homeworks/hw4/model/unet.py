
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings"""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half)
    args = timesteps[:, None].float() * freqs[None].to(timesteps.device)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.temb_proj = nn.Linear(temb_channels, out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, temb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        
        temb = self.temb_proj(temb)[:, :, None, None]
        h = h + temb
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        
        return self.shortcut(x) + h


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, hidden_dims=[64, 128, 256, 512], blocks_per_dim=2):
        super().__init__()
        self.hidden_dims = hidden_dims
        
        # Time embedding
        temb_channels = hidden_dims[0] * 4
        self.temb_net = nn.Sequential(
            nn.Linear(hidden_dims[0], temb_channels),
            nn.SiLU(),
            nn.Linear(temb_channels, temb_channels)
        )
        
        self.conv_in = nn.Conv2d(in_channels, hidden_dims[0], 3, padding=1)
        
        # Down blocks
        self.down_blocks = nn.ModuleList()
        prev_ch = hidden_dims[0]
        self.down_block_chans = [prev_ch]
        
        for i, hidden_dim in enumerate(hidden_dims):
            for _ in range(blocks_per_dim):
                self.down_blocks.append(ResidualBlock(prev_ch, hidden_dim, temb_channels))
                prev_ch = hidden_dim
                self.down_block_chans.append(prev_ch)
            if i != len(hidden_dims) - 1:
                self.down_blocks.append(Downsample(prev_ch))
                self.down_block_chans.append(prev_ch)
        
        # Middle blocks
        self.middle_blocks = nn.ModuleList([
            ResidualBlock(prev_ch, prev_ch, temb_channels),
            ResidualBlock(prev_ch, prev_ch, temb_channels)
        ])
        
        # Up blocks
        self.up_blocks = nn.ModuleList()
        for i, hidden_dim in list(enumerate(hidden_dims))[::-1]:
            for j in range(blocks_per_dim + 1):
                self.up_blocks.append(ResidualBlock(prev_ch + self.down_block_chans.pop(), hidden_dim, temb_channels))
                prev_ch = hidden_dim
                if i and j == blocks_per_dim:
                    self.up_blocks.append(Upsample(prev_ch))
        
        # Output layers
        self.norm_out = nn.GroupNorm(8, prev_ch)
        self.conv_out = nn.Conv2d(prev_ch, in_channels, 3, padding=1)
    
    def forward(self, x, t):
        # Time embedding
        temb = timestep_embedding(t, self.hidden_dims[0])
        temb = self.temb_net(temb)
        
        # Initial convolution
        h = self.conv_in(x)
        
        # Down path
        hs = [h]
        for module in self.down_blocks:
            if isinstance(module, ResidualBlock):
                h = module(h, temb)
            else:
                h = module(h)
            hs.append(h)
        
        # Middle
        for module in self.middle_blocks:
            h = module(h, temb)
        
        # Up path
        for module in self.up_blocks:
            if isinstance(module, ResidualBlock):
                h = torch.cat([h, hs.pop()], dim=1)
                h = module(h, temb)
            else:
                h = module(h)
        
        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        return h
