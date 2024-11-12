import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 1, 1, 0)
        )
        
    def forward(self, x):
        return x + self.layers(x)

class Encoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 4, 2, 1),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim)
        )
        
    def forward(self, x):
        return self.layers(x)

class Decoder(nn.Module):
    def __init__(self, out_channels=3, hidden_dim=256):
        super().__init__()
        self.layers = nn.Sequential(
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, out_channels, 4, 2, 1)
        )
        
    def forward(self, x):
        return self.layers(x)

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=128, embedding_dim=256):
        super().__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        
        # Initialize codebook uniformly in [-1/K, 1/K]
        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1/self.K, 1/self.K)
        
    def forward(self, z_e):
        # z_e: (B, D, H, W)
        # Reshape z_e to (B*H*W, D)
        z_e_flat = z_e.permute(0, 2, 3, 1).contiguous()
        z_e_flat = z_e_flat.view(-1, self.D)
        
        # Calculate distances to all codebook vectors
        d = torch.sum(z_e_flat ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - \
            2 * torch.matmul(z_e_flat, self.embedding.weight.t())
        
        # Find nearest codebook vector
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q_flat = self.embedding(min_encoding_indices)
        
        # Reshape back to match input shape
        z_q = z_q_flat.view(z_e.shape[0], z_e.shape[2], z_e.shape[3], self.D)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        
        # Compute loss terms
        commitment_loss = F.mse_loss(z_q.detach(), z_e)
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        
        # Straight-through estimator
        z_q = z_e + (z_q - z_e).detach()
        
        return z_q, commitment_loss, codebook_loss, min_encoding_indices

class VQVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.vq = VectorQuantizer()
        self.decoder = Decoder()
        
    def forward(self, x):
        z_e = self.encoder(x)
        z_q, commitment_loss, codebook_loss, indices = self.vq(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, commitment_loss, codebook_loss, indices
    
    def encode(self, x):
        z_e = self.encoder(x)
        _, _, _, indices = self.vq(z_e)
        return indices
    
    def decode(self, indices):
        B, H, W = indices.shape
        z_q = self.vq.embedding(indices).permute(0, 3, 1, 2)
        return self.decoder(z_q)

class TransformerPrior(nn.Module):
    def __init__(self, vocab_size=128, d_model=256, nhead=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, d_model)  # +1 for start token
        self.pos_embedding = nn.Parameter(torch.randn(1, 65, d_model))  # 8x8 + 1 start token
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=1024)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.output = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        # Add start token
        B = x.shape[0]
        start_token = torch.full((B, 1), 128, device=x.device)  # vocab_size as start token
        x = torch.cat([start_token, x.view(B, -1)], dim=1)
        
        # Embed tokens and add positional encoding
        x = self.embedding(x)
        x = x + self.pos_embedding
        
        # Run through transformer
        x = self.transformer(x)
        
        # Get logits
        logits = self.output(x)
        
        return logits