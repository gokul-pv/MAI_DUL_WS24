import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
        
        min_encoding_indices = min_encoding_indices.view(z_e.shape[0], z_e.shape[2], z_e.shape[3])

        # Compute loss terms
        commitment_loss = F.mse_loss(z_e, z_q.detach())
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
        indices = indices.contiguous().view(-1)
        z_q = self.vq.embedding(indices)
        z_q = z_q.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return self.decoder(z_q)

# class TransformerPrior(nn.Module):
#     def __init__(self, image_size=(8,8), vocab_size=128, d_model=256, num_heads=4, num_layers=2):
#         super().__init__()
#         self.seq_len = image_size[0] * image_size[1] + 1  # +1 for <bos> token
#         self.d_model = d_model
#         self.vocab_size = vocab_size

#         # Token and position embeddings
#         self.token_embedding = nn.Embedding(vocab_size + 1, d_model)  # +1 for <bos> token
#         self.position_embedding = nn.Embedding(self.seq_len, d_model)

#         # Transformer decoder
#         self.transformer_decoder = nn.TransformerDecoder(
#             nn.TransformerDecoderLayer(d_model, num_heads, dim_feedforward=4 * d_model, dropout=0.1, activation='gelu', batch_first=True),
#             num_layers,
#             nn.LayerNorm(d_model)
#         )

#         # Output head
#         self.norm = nn.LayerNorm(d_model)  # Final layer norm
#         self.out = nn.Linear(d_model, vocab_size)  # Doesn't predict <bos> token

#     def forward(self, x, mask=None):
#         batch_size, seq_len = x.shape

#         # Add embeddings
#         token_emb = self.token_embedding(x) * math.sqrt(self.d_model)

#         pos_emb = self.position_embedding(torch.arange(seq_len, device=x.device))
#         pos_emb = pos_emb.unsqueeze(0).expand(batch_size, -1, -1)

#         x = token_emb + pos_emb

#         # Apply transformer decoder
#         if mask is not None:
#             output = self.transformer_decoder(tgt=x, memory=x, tgt_mask=mask, memory_mask=mask)
#         else:
#             output = self.transformer_decoder(tgt=x, memory=x)

#         # Apply final layer norm and output projection
#         output = self.norm(output)

#         logits = self.out(output)

#         return logits



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # Separate Q, K, V projections and reshape for multi-head attention
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores with scaled dot product
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        # Apply mask if provided (converting mask to proper shape)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Apply softmax and compute weighted sum
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)

        # Reshape and project output
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.d_model)
        output = self.out_proj(context)

        return output


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # FFN with GELU activation
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x, mask=None):
        # Pre-norm architecture
        attended = x + self.attention(self.norm1(x), mask)
        output = attended + self.ffn(self.norm2(attended))
        return output


class TransformerPrior(nn.Module):
    def __init__(self, image_size=(8,8), vocab_size=128, d_model=256, num_heads=4, num_layers=2):
        super().__init__()
        self.seq_len = image_size[0] * image_size[1] + 1  # +1 for <bos> token
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size + 1, d_model)  # +1 for <bos> token
        self.position_embedding = nn.Parameter(torch.zeros(1, self.seq_len, d_model))

        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads) for _ in range(num_layers)
        ])

        # Output head
        self.norm = nn.LayerNorm(d_model)  # Final layer norm
        self.out = nn.Linear(d_model, vocab_size)  # Doesn't predict <bos> token

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize embeddings and position encodings
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)
        
        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(self, x, mask=None):
        batch_size, seq_len = x.shape

        # Add embeddings
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding[:, :seq_len, :]
        x = token_emb + pos_emb

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)

        # Apply final layer norm and output projection
        x = self.norm(x)
        logits = self.out(x)

        return logits