import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class ImageGPT(nn.Module):
    def __init__(self, image_size, vocab_size=2, d_model=128, num_heads=4, num_layers=2):
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