import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_igpt import TransformerBlock


class TextGPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_layers: int, context_length: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, context_length, d_model))

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads) for _ in range(num_layers)
        ])

        # Output head
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)
        
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

        # Apply transformer blocks with causal masking
        for block in self.transformer_blocks:
            x = block(x, mask)

        # Apply final layer norm and output projection
        x = self.norm(x)
        logits = self.out(x)

        return logits

