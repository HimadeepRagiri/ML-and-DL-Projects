import torch
import torch.nn as nn
from .transformer import TransformerBlock


class GPT(nn.Module):
    """
    Complete GPT model implementation with transformer blocks,
    positional embeddings, and token embeddings.
    """

    def __init__(self, vocab_size, embed_size, num_layers, heads,
                 forward_expansion, dropout, max_length):
        super(GPT, self).__init__()

        # Token and position embeddings
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pos_embed = nn.Embedding(max_length, embed_size)

        # Transformer layers
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, heads, dropout, forward_expansion)
             for _ in range(num_layers)]
        )

        self.norm = nn.LayerNorm(embed_size)
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T = x.shape

        # Get positions for each token
        positions = torch.arange(0, T).expand(B, T).to(x.device)

        # Combine token and positional embeddings
        x = self.dropout(self.embed(x) + self.pos_embed(positions))

        # Pass through transformer blocks
        for layer in self.layers:
            x = layer(x, x, x, mask)

        x = self.norm(x)
        return self.fc_out(x)