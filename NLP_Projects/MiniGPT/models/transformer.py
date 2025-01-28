import torch.nn as nn
from .attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    """
    Complete transformer block including multi-head attention,
    feed-forward network, and layer normalization.
    """

    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.GELU(),  # Using GELU activation as in GPT
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        # Multi-head attention
        attention = self.attention(value, key, query, mask)

        # First residual connection and layer norm
        x = self.dropout(self.norm1(attention + query))

        # Feed-forward network
        forward = self.feed_forward(x)

        # Second residual connection and layer norm
        out = self.dropout(self.norm2(forward + x))
        return out