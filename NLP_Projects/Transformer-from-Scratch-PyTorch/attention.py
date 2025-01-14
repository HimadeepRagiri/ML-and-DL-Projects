import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =========================================
# Scaled Dot-Product Attention
# =========================================
def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Compute the scaled dot-product attention.
    Args:
        query: Tensor of shape (batch_size, num_heads, seq_len, d_k)
        key: Tensor of shape (batch_size, num_heads, seq_len, d_k)
        value: Tensor of shape (batch_size, num_heads, seq_len, d_v)
        mask: Optional tensor to mask certain positions (batch_size, 1, seq_len, seq_len)
    Returns:
        attention_output: Tensor of shape (batch_size, num_heads, seq_len, d_v)
        attention_weights: Tensor of shape (batch_size, num_heads, seq_len, seq_len)
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attention_weights = F.softmax(scores, dim=-1)
    attention_output = torch.matmul(attention_weights, value)
    return attention_output, attention_weights

# =========================================
# Multi-Head Attention
# =========================================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projection and split into heads
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled Dot-Product Attention
        attention_output, attention_weights = scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads and apply output linear layer
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        output = self.W_o(attention_output)
        return output

