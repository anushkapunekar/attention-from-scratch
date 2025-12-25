import torch
import torch.nn as nn
import math


class SelfAttention(nn.Module):
    """
    Single-Head Self-Attention module inspired by
    'Attention Is All You Need' (Vaswani et al., 2017).

    Demonstrates the core idea of scaled dot-product
    self-attention using Query, Key, Value.

    Input:
        x: Tensor of shape (batch_size, seq_len, embed_dim)

    Output:
        output: Tensor of shape (batch_size, seq_len, embed_dim)
        attention_weights: Tensor of shape (batch_size, seq_len, seq_len)
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        x: (batch_size, seq_len, embed_dim)
        """

        # 1. Project input embeddings to Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 2. Compute raw attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1))

        # 3. Scale and normalize scores
        scores = scores / math.sqrt(self.embed_dim)
        attention_weights = torch.softmax(scores, dim=-1)

        # 4. Apply attention weights to values
        output = torch.matmul(attention_weights, V)

        return output, attention_weights


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention module inspired by
    'Attention Is All You Need' (Vaswani et al., 2017).

    This module:
    - projects input embeddings into Query, Key, Value
    - applies scaled dot-product attention independently per head
    - concatenates head outputs and projects back to embedding space

    Input:
        x: Tensor of shape (batch_size, seq_len, embed_dim)

    Output:
        output: Tensor of shape (batch_size, seq_len, embed_dim)
        attention_weights: Tensor of shape (batch_size, num_heads, seq_len, seq_len)
    """

    def __init__(self, embed_dim, num_heads):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

        # Final output projection
        self.W_o = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        x: (batch_size, seq_len, embed_dim)
        """

        batch_size, seq_len, _ = x.shape

        # 1. Linear projections
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 2. Split embedding dimension into heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # 3. Transpose to (batch, heads, seq_len, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # 4. Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)
        attention_weights = torch.softmax(scores, dim=-1)

        # 5. Apply attention to values
        out = torch.matmul(attention_weights, V)

        # 6. Concatenate heads
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, self.embed_dim)

        # 7. Final linear projection
        output = self.W_o(out)

        return output, attention_weights
