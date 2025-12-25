import torch
from attention import SelfAttention, MultiHeadSelfAttention

# Example input: 1 sentence, 3 words, embedding size 4
x = torch.randn(1, 3, 4)

print("=== Single-Head Self-Attention ===")
single_attention = SelfAttention(embed_dim=4)
single_output, single_weights = single_attention(x)

print("Output shape:", single_output.shape)
print("Attention weights shape:", single_weights.shape)

print("\n=== Multi-Head Self-Attention ===")
multi_attention = MultiHeadSelfAttention(embed_dim=4, num_heads=2)
multi_output, multi_weights = multi_attention(x)

print("Output shape:", multi_output.shape)
print("Attention weights shape:", multi_weights.shape)
