import torch
from attention import SelfAttention

# fake sentence: 1 sentence, 3 words, embedding size 4
x = torch.randn(1, 3, 4)

attention = SelfAttention(embed_dim=4)

# Q, K, V = attention(x)

# print("Q shape:", Q.shape) 
# print("K shape:", K.shape) 
# print("V shape:", V.shape) 
scores = attention(x)

print("Attention scores shape:" , scores.shape)
print(scores)