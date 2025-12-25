import torch
from attention import MultiHeadSelfAttention

# fake sentence: 1 sentence, 3 words, embedding size 4
x = torch.randn(1, 3, 4)

attention = MultiHeadSelfAttention(embed_dim=4, num_heads=2)

output, weights = attention(x)

print("Output shape:", output.shape)
print("Attention weights shape:", weights.shape)




#single-headed self attention
# attention = SelfAttention(embed_dim=4)

# # Q, K, V = attention(x)

# # print("Q shape:", Q.shape) 
# # print("K shape:", K.shape) 
# # print("V shape:", V.shape) 
# # 
# output,weights = attention(x)


# print("Output shape:", output.shape)
# print("Attention weights shape:",weights.shape)

# print("\nAttention weights: ")
# print(weights)

# #check sanity : rows should sum to 1 (softmax)
# # print("Row sums:", weights.sum(dim=-1))