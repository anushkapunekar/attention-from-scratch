import torch
import torch.nn as nn
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

        # Final output proj
        self.W_o = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        x: (batch_size, seq_len, embed_dim)
        """

        batch_size, seq_len, _ = x.shape

        #  Create Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        #  Split into heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)

        #  Move heads before sequence
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        #  Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)

        weights = torch.softmax(scores, dim=-1)

        # Apply attention to values
        out = torch.matmul(weights, V)

        # Concatenate heads
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, self.embed_dim)

        # Final linear projection
        output = self.W_o(out)

        return output, weights
 


# class SelfAttention(nn.Module):
#     def __init__(self, embed_dim):
#         super().__init__()
#         self.embed_dim = embed_dim

#         # These will create Q, K, V
#         self.W_q = nn.Linear(embed_dim, embed_dim)
#         self.W_k = nn.Linear(embed_dim, embed_dim)
#         self.W_v = nn.Linear(embed_dim, embed_dim)

#     def forward(self, x):
       
#         Q = self.W_q(x)    #take input words vectors and convert them into queries
#         K = self.W_k(x)    #into keys
#         V = self.W_v(x)    #into values
    
#     #raw attention
#         scores = torch.matmul(Q,K.transpose(-2,-1))

#         #scaling the scores
#         scale =math.sqrt(self.embed_dim)
#         scores=scores/scale  #reduces magnitude, stabilizes learning as told in paper

#         #softmax to get attention weights
#         attention_weights = torch.softmax(scores, dim=-1) # applies softmax row-wise, 
        
#         #apply attention to values
#         output = torch.matmul(attention_weights, V)


#         #output=> new word representations ,,, and attention_weights=> for inspection/visualisation    
#         return output, attention_weights  
    
