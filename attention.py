import torch
import torch.nn as nn
import math


class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        # These will create Q, K, V
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
       
        Q = self.W_q(x)    #take input words vectors and convert them into queries
        K = self.W_k(x)    #into keys
        V = self.W_v(x)    #into values
    
    #raw attention
        scores = torch.matmul(Q,K.transpose(-2,-1))

        #scaling the scores
        scale =math.sqrt(self.embed_dim)
        scores=scores/scale   #reduces magnitude, stabilizes learning as told in paper

        #softmax to get attention weights
        attention_weights = torch.softmax(scores, dim=-1) # applies softmax row-wise, 
        return attention_weights
