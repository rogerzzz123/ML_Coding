import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.d_model=d_model
        self.h=heads
        self.d_k=self.d_model//heads

        self.q_linear=nn.Linear(d_model, d_model)
        self.k_linear=nn.Linear(d_model, d_model)
        self.v_linear=nn.Linear(d_model, d_model)

        self.dropout=nn.Dropout(dropout)
        self.out=nn.Linear(d_model, d_model)

    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        # attn_score=torch.matmul(q, k.transpose(-2, -1))/math.sqrt(d_k)
        attn_score=torch.matmul(q, k.transpose(-2, -1))/math.sqrt(d_k)
        if mask is not None:
            mask=mask.unsqueeze(1) # add dimension at specific index
            attn_score=attn_score.masked_fill(mask==0, -1e9)
        attn_score=F.softmax(attn_score, dim=-1)
        if dropout:
            attn_score=dropout(attn_score)
        output=torch.matmul(attn_score, v)
        return output
    
    def forward(self, x ,mask=None):
        batch_size=x.size(0)
        q=self.q_linear(x)
        k=self.k_linear(x)
        v=self.v_linear(x)
        
        q=q.view(batch_size, -1, self.h, self.d_k).transpose(1,2)
        k=k.view(batch_size, -1, self.h, self.d_k).transpose(1,2)
        v=v.view(batch_size, -1, self.h, self.d_k).transpose(1,2)
        scores=self.attention(q, k, v, self.d_k,mask, self.dropout)
        concat_score=scores.transpose(1,2).contiguous().view(batch_size, -1, self.d_model)
        output=self.out(concat_score)
        return output

import torch

# Define input dimensions
batch_size = 2
seq_len = 5  # Number of tokens in a sequence
d_model = 8  # Model dimension (must be divisible by `heads`)
num_heads = 2  # Number of attention heads
dropout_rate = 0.1

# Create a random input tensor (batch_size, seq_len, d_model)
x = torch.randn(batch_size, seq_len, d_model)

# Initialize the Multi-Head Attention model
mha = MultiHeadAttention(heads=num_heads, d_model=d_model, dropout=dropout_rate)

# Forward pass
output = mha(x)

# Print results
print("Input Shape:", x.shape)  # Expected: (batch_size, seq_len, d_model)
print("Output Shape:", output.shape)  # Expected: (batch_size, seq_len, d_model)
print("\nAttention Output:")
print(output)