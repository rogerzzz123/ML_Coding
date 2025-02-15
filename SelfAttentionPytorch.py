from math import sqrt
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v):
        super().__init__()
        self.dim_q=dim_q
        self.dim_k=dim_k
        self.dim_v=dim_v
        self.linear_q=nn.Linear(dim_q, dim_k, bias=False)
        self.linear_k=nn.Linear(dim_q, dim_k, bias=False)
        self.linear_v=nn.Linear(dim_q, dim_v, bias=False)
        self.norm_fact=1/sqrt(self.dim_k)

    
    def forward(self, x):
        batch, n, dim_q=x.shape
        assert dim_q==self.dim_q
        q=self.linear_q(x)
        k=self.linear_k(x)
        v=self.linear_v(x)
        dist=torch.bmm(q, k.transpose(1,2))* self.norm_fact #bmm() is optimized for 3D tensors (batch processing).
        dist=torch.softmax(dist, dim=-1)
        att=torch.bmm(dist, v)
        return dist, att
    
batch_size = 2
seq_len = 4
embedding_dim = 8  # dim_q = embedding_dim
dim_k = 6
dim_v = 6

# Create a random input tensor (batch_size, seq_len, embedding_dim)
x = torch.randn(batch_size, seq_len, embedding_dim)
self_attn = SelfAttention(dim_q=embedding_dim, dim_k=dim_k, dim_v=dim_v)

# Forward pass
attn_output, attn_weights = self_attn(x)

# Print results
print("Attention Output Shape:", attn_output.shape)  # Expected: (batch_size, seq_len, dim_v)
print("Attention Weights Shape:", attn_weights.shape)  # Expected: (batch_size, seq_len, seq_len)

print("\nAttention Output:")
print(attn_output)

print("\nAttention Weights:")
print(attn_weights)
