import torch
import torch.nn.functional as F 

class SelfAttentionNonPytorch:

    def __init__(self, dim_q, dim_k, dim_v):
        self.dim_q=dim_q
        self.dim_k=dim_k
        self.dim_v=dim_v
        self.Wq=torch.randn(dim_q, dim_k)/dim_k**0.5
        self.Wk=torch.randn(dim_q, dim_k)/dim_k**0.5
        self.Wv=torch.randn(dim_q, dim_v)/dim_v**0.5
        self.norm_fact=1/(dim_k**0.5)
    
    def forward(self, x):
        batch, n, dim_q=x.shape
        q=torch.matmul(x, self.Wq)
        k=torch.matmul(x, self.Wk)
        v=torch.matmul(x, self.Wv)
        attention_score=torch.matmul(q, k.transpose(1,2))*self.norm_fact
        attention_weights=F.softmax(attention_score, dim=-1)
        output=torch.matmul(attention_weights, v)
        return output, attention_weights

# Define input
batch_size = 2
seq_len = 4
embedding_dim = 8
dim_k = 6
dim_v = 6

# Create a random input tensor
x = torch.randn(batch_size, seq_len, embedding_dim)

# Initialize the Self-Attention model
self_attention = SelfAttentionNonPytorch(dim_q=embedding_dim, dim_k=dim_k, dim_v=dim_v)

# Forward pass
output, attn_weights = self_attention.forward(x)

# Print results
print("Attention Output Shape:", output.shape)  # Expected: (batch_size, seq_len, dim_v)
print("Attention Weights Shape:", attn_weights.shape)  # Expected: (batch_size, seq_len, seq_len)