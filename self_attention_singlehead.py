import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleHeadAttention(nn.Module):
    
    def __init__(self, d_model, dropout=0.1):

        super().__init__()
        ##这一行代码在 Python 继承 (inheritance) 体系中很重要，作用是调用 nn.Module 的 __init__ 方法，确保 SingleHeadAttention 继承 nn.Module 的功能。
        self.d_model=d_model
        self.dropout=nn.Dropout(dropout)

        self.W_q=nn.Linear(d_model, d_model)
        self.W_k=nn.Linear(d_model, d_model)
        self.W_v=nn.Linear(d_model, d_model)

        self.output_proj=nn.Linear(d_model, d_model)

    
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        dot_prod=torch.matmul(q, k.transpose(-1, -2))
        scale=self.d_model**0.5
        scores=dot_prod/scale

        if mask is not None:
            scores=scores.mask_fill(mask==0, float('-inf'))
        
        attn_weights=F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)  # dropout
        output=torch.matmul(attn_weights,v)
        return output
    def forward(self, x, mask=None):
        q=self.W_q(x)
        k=self.W_q(x)
        v=self.W_v(x)

        output=self.scaled_dot_product_attention(self, q,k,v,mask)
        output=self.output_proj(output)
        return output