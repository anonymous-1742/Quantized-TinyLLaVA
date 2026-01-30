import torch
import torch.nn.functional as F
import torch.nn as nn

class MultiheadAttention(nn.Module):
    def __init__(self,embed_dim,n_heads):
        super(MultiheadAttention,self).__init__()
        self.Q=nn.Linear(embed_dim,embed_dim*n_heads)
        self.K=nn.Linear(embed_dim,embed_dim*n_heads)
        self.V=nn.Linear(embed_dim,embed_dim*n_heads)
        self.dropout1=nn.Dropout(0.1)
        self.n_heads=n_heads
        self.embed_dim=embed_dim
        self.out_proj=nn.Linear(embed_dim*n_heads,embed_dim,bias=False)
    def forward(self,x,mask,is_causal=False):
        n_heads=self.n_heads
        B=x.shape[0]
        S=x.shape[1]
        D=x.shape[2]
        q=self.Q(x)
        q=q.reshape(B,S,n_heads,D)
        q=q.permute(0,2,1,3)
        k=self.K(x)
        k=k.reshape(B,S,n_heads,D)
        k=k.permute(0,2,1,3)
        v=self.V(x)
        v=v.reshape(B,S,n_heads,D)
        v=v.permute(0,2,1,3)
        self_attn=F.scaled_dot_product_attention(q,k,v)
        self_attn=self_attn.transpose(1,2)
        self_attn=self_attn.reshape(B,S,D*n_heads)
        attn_output=self.out_proj(self_attn)
        return(attn_output)
    
class FFN(nn.Module):
    def __init__(self,prob_size):
        super(FFN,self).__init__()
        self.Linear1=nn.Linear(prob_size,prob_size*4)
        self.Linear2=nn.Linear(4*prob_size,prob_size)
        self.drop=nn.Dropout(0.1)
        self.act=nn.ReLU()
    def forward(self,x):
        x=self.Linear1(x)
        x=self.act(x)
        x=self.drop(x)
        x=self.Linear2(x)
        return(x)
