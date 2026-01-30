import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .res import ResidualStack
from .attn import FFN,MultiheadAttention

class VQ_Decoder_img(nn.Module):
    def __init__(self, h_dim,out_dim, n_res_layers, res_h_dim):
        super(VQ_Decoder_img, self).__init__()
        kernel = 4
        stride = 2

        self.inverse_conv_stack = nn.Sequential(
            nn.ConvTranspose1d(
                h_dim, h_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            nn.ConvTranspose1d(h_dim, h_dim // 2,
                               kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(inplace=False),
            nn.ConvTranspose1d(h_dim//2, out_dim, kernel_size=kernel,
                               stride=stride, padding=1)
        )

    def forward(self, x):
        x=x.permute(0,2,1)
        x=self.inverse_conv_stack(x)
        x=x.permute(0,2,1)
        return x
    
class Decoder_block(nn.Module):
    def __init__(self,model_dim,feedforward_dim,n_heads,dropout=0.1):
        super(Decoder_block,self).__init__()
        self.linear1=nn.Sequential(nn.Linear(model_dim,feedforward_dim),nn.ReLU())
        self.linear2=nn.Sequential(nn.Linear(feedforward_dim,model_dim),nn.ReLU())
        self.self_attn=MultiheadAttention(model_dim,n_heads)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.n_heads=n_heads
    def forward(self,x,mask,is_causal=True):
        attn=self.self_attn(x,mask,is_causal)
        x=x+self.dropout1(attn)
        x=self.norm1(x)
        ff_output=self.linear1(x)
        ff_output=self.linear2(self.dropout2(ff_output))
        x=x+self.dropout3(ff_output)
        x=self.norm2(x)
        return(x)
        
        
class VQ_Decoder_text(nn.Module):
    def __init__(self,h_dim,out_dim,num_layers,n_heads):
        super(VQ_Decoder_text, self).__init__()
        self.decode=nn.ModuleList([Decoder_block(out_dim,4*out_dim,n_heads)for _ in range(num_layers)])
        #self.proj=nn.Sequential(nn.Linear(h_dim,h_dim//2),
        #                       nn.ReLU(),
        #                       nn.Linear(h_dim//2,out_dim),
        #                       nn.ReLU())
    def forward(self,x,mask=None): 
        for layer in self.decode:
            x=layer(x,mask)
        #x=self.proj(x)
        return x
    
class FSQ_Decoder(nn.Module):
    def __init__(self,h_dim,num_layers,n_heads,discrete_size):
        super(FSQ_Decoder, self).__init__()
        self.decode=nn.ModuleList([Decoder_block(h_dim,4*h_dim,n_heads)for _ in range(num_layers)])
        #self.proj=nn.Sequential(nn.Linear(h_dim,h_dim//2),
        #                       nn.ReLU(),
        #                       nn.Linear(h_dim//2,out_dim),
        #                       nn.ReLU())
        self.discrete_size=discrete_size
    def forward(self,x,mask=None): 
        x=x/self.discrete_size
        #for layer in self.decode:
        #    x=layer(x,mask)
        #x=self.proj(x)
        return x