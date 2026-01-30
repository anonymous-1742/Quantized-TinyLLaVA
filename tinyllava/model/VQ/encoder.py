import torch
import torch.nn.functional as F
import torch.nn as nn
from .res import ResidualStack
import numpy as np
from .attn import FFN,MultiheadAttention

class VQ_encoder_layer(nn.Module):
    def __init__(self,dim,n_heads):
        super(VQ_encoder_layer,self).__init__()
        self.attn=MultiheadAttention(dim,n_heads)
        self.feedforward=FFN(dim)
        self.norm1=nn.LayerNorm(dim)
        self.norm2=nn.LayerNorm(dim)
        self.dropout1=nn.Dropout(0.1)
        self.dropout2=nn.Dropout(0.1)
    def forward(self,x,mask):
        self_attn=self.attn(x,mask)
        x=x+self.dropout1(self_attn)
        x=self.norm1(x)
        f_out=self.feedforward(x)
        x=x+self.dropout2(f_out)
        return(self.norm2(x))

class VQ_Encoder_img(nn.Module):

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(VQ_Encoder_img, self).__init__()
        kernel = 4
        stride = 2
        self.conv_stack = nn.Sequential(
            nn.Conv1d(in_dim, h_dim // 2, kernel_size=kernel,
                      stride=stride, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv1d(h_dim // 2, h_dim, kernel_size=kernel,
                      stride=stride, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv1d(h_dim, h_dim, kernel_size=kernel,
                      stride=1, padding=1),
            ResidualStack(
                h_dim, h_dim, res_h_dim, n_res_layers)

        )

    def forward(self, x):
        x=x.permute(0,2,1)
        x=self.conv_stack(x)
        x=x.permute(0,2,1)
        return x

class VQ_Encoder_text(nn.Module):
    def __init__(self,in_dim,num_layers,n_heads):
        super(VQ_Encoder_text, self).__init__()
        self.encode=nn.ModuleList([VQ_encoder_layer(in_dim,n_heads) for _ in range(num_layers)])
    def forward(self, x,mask=None):
        for layer in self.encode:
            x=layer(x,mask)
        return x

