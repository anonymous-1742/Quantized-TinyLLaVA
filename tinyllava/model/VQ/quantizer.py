import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .attn import FFN

def cosine_similarity(x, y, dim=-1, eps=1e-4):
    dot_product = (x * y).sum(dim=dim)
    x_norm = x.norm(p=2, dim=dim)
    y_norm = y.norm(p=2, dim=dim)
    return dot_product / (x_norm * y_norm + eps)
class CosineLoss(nn.Module):
    def __init__(self):
        super(CosineLoss, self).__init__()
    def forward(self,x1,x2,target,eps=1e-3):
        sim=cosine_similarity(x1,x2,dim=-1,eps=eps)
        loss=1-sim
        return(loss.sum()/loss.shape[0])

def robust_minmax(X,width=3):
    mean = X.mean()
    std = X.std()
    q_min = mean - width * std
    q_max = mean + width * std
    X_clipped = torch.clamp(X, q_min, q_max)
    X_norm = 2 * (X_clipped - q_min+1e-4) / (q_max - q_min+1e-4) - 1
    #X_norm = 2 * (X_clipped - X_clipped.min()+1e-4) / (X_clipped.max() - X_clipped.min()+1e-4) - 1
    return X_norm,q_min,q_max

class FFN_Proj(nn.Module):
    def __init__(self,in_dim,hidden_dim,out_dim,discrete_size):
        super(FFN_Proj,self).__init__()
        self.linear1=nn.Linear(in_dim,hidden_dim)
        self.linear2=nn.Linear(hidden_dim,out_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu
        self.norm = nn.LayerNorm(out_dim)
    def forward(self, x):
        x_ffn = self.activation(self.fc1(x))
        x_ffn = self.dropout(x_ffn)
        x_ffn = self.fc2(x_ffn)
        return x_ffn
        

def quantize(X, size):
    device=X.device
    half_width=(size-1)/2
    offset=((size-1)%2)/2
    X=X*half_width-offset
    X_round=torch.round(X)
    X_ste=(X_round-X).detach()+X
    X_ste=(X_ste+offset)/half_width
    #X_scaled=X*half_width-offset #modified on 11.3
    #X_round=torch.round(X_scaled) #modified on 11.3
    #X_round=(X_round+offset)/half_width #modified on 11.3
    #X_ste=(X_round-X).detach()+X #modified on 11.3
    indices=torch.round(X_ste*half_width+half_width)
    return X_ste, indices.detach().int()
    
'''
def quantize(X, levels):
    device = X.device
    dtype=X.dtype
    levels = levels.to(device,dtype)

    # [N, D], N = B*S*H, D=1
    X_flat = X.view(-1, 1)  
    levels = levels.view(-1, 1)  # [K, 1]

    # pairwise distance: [N, K]
    dist = torch.cdist(X_flat, levels)  
    indices = dist.argmin(dim=-1)

    X_quant = levels[indices]
    X_ste = (X_quant - X_flat).detach() + X_flat
    return X_ste.view_as(X), indices.view_as(X)
'''
class FSQ(nn.Module):
    def __init__(self,token_dim,code_dim,discrete_size):
        super(FSQ,self).__init__()
        self.token_dim=token_dim
        self.code_dim=code_dim
        self.discrete_size=discrete_size
        self.loss=CosineLoss()
        self.in_proj=nn.Sequential(nn.Linear(token_dim,code_dim))
        #self.in_proj=nn.Sigmoid()
        self.out_proj=nn.Linear(code_dim,token_dim)
        #self.in_proj=nn.Sequential(nn.Linear(token_dim,code_dim),nn.LayerNorm(code_dim,elementwise_affine=False))
        #self.out_proj=nn.Sequential(nn.Linear(code_dim,token_dim),nn.LayerNorm(token_dim,elementwise_affine=False))
        self.levels=torch.linspace(-1,1,steps=discrete_size)
        
    def compress(self, x):
        x_shape = x.shape
        x = self.in_proj(x)
        x, q_min, q_max = robust_minmax(x)

        flattened_x = x.view(-1, x.shape[2])  # [B*S, H]
        flattened_x_quantized, indices = quantize(flattened_x, self.discrete_size)

        target = torch.ones(flattened_x_quantized.shape[0], device=x.device)
        L_comm = self.loss(flattened_x, flattened_x_quantized.detach(), target)

        payload = indices                      # 需要传输
        aux = {
            "embedding_shape": x_shape,
        }

        return payload, aux, L_comm
    
    def decompress(self, payload, aux):
        indices = payload
        x_shape = aux["embedding_shape"]
        half_len = (self.discrete_size - 1) / 2
        flattened_x_quantized = (indices.float() - half_len) / half_len
        output = self.out_proj(flattened_x_quantized)
        return output.view(x_shape)
        
    def forward(self,x,return_indice=False):
        x_shape=x.shape
        x=self.in_proj(x)
        x,q_min,q_max=robust_minmax(x)
        flattened_x=x.view(-1,x.shape[2]) #[B*S,H]
        #flattened_x_quantized,indices=quantize(flattened_x,self.discrete_size)
        flattened_x_quantized,indices=quantize(flattened_x,self.discrete_size)
        target=torch.ones(flattened_x_quantized.shape[0]).to(x.device)
        L_comm=self.loss(flattened_x,flattened_x_quantized.detach(),target)
        if return_indice:
            output=indices.float()
        else:
            output=self.out_proj(flattened_x_quantized)
        L_code=0
        #output=output*(q_max-q_min)+q_min
        return output.reshape(x_shape),L_code,L_comm

class TopKSparse(nn.Module):
    def __init__(self, token_dim, code_dim, discrete_size,randomized=True, random_p=0.1):

        super(TopKSparse, self).__init__()
        self.token_dim = token_dim
        self.code_dim = code_dim
        self.sparsity_ratio = np.log2(discrete_size)/16
        self.randomized = randomized
        self.random_p = random_p
        

    def forward(self, x, return_mask=False):
        x_shape = x.shape  # [B, S, H]
        flattened_x = x.view(-1, x.shape[-1])  # [B*S, H]

        k = max(1, int(self.sparsity_ratio * flattened_x.shape[1]))
        x_abs = flattened_x.abs()
        topk_val, topk_idx = torch.topk(x_abs, k, dim=1)
        mask = torch.zeros_like(flattened_x, dtype=torch.bool)
        mask.scatter_(1, topk_idx, True)
        if self.randomized and self.random_p > 0:
            rand_mask = (torch.rand_like(flattened_x) < self.random_p)
            mask = mask | rand_mask  # union of deterministic and random parts
        x_sparse = flattened_x * mask.half()
        selected_count = mask.sum(dim=1, keepdim=True).clamp(min=1)
        scale = flattened_x.shape[1] / selected_count
        x_sparse = x_sparse * scale
        L_comm = 0
        if return_mask:
            output = mask.half()
        else:
            output = x_sparse

        L_code = 0
        return output.view(x_shape), L_code, L_comm    


    
class FSQ_old(nn.Module):
    def __init__(self,token_dim,code_dim,discrete_size):
        super(FSQ_old,self).__init__()
        self.token_dim=token_dim
        self.code_dim=code_dim
        self.discrete_size=discrete_size
        self.loss=CosineLoss()
        self.in_proj=nn.Sequential(nn.Linear(token_dim,code_dim),nn.Tanh())
        #self.in_proj=nn.Sigmoid()
        self.out_proj=nn.Linear(code_dim,token_dim)
    def forward(self,x):
        x_shape=x.shape
        x=self.in_proj(x)
        flattened_x=x.view(-1,x.shape[2]) #[B*S,H]
        flattened_x_quantized,indices=quantize(flattened_x,self.discrete_size)
        output=self.out_proj(flattened_x_quantized)
        target=torch.ones(flattened_x_quantized.shape[0]).to(x.device)
        L_comm=0
        L_code=0
        return output.reshape(x_shape),L_code,L_comm       
    
    
class ADSQ(nn.Module):
    def __init__(self,token_dim,code_dim,discrete_size):
        super(ADSQ,self).__init__()
        self.token_dim=token_dim
        self.code_dim=code_dim
        self.discrete_size=discrete_size
        self.loss=CosineLoss()
        self.in_proj=nn.Sequential(nn.Linear(token_dim,code_dim))
        #self.in_proj=nn.Sigmoid()
        self.out_proj=nn.Sequential(nn.Linear(code_dim,token_dim))
        self.width=nn.Parameter(torch.tensor([3.0]))
    def forward(self,x,return_indice=False):
        x_shape=x.shape
        x=self.in_proj(x)
        x=robust_minmax(x,self.width)
        flattened_x=x.view(-1,x.shape[2]) #[B*S,H]
        flattened_x_quantized,indices=quantize(flattened_x,self.discrete_size)
        target=torch.ones(flattened_x_quantized.shape[0]).to(x.device)
        L_comm=self.loss(flattened_x,flattened_x_quantized.detach(),target)
        #output=self.out_proj(flattened_x_quantized)
        if return_indice:
            output=indices.float()
        else:
            output=self.out_proj(flattened_x_quantized)
        L_code=0
        return output.reshape(x_shape),L_code,L_comm
    
'''
class ADSQ(nn.Module):
    def __init__(self,token_dim,code_dim,discrete_size,sub_codebook_size):
        super(ADSQ,self).__init__()
        self.token_dim=token_dim
        self.code_dim=code_dim
        self.discrete_size=discrete_size
        self.sigmoid=nn.Sigmoid()
        codebook=torch.linspace(-1,1,steps=discrete_size).repeat(code_dim,1).T.reshape(discrete_size,sub_codebook_size,code_dim//sub_codebook_size)
        self.codebook=nn.Parameter(codebook.permute(2,0,1).contiguous())#[n_split,K,sub_codebook_size]
        self.sub_codebook_size=sub_codebook_size
        self.loss=CosineLoss()
        self.in_proj=nn.Sequential(nn.Linear(token_dim,code_dim))
        self.out_proj=nn.Linear(code_dim,token_dim)
    def forward(self,x):
        x_shape=x.shape
        x=self.in_proj(x)
        x=robust_minmax(x)
        flattened_x=x.view(-1,x.shape[2]) #[B*S,H]
        split_x=flattened_x.reshape(flattened_x.shape[0],-1,self.sub_codebook_size)#[B*S,n_split,sub_codebook_size]
        original_input=flattened_x.detach()
        diff = torch.norm(split_x.unsqueeze(2) - self.codebook.unsqueeze(0),dim=-1) #[B*S,n_split,1,sub_codebook_size],[1,n_split,K,sub_codebook_size]->[B*S,n_split,K]
        indices = torch.argmin(diff, dim=-1)#[B*S,n_split]
        #print(indices.shape)
        col_idx = torch.arange(split_x.shape[1], device=x.device)
        split_x= self.codebook[col_idx,indices]
        #print(split_x.shape)
        flattened_x_quantized=split_x.reshape(flattened_x.shape)
        output=(flattened_x_quantized-flattened_x).detach()+flattened_x
        output=self.out_proj(output)
        target=torch.ones(flattened_x_quantized.shape[0]).to(x.device)
        L_code=self.loss(flattened_x.detach(),flattened_x_quantized,target)
        L_comm=self.loss(flattened_x,flattened_x_quantized.detach(),target)
        return output.reshape(x_shape),L_code,L_comm
 '''

'''
class ADSQ(nn.Module):
    def __init__(self,token_dim,code_dim,discrete_size,sub_codebook_size):
        super(ADSQ,self).__init__()
        self.token_dim=token_dim
        self.code_dim=code_dim
        self.discrete_size=discrete_size
        self.sigmoid=nn.Sigmoid()
        levels=torch.linspace(-1,1,steps=discrete_size).repeat(code_dim,1).T.reshape(discrete_size,code_dim)
        self.levels=nn.Parameter(levels.permute(1,0).contiguous())
        self.loss=CosineLoss()
        self.in_proj=nn.Sequential(nn.Linear(token_dim,code_dim))
        self.out_proj=nn.Linear(code_dim,token_dim)
    def forward(self,x):
        x_shape=x.shape
        x=self.in_proj(x)
        x=robust_minmax(x)
        flattened_x=x.view(-1,x.shape[2]) #[B*S,H]
        diff = torch.norm(flattened_x.unsqueeze(1) - self.levels.unsqueeze(0),dim=-1) #[B*S,H,1],[1,H,K]->[B*S,H,K]
        indices = torch.argmin(diff, dim=-1)#[B*S,H]
        #print(indices.shape)
        col_idx = torch.arange(flattened_x.shape[1], device=x.device)
        flattened_x_quantized= self.levels[col_idx,indices]
        #print(split_x.shape)
        output=(flattened_x_quantized-flattened_x).detach()+flattened_x
        output=self.out_proj(output)
        target=torch.ones(flattened_x_quantized.shape[0]).to(x.device)
        L_code=self.loss(flattened_x.detach(),flattened_x_quantized,target)
        L_comm=self.loss(flattened_x,flattened_x_quantized.detach(),target)
        return output.reshape(x_shape),L_code,L_comm
 '''