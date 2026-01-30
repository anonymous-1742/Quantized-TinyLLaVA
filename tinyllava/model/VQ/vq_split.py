import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import VQ_Encoder_img,VQ_Encoder_text
from .decoder import VQ_Decoder_img,VQ_Decoder_text,FSQ_Decoder
import numpy as np
from scipy.stats import norm


import os

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


def quantize(X, size):
    device=X.device
    half_width=(size-1)/2
    offset=((size-1)%2)/2
    X=X*half_width-offset
    X_round=torch.round(X)
    X=(X_round+offset)/half_width
    indices=torch.round(X*half_width+half_width)
    return X, indices.detach().int()
    

def pack_2bit_tensor(q: torch.Tensor):
    """
    q: torch.Tensor, dtype=torch.uint8, values in {0,1,2,3}
    returns:
        packed: torch.Tensor, dtype=torch.uint8
        pad: int
    """
    assert q.dtype == torch.uint8
    q_shape=q.shape
    q = q.reshape(-1)

    pad = (-q.numel()) % 4
    if pad:
        q = torch.cat([
            q,
            torch.zeros(pad, dtype=q.dtype, device=q.device)
        ])
    else:
        pad=0

    q = q.view(-1, 4)
    packed = (
        (q[:, 0] << 0) |
        (q[:, 1] << 2) |
        (q[:, 2] << 4) |
        (q[:, 3] << 6)
    ).to(torch.uint8)

    return packed,q_shape, pad

def unpack_2bit_tensor(packed: torch.Tensor,q_shape,pad=0):
    """
    packed: torch.uint8 tensor
    pad: int
    """
    assert packed.dtype == torch.uint8

    q = torch.empty(
        (packed.numel(), 4),
        dtype=torch.uint8,
        device=packed.device
    )

    q[:, 0] = (packed >> 0) & 0b11
    q[:, 1] = (packed >> 2) & 0b11
    q[:, 2] = (packed >> 4) & 0b11
    q[:, 3] = (packed >> 6) & 0b11

    q = q.view(-1)
    if pad:
        q = q[:-pad]
    q=q.reshape(q_shape)
    return q.float()

def pack_4bit_tensor(q: torch.Tensor):
    """
    q: torch.Tensor, dtype=torch.uint8, values in {0..15}
    returns:
        packed: torch.Tensor, dtype=torch.uint8
        q_shape: original shape
        pad: int
    """
    assert q.dtype == torch.uint8
    q_shape = q.shape
    q = q.reshape(-1)

    # 每个 uint8 存 2 个 4-bit
    pad = (-q.numel()) % 2
    if pad:
        q = torch.cat([
            q,
            torch.zeros(pad, dtype=q.dtype, device=q.device)
        ])
    else:
        pad = 0

    q = q.view(-1, 2)
    packed = (
        (q[:, 0] & 0x0F) |
        ((q[:, 1] & 0x0F) << 4)
    ).to(torch.uint8)

    return packed, q_shape, pad

def unpack_4bit_tensor(packed: torch.Tensor, q_shape, pad=0):
    """
    packed: torch.uint8 tensor
    q_shape: original shape
    pad: int
    """
    assert packed.dtype == torch.uint8

    q = torch.empty(
        (packed.numel(), 2),
        dtype=torch.uint8,
        device=packed.device
    )

    q[:, 0] = packed & 0x0F
    q[:, 1] = (packed >> 4) & 0x0F

    q = q.view(-1)
    if pad:
        q = q[:-pad]

    q = q.reshape(q_shape)
    return q.float()

def pack_3bit_tensor(q: torch.Tensor):
    """
    q: torch.Tensor, dtype=torch.uint8, values in {0..7}
    returns:
        packed: torch.Tensor, dtype=torch.uint8
        q_shape: original shape
        pad: int
    """
    assert q.dtype == torch.uint8
    q_shape = q.shape
    q = q.reshape(-1)

    # 每 8 个 3-bit -> 3 个 uint8
    pad = (-q.numel()) % 8
    if pad:
        q = torch.cat([
            q,
            torch.zeros(pad, dtype=q.dtype, device=q.device)
        ])
    else:
        pad = 0

    q = q.view(-1, 8)

    packed = torch.empty(
        (q.size(0), 3),
        dtype=torch.uint8,
        device=q.device
    )

    packed[:, 0] = (
        (q[:, 0] & 0x07) |
        ((q[:, 1] & 0x07) << 3) |
        ((q[:, 2] & 0x03) << 6)
    )

    packed[:, 1] = (
        ((q[:, 2] >> 2) & 0x01) |
        ((q[:, 3] & 0x07) << 1) |
        ((q[:, 4] & 0x07) << 4) |
        ((q[:, 5] & 0x01) << 7)
    )

    packed[:, 2] = (
        ((q[:, 5] >> 1) & 0x03) |
        ((q[:, 6] & 0x07) << 2) |
        ((q[:, 7] & 0x07) << 5)
    )

    return packed.view(-1), q_shape, pad

def unpack_3bit_tensor(packed: torch.Tensor, q_shape, pad=0):
    """
    packed: torch.uint8 tensor
    q_shape: original shape
    pad: int
    """
    assert packed.dtype == torch.uint8

    packed = packed.view(-1, 3)

    q = torch.empty(
        (packed.size(0), 8),
        dtype=torch.uint8,
        device=packed.device
    )

    q[:, 0] = packed[:, 0] & 0x07
    q[:, 1] = (packed[:, 0] >> 3) & 0x07
    q[:, 2] = ((packed[:, 0] >> 6) & 0x03) | ((packed[:, 1] & 0x01) << 2)
    q[:, 3] = (packed[:, 1] >> 1) & 0x07
    q[:, 4] = (packed[:, 1] >> 4) & 0x07
    q[:, 5] = ((packed[:, 1] >> 7) & 0x01) | ((packed[:, 2] & 0x03) << 1)
    q[:, 6] = (packed[:, 2] >> 2) & 0x07
    q[:, 7] = (packed[:, 2] >> 5) & 0x07

    q = q.view(-1)
    if pad:
        q = q[:-pad]

    return q.reshape(q_shape).float()



class VQ_config:
    def __init__(self,token_dim,discrete_size,embed_dim=2048,code_dim=64,L_comm_cost=0.25,perp_cost=0.1,L_code_cost=0.25,sub_codebook_size=1):
        self.sub_codebook_size=sub_codebook_size
        self.token_dim=token_dim
        self.embed_dim=embed_dim
        self.code_dim=code_dim
        self.discrete_size=discrete_size
        self.comm_cost=L_comm_cost
        self.code_cost=L_code_cost
        self.perp_cost=perp_cost

def pad_sequence_to_length(x: torch.Tensor, target_len: int, pad_value: float = 0.0):
    B, L, H = x.shape
    if L >= target_len:
        return x[:, :target_len, :]
    pad_len = target_len - L
    return F.pad(x, pad=(0, 0, 0, pad_len), value=pad_value)

class empty_VQ(nn.Module):
    def __init__(self):
        super(empty_VQ,self).__init__()
    def forward(self,inputs_embeds,mask=None,return_indice=False):
        return(inputs_embeds,0)

    
class FSQ_block_split(nn.Module):
    def __init__(self,config):
        super(FSQ_block_split,self).__init__()
        self.comm_cost=config.comm_cost
        self.quantizer=FSQ_split(config.token_dim,config.code_dim,config.discrete_size)
    def forward(self,inputs_embeds,return_indice=False):
        output_embeds,L_code,L_comm=self.quantizer(inputs_embeds,return_indice)
        vq_loss=self.comm_cost*L_comm
        return(output_embeds,vq_loss)
    
class Qlora_block_split(nn.Module):
    def __init__(self,config):
        super(Qlora_block_split,self).__init__()
        bits=int(np.log2(config.discrete_size))
        self.quantizer=NFNDoubleQuantizer_split(bits=bits)
        self.comm_cost=config.comm_cost
    def forward(self,inputs_embeds,return_indice=False):
        output_embeds=self.quantizer(inputs_embeds)
        vq_loss=0
        return(output_embeds,vq_loss)

class TopKSparse_block_split(nn.Module):
    def __init__(self,config):
        super(TopKSparse_block_split,self).__init__()
        self.comm_cost=config.comm_cost
        self.quantizer=TopKSparse_split(config.token_dim,config.code_dim,config.discrete_size)
    def forward(self,inputs_embeds,return_mask=False):
        output_embeds,L_code,L_comm=self.quantizer(inputs_embeds)
        vq_loss=self.comm_cost*L_comm
        return(output_embeds.to(dtype=inputs_embeds.dtype),vq_loss)  
    
class FSQ_old_block_split(nn.Module):
    def __init__(self,config):
        super(FSQ_old_block_split,self).__init__()
        self.comm_cost=config.comm_cost
        self.quantizer=FSQ_old_split(config.token_dim,config.code_dim,config.discrete_size)
    def forward(self,inputs_embeds,return_indice=False):
        output_embeds,L_code,L_comm=self.quantizer(inputs_embeds)
        vq_loss=self.comm_cost*L_comm
        return(output_embeds,vq_loss) 
    
class FSQ_old_split(nn.Module):
    def __init__(self,token_dim,code_dim,discrete_size):
        super(FSQ_old_split,self).__init__()
        self.token_dim=token_dim
        self.code_dim=code_dim
        self.discrete_size=discrete_size
        self.loss=CosineLoss()
        self.in_proj=nn.Sequential(nn.Linear(token_dim,code_dim),nn.Tanh())
        #self.in_proj=nn.Sigmoid()
        self.out_proj=nn.Linear(code_dim,token_dim)
    def encode(self,x):
        x=self.in_proj(x)
        return(x)
    
    def compress(self, x):
        """
        x: [B, S, token_dim]
        """
        x_shape = x.shape

        flattened_x = x.view(-1, x.shape[-1])   # [B*S, H]
        flattened_x_q, indices = quantize(
            flattened_x, self.discrete_size
        )                                       # indices ∈ {0,1,2,3}
        target = torch.ones(flattened_x_q.shape[0], device=x.device)
        indices_uint8 = indices.to(torch.uint8)
        
        if(self.discrete_size==4):
            packed,q_shape, pad = pack_2bit_tensor(indices_uint8)
        elif(self.discrete_size==16):
            packed,q_shape, pad = pack_4bit_tensor(indices_uint8)

        payload = {
            "packed_indices": packed,   # torch.uint8
            "pad": pad,
            "q_shape":q_shape
        }

        aux = {
            "embedding_shape": x_shape
        }

        return payload, aux, 0

    def decompress(self, payload, aux):
        packed = payload["packed_indices"]
        pad = payload["pad"]
        q_shape = payload["q_shape"]
        x_shape = aux["embedding_shape"]
        if self.discrete_size==4:
            indices = unpack_2bit_tensor(packed,q_shape, pad)   # float tensor
        elif self.discrete_size==16:
            indices = unpack_4bit_tensor(packed,q_shape, pad)   # float tensor
        indices = indices.view(-1, self.code_dim)

        half_len = (self.discrete_size - 1) / 2
        flattened_x_q = (indices - half_len) / half_len
        output = flattened_x_q

        return output.view(x_shape)
    def decode(self,x):
        x=self.out_proj(x)
        return(x)  
    
    
def generate_nf_table(bits: int, device="cpu", dtype=torch.float16):
    n_levels = 2 ** bits
    p = (np.arange(n_levels) + 0.5) / n_levels
    q = norm.ppf(p)

    q = q / np.max(np.abs(q))
    q=torch.tensor(q).to(device=device, dtype=dtype)
    return q    
    
class NFNDoubleQuantizer_split(nn.Module):
    def __init__(self, bits=4, block_size=64, use_double_quant=True):
        super().__init__()
        self.bits = bits
        self.block_size = block_size
        self.use_double_quant = use_double_quant
        self.table=generate_nf_table(bits)
        self.discrete_size=2**bits
    def encode(self,x):
        return x

    def compress(self, x):
        original_shape=x.shape
        x=x.view(-1,x.shape[-1])
        x_shape = x.shape
        x = x.view(x_shape[0], x_shape[1] // self.block_size, self.block_size)
        x_min = x.min(dim=2).values.unsqueeze(-1)
        x_max = x.max(dim=2).values.unsqueeze(-1)
        scales = (x_max - x_min).squeeze(-1)

        x_norm = 2 * (x - x_min) / (x_max - x_min + 1e-8) - 1
        dist = torch.abs(x_norm.unsqueeze(-1) - self.table.to(x.device))
        q_idx = torch.argmin(dist, dim=-1).to(torch.uint8)

        if self.use_double_quant:
            s_min = scales.min(dim=-1).values.unsqueeze(-1)
            s_max = scales.max(dim=-1).values.unsqueeze(-1)
            scales_q = ((scales - s_min) / (s_max - s_min + 1e-8) * 255).round().to(torch.uint8)
        else:
            scales_q, s_min, s_max = None, scales, None
    
        if(self.discrete_size==4):
            packed,q_shape, pad = pack_2bit_tensor(q_idx)
        elif(self.discrete_size==8):
            packed,q_shape, pad = pack_3bit_tensor(q_idx)
        elif(self.discrete_size==16):
            packed,q_shape, pad = pack_4bit_tensor(q_idx)
        payload = {
            "packed_indices": packed,   # torch.uint8
            "pad": pad,
            "q_shape":q_shape
        }

        aux = {
            "scales_q": scales_q,
            "s_min": s_min,
            "s_max": s_max,
            "mins": x_min,
            "x_shape": x_shape,
            "original_shape":original_shape
        }

        return payload, aux, 0
    def decompress(self, payload, aux):
        packed = payload["packed_indices"]
        pad = payload["pad"]
        q_shape = payload["q_shape"]
        if self.discrete_size==4:
            q_idx = unpack_2bit_tensor(packed,q_shape, pad).to("cpu")
        elif self.discrete_size==8:
            q_idx = unpack_3bit_tensor(packed,q_shape, pad).to("cpu")# float tensor
        elif self.discrete_size==16:
            q_idx = unpack_4bit_tensor(packed,q_shape, pad).to("cpu")   # float tensor
        scales_q = aux["scales_q"]
        s_min = aux["s_min"]
        s_max = aux["s_max"]
        mins = aux["mins"]
        x_shape = aux["x_shape"]
        original_shape=aux["original_shape"]

        if scales_q is not None:
            scales = s_min + (scales_q.float() / 255) * (s_max - s_min)
        else:
            scales = s_min

        scales = scales.unsqueeze(-1)

        w_block = self.table[q_idx.long()].to(dtype=torch.float32, device=scales.device)
        w_block = (w_block + 1) / 2 * scales + mins
        flatten_x=w_block.view(x_shape)
        return flatten_x.view(original_shape)
    
    def decode(self,x):
        return x
    
class FSQ_split(nn.Module):
    def __init__(self, token_dim, code_dim, discrete_size=4):
        super(FSQ_split, self).__init__()

        self.token_dim = token_dim
        self.code_dim = code_dim
        self.discrete_size = discrete_size

        self.loss = CosineLoss()
        self.in_proj = nn.Linear(token_dim, code_dim)
        self.out_proj = nn.Linear(code_dim, token_dim)

        self.levels = torch.linspace(-1, 1, steps=discrete_size)

    # =========================
    # Client-side
    # =========================
    def encode(self,x):
        x = self.in_proj(x)
        x, q_min, q_max = robust_minmax(x)
        return x
    
    def compress(self, x):
        """
        x: [B, S, token_dim]
        """
        x_shape = x.shape

        flattened_x = x.view(-1, x.shape[-1])   # [B*S, H]
        flattened_x_q, indices = quantize(
            flattened_x, self.discrete_size
        )                                       # indices ∈ {0,1,2,3}
        target = torch.ones(flattened_x_q.shape[0], device=x.device)
        L_comm = self.loss(flattened_x, flattened_x_q.detach(), target)
        indices_uint8 = indices.to(torch.uint8)
        if self.discrete_size==4:
            packed,q_shape, pad = pack_2bit_tensor(indices_uint8)
        elif self.discrete_size==8:
            packed,q_shape, pad = pack_3bit_tensor(indices_uint8)
        elif self.discrete_size==16:
            packed,q_shape, pad = pack_4bit_tensor(indices_uint8)    
        payload = {
            "packed_indices": packed,   # torch.uint8
            "pad": pad,
            "q_shape":q_shape
        }

        aux = {
            "embedding_shape": x_shape
        }

        return payload, aux, L_comm

    # =========================
    # Server-side
    # =========================
    def decompress(self, payload, aux):
        packed = payload["packed_indices"]
        pad = payload["pad"]
        q_shape = payload["q_shape"]
        x_shape = aux["embedding_shape"]
        if self.discrete_size==4:
            indices = unpack_2bit_tensor(packed,q_shape, pad)   # float tensor
        elif self.discrete_size==8:
            indices = unpack_3bit_tensor(packed,q_shape, pad)
        elif self.discrete_size==16: 
            indices = unpack_4bit_tensor(packed,q_shape, pad)
        indices = indices.view(-1, self.code_dim)

        half_len = (self.discrete_size - 1) / 2
        flattened_x_q = (indices - half_len) / half_len
        output = flattened_x_q

        return output.view(x_shape)
    def decode(self,x):
        x=self.out_proj(x)
        return(x)
    
    
class empty_VQ_split(nn.Module):
    def __init__(self):
        super(empty_VQ_split,self).__init__()
        self.comm_cost=0
        self.quantizer=empty_split()
    def forward(self,inputs_embeds,return_indice=False):
        output_embeds,L_code,L_comm=self.quantizer(inputs_embeds)
        vq_loss=self.comm_cost*L_comm
        return(output_embeds,vq_loss) 
    
class empty_split(nn.Module):
    def __init__(self):
        super(empty_split, self).__init__()

    def encode(self,x):
        return x
    
    def compress(self, x):
        """
        x: [B, S, token_dim]
        """
        x=x.half()
        payload = {
            "values": x  # torch.uint8
        }

        aux = {
            "embedding_shape": x.shape
        }

        return payload, aux, 0

    # =========================
    # Server-side
    # =========================
    def decompress(self, payload, aux):
        x = payload["values"]
        x=x.float()
        return x
    def decode(self,x):
        return(x)


import torch
import torch.nn as nn
import numpy as np


import torch
import torch.nn as nn
import numpy as np

def packbits(mask: torch.Tensor) -> torch.Tensor:
    """
    mask: BoolTensor [BS, H]
    return: UInt8Tensor [BS, ceil(H/8)]
    """
    BS, H = mask.shape
    pad = (8 - H % 8) % 8
    if pad > 0:
        mask = torch.nn.functional.pad(mask, (0, pad), value=0)

    mask = mask.view(BS, -1, 8)  # [BS, ?, 8]

    weights = (2 ** torch.arange(8, device=mask.device)).view(1, 1, 8)
    packed = (mask * weights).sum(dim=-1).to(torch.uint8)
    return packed

def unpackbits(packed: torch.Tensor, H: int) -> torch.Tensor:
    """
    packed: UInt8Tensor [BS, ceil(H/8)]
    return: BoolTensor [BS, H]
    """
    BS, L = packed.shape
    bits = ((packed.unsqueeze(-1) >> torch.arange(8, device=packed.device)) & 1)
    bits = bits.view(BS, -1)
    return bits[:, :H].bool()



class TopKSparse_split(nn.Module):
    def __init__(
        self,
        token_dim,
        code_dim,
        discrete_size,
        randomized=True,
        random_p=0.1,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.code_dim = code_dim
        self.discrete_size = discrete_size

        self.sparsity_ratio = np.log2(discrete_size) / 16
        self.randomized = randomized
        self.random_p = random_p

    def encode(self, x):
        return x

    def compress(self, x):
        """
        x: [BS, H]
        """
        BS, H = x.shape
        device = x.device

        # ---------- 1. Top-K ----------
        k = max(1, int(self.sparsity_ratio * H))
        _, topk_idx = torch.topk(x.abs(), k, dim=1)  # [BS, k]

        # ---------- 2. Random augmentation ----------
        if self.randomized and self.random_p > 0:
            rand_k = max(1, int(self.random_p * H))
            rand_idx = torch.randint(0, H, (BS, rand_k), device=device)
            col_idx = torch.cat([topk_idx, rand_idx], dim=1)
        else:
            col_idx = topk_idx

        # ---------- 3. Build boolean mask ----------
        mask = torch.zeros((BS, H), device=device, dtype=torch.bool)
        row_idx = torch.arange(BS, device=device).unsqueeze(1)
        mask[row_idx, col_idx] = True

        # ---------- 4. Pack mask to 1-bit ----------
        packed_mask = packbits(mask)  # [BS, ceil(H/8)], uint8

        # ---------- 5. Gather values (detach) ----------
        # values 顺序严格与 mask 展开顺序一致
        values = x[mask].detach()  # [N]

        payload = {
            "packed_mask": packed_mask,  # uint8, true 1-bit per dim
            "values": values.half(),
        }

        aux = {
            "embedding_shape": (BS, H),
        }

        return payload, aux, 0

    @torch.no_grad()
    def decompress(self, payload, aux):
        packed_mask = payload["packed_mask"]  # [BS, ceil(H/8)]
        values = payload["values"].float()

        BS, H = aux["embedding_shape"]

        # ---------- 1. Unpack mask ----------
        mask = unpackbits(packed_mask,H)

        # ---------- 2. Reconstruct ----------
        x_flat = torch.zeros((BS, H), device=values.device, dtype=values.dtype)
        x_flat[mask] = values

        return x_flat  # [BS, H]

    def decode(self, x):
        return x
    
    
class VQ_split(nn.Module):
    def __init__(self,VQ_type,config_text,config_image=None):
        super(VQ_split,self).__init__()
        self.VQ_type=VQ_type
        
        if(self.VQ_type=="FSQ"):
            self.quantizer_single=empty_VQ()
            self.quantizer_text=FSQ_block_split(config_text)
            self.quantizer_image=FSQ_block_split(config_image)
        elif(self.VQ_type=="FSQ_old"):
            self.quantizer_single=empty_VQ()
            self.quantizer_text=FSQ_old_block_split(config_text)
            self.quantizer_image=FSQ_old_block_split(config_image)
        elif(self.VQ_type=="Qlora"):
            self.quantizer_single=empty_VQ()
            self.quantizer_text=Qlora_block_split(config_text)
            self.quantizer_image=Qlora_block_split(config_image)
        elif(self.VQ_type=="TopK"):
            self.quantizer_single=empty_VQ()
            self.quantizer_text= TopKSparse_block_split(config_text)
            self.quantizer_image=TopKSparse_block_split(config_image)
        elif(self.VQ_type=="None"):
            self.quantizer_single=empty_VQ_split()
            self.quantizer_text=empty_VQ_split()
            self.quantizer_image=empty_VQ_split()
        else:
            raise ValueError("Compressor type not supported")

    def load_model(self,**kwargs):
        pretrained_vq_path = kwargs.get('pretrained_vq_path', None)
        if pretrained_vq_path is not None:
            pretrained_vq_path = os.path.join(pretrained_vq_path, 'pytorch_model.bin')
            vq_weights = torch.load(pretrained_vq_path, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            self.quantizer_single.load_state_dict(get_w(vq_weights, 'quantizer_single'))
            self.quantizer_text.load_state_dict(get_w(vq_weights, 'quantizer_text'))
            self.quantizer_image.load_state_dict(get_w(vq_weights, 'quantizer_image'))
            print(f'Loading quantizer from {pretrained_vq_path}...')
