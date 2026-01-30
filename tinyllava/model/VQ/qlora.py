import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from scipy.stats import norm

def generate_nf_table(bits: int, device="cpu", dtype=torch.float16):
    n_levels = 2 ** bits
    p = (np.arange(n_levels) + 0.5) / n_levels
    q = norm.ppf(p)

    q = q / np.max(np.abs(q))
    q=torch.tensor(q).to(device=device, dtype=dtype)
    return q

'''
def generate_nf_table(bits: int, device="cpu", dtype=torch.float16):
    n_levels = 2 ** bits
    # 在 (-1, 1) 的概率区间上均匀采样
    probs = torch.linspace(0, 1, n_levels + 1)[1:-1]
    # 根据标准正态分布的分位数函数生成
    print(probs)
    values = torch.erfinv(2 * probs - 1)
    values = values / (values.abs().max()+1e-8)  # 归一化到 [-1, 1]
    values = torch.cat([torch.tensor([-1.0]), values, torch.tensor([1.0])])
    return values.to(device=device, dtype=dtype)
    '''

class NFNDoubleQuantizer_split(nn.Module):
    def __init__(self, bits=4, block_size=64, use_double_quant=True):
        super().__init__()
        self.bits = bits
        self.block_size = block_size
        self.use_double_quant = use_double_quant
        self.table=generate_nf_table(bits)

    def compress(self, x):
        flattened_x=x.view(-1,x.shape[2])
        orig_shape = x.shape
        x = x.view(orig_shape[0], orig_shape[1] // self.block_size, self.block_size)

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
    
        payload = q_idx
        aux = {
            "scales_q": scales_q,
            "s_min": s_min,
            "s_max": s_max,
            "mins": x_min,
            "orig_shape": orig_shape,
            "payload_shape":payload.shape
        }

        return payload, aux, 0


    def decompress(self, payload, aux):
        q_idx = payload
        scales_q = aux["scales_q"]
        s_min = aux["s_min"]
        s_max = aux["s_max"]
        mins = aux["mins"]
        orig_shape = aux["orig_shape"]

        if scales_q is not None:
            scales = s_min + (scales_q.float() / 255) * (s_max - s_min)
        else:
            scales = s_min

        scales = scales.unsqueeze(-1)

        w_block = self.table[q_idx.long()].to(dtype=torch.float32, device=scales.device)
        w_block = (w_block + 1) / 2 * scales + mins

        return w_block.view(orig_shape)
    
class NFNDoubleQuantizer(nn.Module):
    def __init__(self, bits=4, block_size=64, use_double_quant=True):
        super().__init__()
        self.bits = bits
        self.block_size = block_size
        self.use_double_quant = use_double_quant
        self.table=generate_nf_table(bits)

    def quantize(self, x):
        orig_shape = x.shape
        x = x.view(orig_shape[0],orig_shape[1]//self.block_size, self.block_size)

        q_idx_list, scale_list, min_list = [], [], []
        x_min, x_max = x.min(dim=2).values.unsqueeze(-1), x.max(dim=2).values.unsqueeze(-1)
        scales = (x_max - x_min).squeeze(-1)
        x_norm = 2 * (x - x_min) / (x_max - x_min + 1e-8) - 1
        dist = torch.abs(x_norm.unsqueeze(-1) - self.table.to(x.device))
        q_idx = torch.argmin(dist, dim=-1).to(torch.uint8)
        mins = x_min
        # Double Quantization: 再量化scale (8bit)
        if self.use_double_quant:
            s_min, s_max = scales.min(dim=-1).values.unsqueeze(-1), scales.max(dim=-1).values.unsqueeze(-1)
            scales_q = ((scales - s_min) / (s_max - s_min + 1e-8) * 255).round().to(torch.uint8)
        else:
            scales_q, s_min, s_max = None, None, None

        return q_idx, scales_q, s_min, s_max, mins

    def dequantize(self, q_idx, scales_q, s_min, s_max, mins):
        if scales_q is not None:
            scales = s_min + (scales_q.float() / 255) * (s_max - s_min)
        else:
            scales = s_min
        scales=scales.unsqueeze(-1)
        w_block = self.table[q_idx.to(device=self.table.device,dtype=torch.long)].to(dtype=torch.float32,device=scales.device)
        w_block = (w_block + 1) / 2 * scales +mins
 
        return w_block.reshape(-1,q_idx.shape[1]*self.block_size)
    
class Qlora_quantize(nn.Module):
    def __init__(self, bits=4, block_size=64, use_double_quant=True):
        super().__init__()
        self.quantizer = NFNDoubleQuantizer(bits, block_size, use_double_quant)

    def forward(self, x):
        flattened_x=x.view(-1,x.shape[2])
        q_idx, s_q, s_min, s_max, mins = self.quantizer.quantize(flattened_x)
        flattened_x_q = self.quantizer.dequantize(q_idx, s_q, s_min, s_max, mins).to(dtype=x.dtype)
        output=flattened_x + (flattened_x_q - flattened_x).detach()
        return output.reshape(x.shape)