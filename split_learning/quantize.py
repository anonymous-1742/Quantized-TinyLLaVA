import numpy as np
import torch

def random_fp32_embedding_torch(num_tokens, dim, device="cpu", seed=None):
    """
    随机生成 FP32 embedding（PyTorch）

    Args:
        num_tokens (int): token 数量
        dim (int): embedding 维度
        device (str): "cpu" or "cuda"
        seed (int, optional): 随机种子

    Returns:
        torch.Tensor: shape = (num_tokens, dim), dtype = torch.float32
    """
    if seed is not None:
        torch.manual_seed(seed)

    embedding = torch.randn(num_tokens, dim, dtype=torch.float32, device=device)
    return embedding

def quantize_fp32_to_2bit_torch(x):
    max_val = x.abs().max() + 1e-8
    scale = max_val / 1.5
    q = torch.round(x / scale + 1.5).clamp(0, 3).to(torch.uint8)
    return q, scale


def pack_2bit_tensor(q: torch.Tensor):
    """
    q: torch.Tensor, dtype=torch.uint8, values in {0,1,2,3}
    returns:
        packed: torch.Tensor, dtype=torch.uint8
        pad: int
    """
    assert q.dtype == torch.uint8

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

    return packed, pad

def unpack_2bit_tensor(packed: torch.Tensor, pad=None):
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
    return q.float()
