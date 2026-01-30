import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math


class LearnableBinarization(Function):
    @staticmethod
    def forward(ctx, x, alpha, training=True):
        noise = torch.rand_like(x)
        ctx.save_for_backward(x, alpha,noise)
        ctx.training = training
        p = (alpha + x) / (2 * alpha + 1e-8)
        binary = torch.where(noise < p, alpha, -alpha)
        binary = torch.clamp(binary, min=-alpha, max=alpha)
        return binary

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha,noise = ctx.saved_tensors
        training = ctx.training
        grad_x = torch.zeros_like(x)
        grad_alpha = torch.zeros_like(x)
        
        mask_mid = (x.abs() <= alpha)  # 中间区域
        mask_pos = (x > alpha)         # 大于α
        mask_neg = (x < -alpha)        # 小于-α
        if ctx.needs_input_grad[0]:
            grad_x[mask_mid] = grad_output[mask_mid]  # 中间区域梯度为1
            # 其他区域梯度为0（已初始化为0）
        
        # 对 α 的梯度：简化版本（可改进为论文精确公式）
        if ctx.needs_input_grad[1]:
            # 公式(9)的近似
            grad_alpha[mask_pos] = grad_output[mask_pos]  # +1
            grad_alpha[mask_neg] = -grad_output[mask_neg] # -1
            if training:
                # 中间区域：近似梯度
                grad_alpha[mask_mid] = grad_output[mask_mid] * (2 * torch.floor((x[mask_mid]-alpha)/(2*alpha)+noise[mask_mid]) - x[mask_mid]/alpha -1)
            else:
                grad_alpha[mask_mid] = 0
            grad_alpha=torch.sum(grad_alpha)
            grad_alpha=(torch.ones(1, 1, 1).to(x.device)) * grad_alpha
        return grad_x, grad_alpha, None

def learnable_binarize(x, alpha, training=True):
    """包装函数，便于调用"""
    return LearnableBinarization.apply(x, alpha, training)    
    
class LearnableBinarizationLayer(nn.Module):
    def __init__(self, alpha_init=1.0, rho=6.0, use_exp_param=True):
        super().__init__()
        self.use_exp_param = use_exp_param
        self.rho = rho
        
        if use_exp_param:
            self.alpha_prime = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
            self.alpha_e = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        else:
            self.raw_alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
    
    def forward(self, x, training=True):
        if self.use_exp_param:
            alpha = self.alpha_prime * torch.exp(self.rho * self.alpha_e)
        else:
            alpha = F.softplus(self.raw_alpha)      
        if alpha.dim() < x.dim():
            alpha = alpha.view(*[1]*x.dim())
        
        return learnable_binarize(x, alpha, training)
    
