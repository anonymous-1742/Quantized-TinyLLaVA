import numpy as np
import torch.nn as nn
import torch
# ==========================================================
# 基础量化器：Two-stage & Mean-value
# ==========================================================

def uniform_quantizer(x,qmin,qmax,Q):
    x=np.clip(x,qmin,qmax)
    indices=np.around((x-qmin)/(qmax-qmin)*(Q-1))
    return(indices,qmin,qmax)

def two_stage_quantizer(a,Q_ep=16,Q_entry=64):
    a_min= np.min(a,axis=0)
    a_max= np.max(a,axis=0)
    min_indices,min_low,min_up=uniform_quantizer(a_min,np.min(a_min),np.max(a_min),Q_ep)
    max_indices,max_low,max_up=uniform_quantizer(a_max,np.min(a_max),np.max(a_max),Q_ep)
    min_q=min_indices/(Q_ep-1)*(min_up-min_low)+min_low
    max_q=max_indices/(Q_ep-1)*(max_up-max_low)+max_low
    entry_indices,entry_low,entry_up=uniform_quantizer(a,min_q,max_q,Q_entry)
    Q=entry_indices/Q_entry*(entry_up-entry_low)+entry_low
    return(Q)


def mean_value_quantizer(a, Q0=8):
    """均值量化器 (Mean-value quantizer)"""
    a_mean = np.mean(a,axis=0)
    mean_indices,mean_low,mean_up=uniform_quantizer(a_mean,np.min(a_mean),np.max(a_mean),Q0)
    mean_q=mean_indices/Q0*(mean_up-mean_low)+mean_low
    a_q = np.repeat(mean_q.reshape(1, -1),a.shape[0], axis=0)
    return a_q,mean_q


# ==========================================================
# (P) 问题求解：Water-filling 分配 Q_i
# ==========================================================
def solve_quantization_levels(a_tilde,a_tilde_0, B, Cava, tol=1e-4, max_iter=100):
    """
    求解 (P): 最优 Q_i 分配（基于 KKT 条件的 water-filling）
    a_tilde: 向量 range 数组
    B: batch size
    Cava: 可用通信预算 (bit)
    """
    M = len(a_tilde)
    nu_low, nu_high = 1e-12, 1e6

    def compute_Q(nu):
        u = (a_tilde**2 * np.log(2)) / (2 * nu)
        u0=(a_tilde_0**2*B* np.log(2)) / (nu)
        u=np.insert(u,0,u0)
        v = (u * np.sqrt(81 - 12*u) + 9*u) ** (1/3)
        Q = ((2/3)**(1/3)) * (u / v) + v / (2**(1/3) * 3**(2/3)) + 1
        Q = np.clip(Q, 2, 2**32)

        return Q
    for _ in range(max_iter):
        nu_mid = (nu_low + nu_high) / 2
        Q_mid = compute_Q(nu_mid)
        bit_sum = np.sum(np.log2(Q_mid))  # 总通信开销近似
        if bit_sum > Cava:  # 超出预算 -> 增大 ν
            nu_low = nu_mid
        else:
            nu_high = nu_mid
        if abs(bit_sum - Cava) < tol:
            break

    return compute_Q(nu_mid), nu_mid


# ==========================================================
# 自动确定 M* ：搜索最优 M 来最小化量化误差
# ==========================================================
def auto_determine_M_and_Q(a_ranges,a_tilde_0, B, D_hat, Cava):
    """
    自动搜索最优 M* 并为其求解 (P)
    """
    candidates = np.unique(np.linspace(1, D_hat // 2, num=8, dtype=int))
    best_M, best_Q, best_err = 0, None, np.inf
    for M in candidates:
        a_tilde = a_ranges[:M]
        a_bar=a_ranges[M:]
        Q_all, _ = solve_quantization_levels(a_tilde,a_tilde_0, B, Cava)
        Q0=Q_all[0]
        Q_entry=Q_all[1:]
        # 计算误差上界 (式19)
        err_two = np.sum((a_tilde**2 * B) / (4 * (Q_entry - 1)**2))
        err_mean_1 = np.sum((a_bar**2 * B)/2) # mean-quantizer误差近似
        err_mean_2=a_tilde_0**2*B*(D_hat-M)/(2*(Q0 - 1)**2)
        total_err = err_two + err_mean_1+err_mean_2
        if total_err < best_err:
            best_err = total_err
            best_M = M
            best_Q = Q_all

    return best_M, best_Q


# ==========================================================
# 主算法：Adaptive Feature-Wise Quantization
# ==========================================================
def adaptive_featurewise_quantization(A, Cava, Q_ep=200):
    """
    自适应特征量化算法（完整版本）
    输入：
        A: (B×D̂) 中间特征矩阵
        Cava: 总通信预算（bit）
    输出：
        Q: 量化后矩阵
        mu: 均值量化向量
        M*: 使用 two-stage quantizer 的列数
        Q_entry_list: 对应的量化级数组
    """
    B, D_hat = A.shape
    A=A.detach().to('cpu').numpy()
    ranges = np.max(A, axis=0) - np.min(A, axis=0)
    idx_sorted = np.argsort(-ranges)  # 按range从大到小排序
    A_sorted = A[:, idx_sorted]
    ranges_sorted = ranges[idx_sorted]
    a_tilde_0=np.max(np.mean(A,axis=0))-np.min(np.mean(A,axis=0))
    # ① 自动确定 M* 和每列最优 Q_i (P问题)
    M_star, Q_entry_list = auto_determine_M_and_Q(ranges_sorted,a_tilde_0, B, D_hat, Cava)
    # ② 执行量化
    Q = np.zeros_like(A_sorted)
    mu = np.zeros(D_hat)
    Q[:,:M_star]=two_stage_quantizer(A_sorted[:, 0:M_star], Q_ep=Q_ep,Q_entry=np.around(Q_entry_list[1:]))
    Q[:,M_star:],mu= mean_value_quantizer(A_sorted[:,M_star:], np.around(Q_entry_list[0]))

    inv_idx = np.argsort(idx_sorted)
    Q = Q[:, inv_idx]
    return Q, M_star, Q_entry_list

class FWQ(nn.Module):
    def __init__(self,token_dim,code_dim,discrete_size):
        super(FWQ,self).__init__()
        self.discrete_size=discrete_size
    def forward(self,x,return_indice=False):
        x_shape=x.shape
        flattened_x=x.view(-1,x.shape[2]) #[B*S,H]
        flattened_x_quantized,M_star, Q_entry_list=adaptive_featurewise_quantization(flattened_x,int(np.log2(self.discrete_size)))       
        output=(torch.tensor(flattened_x_quantized).to(x.device)-flattened_x).detach()+flattened_x
        L_comm=0
        L_code=0
        return output.reshape(x_shape),L_code,L_comm