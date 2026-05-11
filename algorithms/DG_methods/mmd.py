"""
Maximum Mean Discrepancy (MMD) implementation for domain adaptation
"""

import torch
import torch.nn as nn


def compute_mmd_loss(source_features, target_features, kernel_type='rbf', kernel_num=5, kernel_mul=1e-5):

    """
    compute the Maximum Mean Discrepancy (Maximum Mean Discrepancy, MMD)
    
    Args:
        source_features:  [batch_size, feature_dim]
        target_features: [batch_size, feature_dim]
        kernel_type: 'rbf'
        kernel_num: number of kernel functions
        kernel_mul: kernel function parameter multiplier  
        
    Returns:
        mmd_loss: MMD loss value
    """
    if not isinstance(source_features, torch.Tensor) or not isinstance(target_features, torch.Tensor):
        return torch.tensor(0.0).cuda()

    source_features = torch.nn.functional.normalize(source_features, p=2, dim=1)
    target_features = torch.nn.functional.normalize(target_features, p=2, dim=1)
    
    batch_size = int(source_features.size()[0])
    kernels = [kernel_mul * (2.0 ** i) for i in range(kernel_num)]
    
     
    XX = torch.matmul(source_features, source_features.t())
    YY = torch.matmul(target_features, target_features.t())
    XY = torch.matmul(source_features, target_features.t())
    
    X_sqnorms = torch.diag(XX)
    Y_sqnorms = torch.diag(YY)
    
    # compute RBF kernel
    mmd_loss = 0.0
    for kernel_val in kernels:
        # K(x,x)
        K_XX = torch.exp(-kernel_val * (X_sqnorms.unsqueeze(1) + X_sqnorms.unsqueeze(0) - 2 * XX))
        # K(y,y) 
        K_YY = torch.exp(-kernel_val * (Y_sqnorms.unsqueeze(1) + Y_sqnorms.unsqueeze(0) - 2 * YY))
        # K(x,y)
        K_XY = torch.exp(-kernel_val * (X_sqnorms.unsqueeze(1) + Y_sqnorms.unsqueeze(0) - 2 * XY))
        
        mmd_loss += K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
    
    return mmd_loss / kernel_num


def compute_multi_kernel_mmd(source_features, target_features, sigmas=None):
    """
    compute multi-kernel MMD, using multiple different kernel function parameters
    
    Args:
        source_features: source domain features
        target_features: target domain features
        sigmas: kernel function parameter list, if None then automatically compute
        
    Returns:
        mmd_loss: MMD loss value
    """
    if not isinstance(source_features, torch.Tensor) or not isinstance(target_features, torch.Tensor):
        return torch.tensor(0.0).cuda()
    
    if sigmas is None:
        # automatically compute suitable kernel parameters
        with torch.no_grad():
            # compute the median of the distance between features as a reference
            combined = torch.cat([source_features, target_features], dim=0)
            pairwise_dist = torch.cdist(combined, combined, p=2)
            median_dist = pairwise_dist.median()
            sigmas = [median_dist / 4, median_dist / 2, median_dist, median_dist * 2, median_dist * 4]
    
    mmd_loss = 0.0
    for sigma in sigmas:
        gamma = 1.0 / (2 * sigma ** 2)
        mmd_loss += compute_mmd_loss(source_features, target_features, 
                                   kernel_type='rbf', kernel_num=1, kernel_mul=gamma)
    
    return mmd_loss / len(sigmas)


class MMDLoss(nn.Module):
    """
    PyTorch module wrapper for MMD loss
    """
    def __init__(self, kernel_type='rbf', kernel_num=5, kernel_mul=2.0, use_multi_kernel=False):
        super(MMDLoss, self).__init__()
        self.kernel_type = kernel_type
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.use_multi_kernel = use_multi_kernel
    
    def forward(self, source_features, target_features):
        if self.use_multi_kernel:
            return compute_multi_kernel_mmd(source_features, target_features)
        else:
            return compute_mmd_loss(source_features, target_features, 
                                  self.kernel_type, self.kernel_num, self.kernel_mul)




import torch

def _rbf_kernel_from_sqdist(D2, gammas):
    # D2: [N, M] pairwise squared distances
    # gammas: list[float] or 1D tensor of shape [K]
    if not torch.is_tensor(gammas):
        gammas = torch.tensor(gammas, device=D2.device, dtype=D2.dtype)
    # [K, N, M]
    K = torch.exp(-gammas[:, None, None] * D2[None, :, :])
    # average multiple kernels
    return K.mean(dim=0)  # [N, M]

def mmd_rbf_unbiased(source, target, normalize=True, gammas=None):
    """
    Unbiased MMD^2 with RBF kernel.
    - source: [n, d], target: [m, d]
    - normalize: L2-normalize features along dim=1
    - gammas: None -> median heuristic; else list/1D tensor of gamma(s)
    """
    assert source.dim() == 2 and target.dim() == 2
    device = source.device
    dtype = source.dtype

    x = torch.nn.functional.normalize(source, p=2, dim=1) if normalize else source
    y = torch.nn.functional.normalize(target, p=2, dim=1) if normalize else target

    # pairwise squared distances
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2a·b
    x2 = (x * x).sum(dim=1, keepdim=True)           # [n,1]
    y2 = (y * y).sum(dim=1, keepdim=True)           # [m,1]
    XX = x @ x.t()                                  # [n,n]
    YY = y @ y.t()                                  # [m,m]
    XY = x @ y.t()                                  # [n,m]
    Dxx = (x2 + x2.t() - 2*XX).clamp_min_(0)        # [n,n]
    Dyy = (y2 + y2.t() - 2*YY).clamp_min_(0)        # [m,m]
    Dxy = (x2 + y2.t() - 2*XY).clamp_min_(0)        # [n,m]

    # median heuristic if needed 
    if gammas is None:
        with torch.no_grad():
            med = Dxy.detach().flatten()
            med = med[med > 0].median() if (med > 0).any() else Dxy.detach().median()
            # avoid 0/NaN
            med = med if torch.isfinite(med) and med > 0 else torch.tensor(1.0, device=device, dtype=dtype)
        gamma_med = 1.0 / (2.0 * med)
        gammas = [gamma_med/4, gamma_med/2, gamma_med, gamma_med*2, gamma_med*4]

    Kxx = _rbf_kernel_from_sqdist(Dxx, gammas)
    Kyy = _rbf_kernel_from_sqdist(Dyy, gammas)
    Kxy = _rbf_kernel_from_sqdist(Dxy, gammas)

    n = x.size(0)
    m = y.size(0)

    # unbiased estimation: remove diagonal elements
    if n > 1:
        mmd_xx = (Kxx.sum() - Kxx.diag().sum()) / (n*(n-1))
    else:
        mmd_xx = torch.tensor(0.0, device=device, dtype=dtype)
    if m > 1:
        mmd_yy = (Kyy.sum() - Kyy.diag().sum()) / (m*(m-1))
    else:
        mmd_yy = torch.tensor(0.0, device=device, dtype=dtype)

    mmd_xy = Kxy.mean()

    mmd2 = mmd_xx + mmd_yy - 2*mmd_xy
    return mmd2


def mmd_rbf_biased(x, y, gammas=None, normalize=True):
    if normalize:
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        y = torch.nn.functional.normalize(y, p=2, dim=1)

    x2 = (x * x).sum(1, keepdim=True)
    y2 = (y * y).sum(1, keepdim=True)
    XX = x @ x.t()
    YY = y @ y.t()
    XY = x @ y.t()
    Dxx = (x2 + x2.t() - 2*XX).clamp_min_(0)
    Dyy = (y2 + y2.t() - 2*YY).clamp_min_(0)
    Dxy = (x2 + y2.t() - 2*XY).clamp_min_(0)

    # median heuristic on cross-domain distances
    if gammas is None:
        with torch.no_grad():
            med = Dxy.detach().flatten()
            med = med[med > 0].median() if (med > 0).any() else Dxy.detach().median()
            med = med if torch.isfinite(med) and med > 0 else torch.tensor(1.0, device=x.device, dtype=x.dtype)
        g = 1.0/(2.0*med)
        gammas = [g/2, g, g*2]

    if not torch.is_tensor(gammas):
        gammas = torch.tensor(gammas, device=x.device, dtype=x.dtype)

    Kxx = torch.exp(-gammas[:, None, None] * Dxx[None]).mean(0)
    Kyy = torch.exp(-gammas[:, None, None] * Dyy[None]).mean(0)
    Kxy = torch.exp(-gammas[:, None, None] * Dxy[None]).mean(0)

    mmd2 = Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()
    return mmd2