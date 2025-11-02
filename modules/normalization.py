"""
归一化层实现
包括 Global LayerNorm, Cumulative LayerNorm, Channel-wise LayerNorm
"""

import torch
import torch.nn as nn


class GlobalLayerNorm(nn.Module):
    """
    Global Layer Normalization (gLN)
    
    对整个序列进行归一化，适用于非因果模型。
    
    输入: [B, N, K] 或 [B, N, K, ...]
    输出: [B, N, K] 或 [B, N, K, ...]
    
    归一化公式:
        y = γ * (x - E[x]) / sqrt(Var[x] + ε) + β
    
    其中 E[x] 和 Var[x] 在除 batch 维度外的所有维度上计算
    """
    
    def __init__(self, num_features, eps=1e-8):
        """
        Args:
            num_features: 通道数 (N)
            eps: 用于数值稳定性的小常数
        """
        super(GlobalLayerNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        
        # 可学习的缩放和偏移参数
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1))
    
    def forward(self, x):
        """
        Args:
            x: [B, N, K] - 输入特征
        
        Returns:
            x_norm: [B, N, K] - 归一化后的特征
        """
        # 计算均值和方差（在 N 和 K 维度上）
        # mean, var: [B, 1, 1]
        mean = torch.mean(x, dim=(1, 2), keepdim=True)
        var = torch.var(x, dim=(1, 2), keepdim=True, unbiased=False)
        
        # 归一化
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # 应用可学习参数
        x_norm = self.gamma * x_norm + self.beta
        
        return x_norm


class CumulativeLayerNorm(nn.Module):
    """
    Cumulative Layer Normalization (cLN)
    
    对累积序列进行归一化，适用于因果模型（在线处理）。
    
    输入: [B, N, K]
    输出: [B, N, K]
    
    对于每个时间步 k，只使用 [0, k] 的统计信息进行归一化
    """
    
    def __init__(self, num_features, eps=1e-8):
        """
        Args:
            num_features: 通道数 (N)
            eps: 用于数值稳定性的小常数
        """
        super(CumulativeLayerNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        
        # 可学习的缩放和偏移参数
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1))
    
    def forward(self, x):
        """
        Args:
            x: [B, N, K] - 输入特征
        
        Returns:
            x_norm: [B, N, K] - 归一化后的特征
        """
        batch_size, num_channels, seq_len = x.shape
        
        # 累积和 (cumulative sum)
        x_cumsum = torch.cumsum(x, dim=2)  # [B, N, K]
        x_square_cumsum = torch.cumsum(x ** 2, dim=2)  # [B, N, K]
        
        # 每个位置的样本数
        sample_count = torch.arange(1, seq_len + 1, device=x.device).view(1, 1, -1)  # [1, 1, K]
        
        # 累积均值和方差
        cumulative_mean = x_cumsum / sample_count  # [B, N, K]
        cumulative_var = (x_square_cumsum / sample_count) - cumulative_mean ** 2  # [B, N, K]
        
        # 归一化
        x_norm = (x - cumulative_mean) / torch.sqrt(cumulative_var + self.eps)
        
        # 应用可学习参数
        x_norm = self.gamma * x_norm + self.beta
        
        return x_norm


class ChannelWiseLayerNorm(nn.Module):
    """
    Channel-wise Layer Normalization
    
    对每个通道独立进行归一化。
    
    输入: [B, N, K]
    输出: [B, N, K]
    
    对于每个通道，在时间维度 K 上计算均值和方差
    """
    
    def __init__(self, num_features, eps=1e-8):
        """
        Args:
            num_features: 通道数 (N)
            eps: 用于数值稳定性的小常数
        """
        super(ChannelWiseLayerNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        
        # 可学习的缩放和偏移参数（每个通道一个）
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1))
    
    def forward(self, x):
        """
        Args:
            x: [B, N, K] - 输入特征
        
        Returns:
            x_norm: [B, N, K] - 归一化后的特征
        """
        # 计算均值和方差（在时间维度 K 上）
        # mean, var: [B, N, 1]
        mean = torch.mean(x, dim=2, keepdim=True)
        var = torch.var(x, dim=2, keepdim=True, unbiased=False)
        
        # 归一化
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # 应用可学习参数
        x_norm = self.gamma * x_norm + self.beta
        
        return x_norm


def select_norm(norm_type, num_features, eps=1e-8):
    """
    根据类型选择归一化层
    
    Args:
        norm_type: 归一化类型 ('gLN', 'cLN', 'BN')
        num_features: 特征数量
        eps: 数值稳定性常数
    
    Returns:
        norm_layer: 归一化层
    """
    if norm_type == 'gLN':
        return GlobalLayerNorm(num_features, eps)
    elif norm_type == 'cLN':
        return CumulativeLayerNorm(num_features, eps)
    elif norm_type == 'BN':
        return nn.BatchNorm1d(num_features, eps=eps)
    elif norm_type == 'cBN':
        return ChannelWiseLayerNorm(num_features, eps)
    else:
        raise ValueError(f"Unsupported norm type: {norm_type}")


if __name__ == "__main__":
    # 测试代码
    print("Testing Normalization Layers...")
    
    batch_size = 4
    num_features = 128
    seq_len = 100
    
    # 生成测试数据
    x = torch.randn(batch_size, num_features, seq_len)
    
    # 测试 Global LayerNorm
    gln = GlobalLayerNorm(num_features)
    x_gln = gln(x)
    print(f"\nGlobal LayerNorm:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {x_gln.shape}")
    print(f"  Output mean: {x_gln.mean().item():.6f}")
    print(f"  Output std: {x_gln.std().item():.6f}")
    
    # 测试 Cumulative LayerNorm
    cln = CumulativeLayerNorm(num_features)
    x_cln = cln(x)
    print(f"\nCumulative LayerNorm:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {x_cln.shape}")
    print(f"  Output mean: {x_cln.mean().item():.6f}")
    print(f"  Output std: {x_cln.std().item():.6f}")
    
    # 测试 Channel-wise LayerNorm
    cwln = ChannelWiseLayerNorm(num_features)
    x_cwln = cwln(x)
    print(f"\nChannel-wise LayerNorm:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {x_cwln.shape}")
    print(f"  Output mean: {x_cwln.mean().item():.6f}")
    print(f"  Output std: {x_cwln.std().item():.6f}")
    
    # 测试选择函数
    norm = select_norm('gLN', num_features)
    print(f"\nselect_norm('gLN'): {type(norm).__name__}")
    
    print("\nNormalization layers test passed!")
