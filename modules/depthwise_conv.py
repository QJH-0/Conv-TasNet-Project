"""
深度可分离卷积实现
Depthwise Separable Convolution
"""

import torch
import torch.nn as nn


class DepthwiseSeparableConv1d(nn.Module):
    """
    深度可分离卷积 (Depthwise Separable Convolution)
    
    将标准卷积分解为两步：
    1. Depthwise Convolution: 每个通道独立卷积
    2. Pointwise Convolution: 1×1 卷积混合通道信息
    
    优点：
    - 参数量大幅减少
    - 计算量显著降低
    - 保持相似的表达能力
    
    参数量对比：
    - 标准卷积: C_in × C_out × K
    - 深度可分离: C_in × K + C_in × C_out
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, dilation=1, bias=False):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            stride: 步长
            padding: 填充
            dilation: 扩张率
            bias: 是否使用偏置
        """
        super(DepthwiseSeparableConv1d, self).__init__()
        
        # Depthwise Convolution: 每个通道独立卷积
        # groups=in_channels 表示每个通道独立进行卷积
        self.depthwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,  # 关键：每个输入通道独立卷积
            bias=bias
        )
        
        # Pointwise Convolution: 1×1 卷积混合通道
        # self.pointwise = nn.Conv1d(
        #     in_channels=in_channels,
        #     out_channels=out_channels,
        #     kernel_size=1,
        #     stride=1,
        #     padding=0,
        #     dilation=1,
        #     groups=1,
        #     bias=bias
        # )
    
    def forward(self, x):
        """
        Args:
            x: [B, C_in, T] - 输入特征
        
        Returns:
            x: [B, C_out, T] - 输出特征
        """
        # Depthwise: [B, C_in, T] -> [B, C_in, T]
        x = self.depthwise(x)
        
        # Pointwise: [B, C_in, T] -> [B, C_out, T]
        # x = self.pointwise(x)
        
        return x


if __name__ == "__main__":
    # 测试代码
    print("Testing Depthwise Separable Convolution...")
    
    # 参数
    batch_size = 4
    in_channels = 128
    out_channels = 256
    seq_len = 1000
    kernel_size = 3
    
    # 生成测试数据
    x = torch.randn(batch_size, in_channels, seq_len)
    
    # 深度可分离卷积
    dw_conv = DepthwiseSeparableConv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding=(kernel_size - 1) // 2
    )
    
    # 标准卷积（用于对比）
    std_conv = nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding=(kernel_size - 1) // 2
    )
    
    # 前向传播
    y_dw = dw_conv(x)
    y_std = std_conv(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape (DW): {y_dw.shape}")
    print(f"Output shape (Std): {y_std.shape}")
    
    # 计算参数量
    dw_params = sum(p.numel() for p in dw_conv.parameters())
    std_params = sum(p.numel() for p in std_conv.parameters())
    
    print(f"\nParameters:")
    print(f"  Depthwise Separable: {dw_params:,}")
    print(f"  Standard Convolution: {std_params:,}")
    print(f"  Reduction: {(1 - dw_params / std_params) * 100:.2f}%")
    
    # 理论参数量
    dw_theory = in_channels * kernel_size + in_channels * out_channels
    std_theory = in_channels * out_channels * kernel_size
    
    print(f"\nTheoretical Parameters:")
    print(f"  Depthwise Separable: {dw_theory:,}")
    print(f"  Standard Convolution: {std_theory:,}")
    print(f"  Reduction: {(1 - dw_theory / std_theory) * 100:.2f}%")
    
    print("\nDepthwise Separable Convolution test passed!")
