"""
BTCN (Binary Temporal Convolutional Network) 模块
二值化的时序卷积网络 - 策略2：二值化深度卷积 + 1×1卷积
"""

import torch
import torch.nn as nn
from modules.normalization import select_norm
from modules.binarization import BinaryDepthwiseConv1d, BinaryConv1d


class BinaryTemporalBlock(nn.Module):
    """
    二值化时序卷积块 - 策略2
    
    结构:
        输入 → Binary 1×1 Conv → Norm → PReLU  (二值化)
             → Binary D-Conv → Norm → PReLU     (二值化)
             → Binary 1×1 Conv → Residual + Skip (二值化)
    
    特点:
    - 二值化所有卷积层（1×1卷积 + 深度卷积）
    - 归一化层和激活层保持全精度
    - 残差连接保持全精度
    
    二值化率: 约94%的参数
    """
    
    def __init__(self, in_channels, hidden_channels, skip_channels, 
                 kernel_size, dilation, norm_type='gLN', causal=False):
        """
        Args:
            in_channels: 输入通道数 (B)
            hidden_channels: 隐藏层通道数 (H)
            skip_channels: 跳跃连接通道数 (Sc)
            kernel_size: 卷积核大小 (P)
            dilation: 扩张因子 (d)
            norm_type: 归一化类型
            causal: 是否因果卷积
        """
        super(BinaryTemporalBlock, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.skip_channels = skip_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.causal = causal
        
        # 计算填充
        if causal:
            self.padding = (kernel_size - 1) * dilation
        else:
            self.padding = (kernel_size - 1) * dilation // 2
        
        # 二值化1×1卷积：bottleneck -> hidden
        self.binary_conv1x1_1 = BinaryConv1d(in_channels, hidden_channels, kernel_size=1)

        self.prelu1 = nn.PReLU()
        self.norm1 = select_norm(norm_type, hidden_channels)

        # 二值化深度卷积
        self.binary_dw_conv = BinaryDepthwiseConv1d(
            channels=hidden_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            dilation=dilation,
            bias=False
        )

        self.prelu2 = nn.PReLU()
        self.norm2 = select_norm(norm_type, hidden_channels)

        # 二值化1×1卷积：hidden -> bottleneck (residual path)
        self.binary_conv1x1_2 = BinaryConv1d(hidden_channels, in_channels, kernel_size=1)
        
        # 二值化1×1卷积：hidden -> skip_channels (skip connection)
        self.binary_skip_conv = BinaryConv1d(hidden_channels, skip_channels, kernel_size=1)
    
    def forward(self, x):
        """
        Args:
            x: [B, in_channels, T] - 输入特征
        
        Returns:
            residual: [B, in_channels, T] - 残差输出
            skip: [B, skip_channels, T] - 跳跃连接输出
        """
        # 保存输入用于残差连接
        residual = x
        
        # Binary 1×1 Conv + Norm + PReLU
        x = self.binary_conv1x1_1(x)
        x = self.norm1(x)
        x = self.prelu1(x)
        
        # Binary Depthwise Conv
        x = self.binary_dw_conv(x)
        
        # 如果是因果卷积，需要裁剪多余的填充
        if self.causal and self.padding > 0:
            x = x[:, :, :-self.padding]
        
        # Norm + PReLU
        x = self.norm2(x)
        x = self.prelu2(x)
        
        # Binary Skip Conv
        skip = self.binary_skip_conv(x)
        
        # Binary 1×1 Conv (残差路径)
        x = self.binary_conv1x1_2(x)
        
        # 残差连接 (全精度)
        residual = residual + x
        
        return residual, skip


class BTCNBlock(nn.Module):
    """
    BTCN 块：包含多个 BinaryTemporalBlock - 与原TCNBlock结构完全一致
    
    结构:
        输入 → BinaryTemporalBlock (d=1)
             → BinaryTemporalBlock (d=2)
             → BinaryTemporalBlock (d=4)
             → ...
             → BinaryTemporalBlock (d=2^(M-1))
             → 输出
    
    每层的扩张因子呈指数增长：1, 2, 4, 8, 16, 32, 64, 128, ...
    """
    
    def __init__(self, num_blocks, in_channels, hidden_channels, skip_channels,
                 kernel_size, norm_type='gLN', causal=False):
        """
        Args:
            num_blocks: BTCN块中的层数 (M)
            in_channels: 输入通道数 (B)
            hidden_channels: 隐藏层通道数 (H)
            skip_channels: 跳跃连接通道数 (Sc)
            kernel_size: 卷积核大小 (P)
            norm_type: 归一化类型
            causal: 是否因果卷积
        """
        super(BTCNBlock, self).__init__()
        
        self.num_blocks = num_blocks
        
        # 创建多个 BinaryTemporalBlock
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            dilation = 2 ** i  # 扩张因子：1, 2, 4, 8, ...
            block = BinaryTemporalBlock(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                skip_channels=skip_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                norm_type=norm_type,
                causal=causal
            )
            self.blocks.append(block)
    
    def forward(self, x):
        """
        Args:
            x: [B, in_channels, T] - 输入特征
        
        Returns:
            x: [B, in_channels, T] - 残差输出
            skip_sum: [B, skip_channels, T] - 跳跃连接求和
        """
        skip_connections = []
        
        # 通过所有 BinaryTemporalBlock
        for block in self.blocks:
            x, skip = block(x)
            skip_connections.append(skip)
        
        # 求和所有跳跃连接
        skip_sum = torch.stack(skip_connections, dim=0).sum(dim=0)
        
        return x, skip_sum


if __name__ == "__main__":
    # 测试代码
    print("Testing BTCN Module...")
    print("=" * 80)
    
    # 参数
    batch_size = 4
    in_channels = 128
    hidden_channels = 512
    skip_channels = 128
    kernel_size = 3
    num_blocks = 8
    seq_len = 1000
    
    # 生成测试数据
    x = torch.randn(batch_size, in_channels, seq_len)
    
    # 测试 BinaryTemporalBlock
    print("\n1. Testing BinaryTemporalBlock...")
    binary_temporal_block = BinaryTemporalBlock(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        skip_channels=skip_channels,
        kernel_size=kernel_size,
        dilation=2,
        norm_type='gLN',
        dropout=0.1
    )
    
    residual, skip = binary_temporal_block(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Residual shape: {residual.shape}")
    print(f"  Skip shape: {skip.shape}")
    assert residual.shape == x.shape, "Residual shape mismatch!"
    assert skip.shape == (batch_size, skip_channels, seq_len), "Skip shape mismatch!"
    print("  ✓ BinaryTemporalBlock passed!")
    
    # 测试 BTCNBlock
    print("\n2. Testing BTCNBlock...")
    btcn_block = BTCNBlock(
        num_blocks=num_blocks,
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        skip_channels=skip_channels,
        kernel_size=kernel_size,
        norm_type='gLN',
        dropout=0.1
    )
    
    output, skip_sum = btcn_block(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Skip sum shape: {skip_sum.shape}")
    assert output.shape == x.shape, "Output shape mismatch!"
    assert skip_sum.shape == (batch_size, skip_channels, seq_len), "Skip sum shape mismatch!"
    print("  ✓ BTCNBlock passed!")
    
    # 计算感受野
    receptive_field = 1 + sum(2 ** i * (kernel_size - 1) for i in range(num_blocks))
    print(f"\n3. Receptive Field:")
    print(f"  BTCN blocks: {num_blocks}")
    print(f"  Kernel size: {kernel_size}")
    print(f"  Receptive field: {receptive_field} samples")
    print(f"  Time span @ 8kHz: {receptive_field / 8000:.3f} seconds")
    print(f"  Time span @ 16kHz: {receptive_field / 16000:.3f} seconds")
    
    # 计算参数量
    total_params = sum(p.numel() for p in btcn_block.parameters())
    
    # 统计二值化参数和全精度参数
    binary_params = 0
    full_precision_params = 0
    for name, param in btcn_block.named_parameters():
        if 'binary_dw_conv' in name:
            binary_params += param.numel()
        else:
            full_precision_params += param.numel()
    
    print(f"\n4. Parameters:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Binary parameters: {binary_params:,} ({binary_params/total_params*100:.1f}%)")
    print(f"  Full-precision parameters: {full_precision_params:,} ({full_precision_params/total_params*100:.1f}%)")
    
    # 估算模型大小（假设二值化参数存储为1-bit）
    model_size_fp32 = total_params * 4 / (1024 ** 2)  # FP32
    model_size_binary = (binary_params / 32 + full_precision_params) * 4 / (1024 ** 2)  # 二值化
    print(f"\n5. Model Size:")
    print(f"  FP32: {model_size_fp32:.2f} MB")
    print(f"  Binary (1-bit for binary params): {model_size_binary:.2f} MB")
    print(f"  Compression ratio: {model_size_fp32/model_size_binary:.2f}x")
    
    # 测试反向传播
    print("\n6. Testing backward pass...")
    x_test = torch.randn(batch_size, in_channels, seq_len, requires_grad=True)
    output, skip_sum = btcn_block(x_test)
    loss = output.sum() + skip_sum.sum()
    loss.backward()
    
    print(f"  Gradient exists: {x_test.grad is not None}")
    print(f"  Gradient shape: {x_test.grad.shape}")
    assert x_test.grad is not None, "Gradient should exist!"
    print("  ✓ Backward pass passed!")
    
    print("\n" + "=" * 80)
    print("All tests passed! ✓")

