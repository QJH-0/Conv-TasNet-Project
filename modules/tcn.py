"""
TCN (Temporal Convolutional Network) 模块
时序卷积网络实现
"""

import torch
import torch.nn as nn
from modules.normalization import select_norm


class TemporalBlock(nn.Module):
    """
    单个时序卷积块 (Temporal Block)
    
    结构:
        输入 → 1×1 Conv → Norm → PReLU
             → D-Conv → Norm → PReLU  
             → 1×1 Conv → Residual + Skip
    
    包含:
    - 残差连接 (Residual Connection)
    - 跳跃连接 (Skip Connection)
    - 扩张卷积 (Dilated Convolution)
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
        super(TemporalBlock, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.skip_channels = skip_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.causal = causal
        
        # 计算填充
        if causal:
            # 因果卷积：只能看到过去的信息
            self.padding = (kernel_size - 1) * dilation
        else:
            # 非因果卷积：可以看到过去和未来的信息
            self.padding = (kernel_size - 1) * dilation // 2
        
        # 1×1 卷积：bottleneck -> hidden
        self.conv1x1_1 = nn.Conv1d(in_channels, hidden_channels, 1)

        self.prelu1 = nn.PReLU()
        self.norm1 = select_norm(norm_type, hidden_channels)

        # 深度卷积（Depthwise Conv）
        self.dw_conv = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            dilation=dilation,
            groups=hidden_channels,  # 关键：groups=channels 使其成为Depthwise Conv
            bias=False
        )

        self.prelu2 = nn.PReLU()
        self.norm2 = select_norm(norm_type, hidden_channels)

        # 1×1 卷积：hidden -> bottleneck (residual path)
        self.conv1x1_2 = nn.Conv1d(hidden_channels, in_channels, 1)
        
        # 1×1 卷积：hidden -> skip_channels (skip connection)
        self.skip_conv = nn.Conv1d(hidden_channels, skip_channels, 1)
    
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
        
        # 1×1 Conv + Norm + PReLU
        x = self.conv1x1_1(x)
        x = self.norm1(x)
        x = self.prelu1(x)
        
        # Depthwise Separable Conv + Norm + PReLU
        x = self.dw_conv(x)
        
        # 如果是因果卷积，需要裁剪多余的填充
        if self.causal:
            x = x[:, :, :-self.padding] if self.padding > 0 else x
        
        x = self.norm2(x)
        x = self.prelu2(x)
        
        # 跳跃连接
        skip = self.skip_conv(x)
        
        # 1×1 Conv (残差路径)
        x = self.conv1x1_2(x)
        
        # 残差连接
        residual = residual + x
        
        return residual, skip


class TCNBlock(nn.Module):
    """
    TCN 块：包含多个 TemporalBlock
    
    结构:
        输入 → TemporalBlock (d=1)
             → TemporalBlock (d=2)
             → TemporalBlock (d=4)
             → ...
             → TemporalBlock (d=2^(M-1))
             → 输出
    
    每层的扩张因子呈指数增长：1, 2, 4, 8, 16, 32, 64, 128, ...
    """
    
    def __init__(self, num_blocks, in_channels, hidden_channels, skip_channels,
                 kernel_size, norm_type='gLN', causal=False):
        """
        Args:
            num_blocks: TCN块中的层数 (M)
            in_channels: 输入通道数 (B)
            hidden_channels: 隐藏层通道数 (H)
            skip_channels: 跳跃连接通道数 (Sc)
            kernel_size: 卷积核大小 (P)
            norm_type: 归一化类型
            causal: 是否因果卷积
        """
        super(TCNBlock, self).__init__()
        
        self.num_blocks = num_blocks
        
        # 创建多个 TemporalBlock
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            dilation = 2 ** i  # 扩张因子：1, 2, 4, 8, ...
            block = TemporalBlock(
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
        
        # 通过所有 TemporalBlock
        for block in self.blocks:
            x, skip = block(x)
            skip_connections.append(skip)
        
        # 求和所有跳跃连接
        skip_sum = torch.stack(skip_connections, dim=0).sum(dim=0)
        
        return x, skip_sum


if __name__ == "__main__":
    # 测试代码
    print("Testing TCN Module...")
    
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
    
    # 测试 TemporalBlock
    print("\n1. Testing TemporalBlock...")
    temporal_block = TemporalBlock(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        skip_channels=skip_channels,
        kernel_size=kernel_size,
        dilation=2,
        norm_type='gLN'
    )
    
    residual, skip = temporal_block(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Residual shape: {residual.shape}")
    print(f"  Skip shape: {skip.shape}")
    
    # 测试 TCNBlock
    print("\n2. Testing TCNBlock...")
    tcn_block = TCNBlock(
        num_blocks=num_blocks,
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        skip_channels=skip_channels,
        kernel_size=kernel_size,
        norm_type='gLN'
    )
    
    output, skip_sum = tcn_block(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Skip sum shape: {skip_sum.shape}")
    
    # 计算感受野
    receptive_field = 1 + sum(2 ** i * (kernel_size - 1) for i in range(num_blocks))
    print(f"\n3. Receptive Field:")
    print(f"  TCN blocks: {num_blocks}")
    print(f"  Kernel size: {kernel_size}")
    print(f"  Receptive field: {receptive_field} samples")
    print(f"  Time span @ 16kHz: {receptive_field / 16000:.3f} seconds")
    
    # 计算参数量
    params = sum(p.numel() for p in tcn_block.parameters())
    print(f"\n4. Parameters:")
    print(f"  Total parameters: {params:,}")
    
    print("\nTCN Module test passed!")
