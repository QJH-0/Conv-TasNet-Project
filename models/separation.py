"""
Separation 模块
TCN 分离网络，生成语音分离掩码
"""

import torch
import torch.nn as nn
import sys
import os

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.normalization import GlobalLayerNorm
from modules.tcn import TCNBlock


class Separation(nn.Module):
    """
    分离模块（包含 LayerNorm）
    
    流程:
    1. LayerNorm（归一化Encoder输出）
    2. Bottleneck (1×1 Conv，降维)
    3. TCN Blocks (R=3，重复3次)
    4. Mask Generation (生成C个说话人的掩码)
    
    输入: [B, N, K] - Encoder输出
    输出: [B, C, N, K] - C个说话人的掩码
    """
    
    def __init__(self, num_speakers=2, encoder_filters=512, 
                 bottleneck_channels=128, hidden_channels=512, 
                 skip_channels=128, kernel_size=3, num_blocks=8, 
                 num_repeats=3, norm_type='gLN', causal=False):
        """
        Args:
            num_speakers (C): 说话人数量
            encoder_filters (N): Encoder滤波器数量
            bottleneck_channels (B): 瓶颈层通道数
            hidden_channels (H): TCN隐藏层通道数
            skip_channels (Sc): 跳跃连接通道数
            kernel_size (P): TCN卷积核大小
            num_blocks (M): 每个TCN块的层数
            num_repeats (R): TCN块重复次数
            norm_type: 归一化类型
            causal: 是否因果卷积
        """
        super(Separation, self).__init__()
        
        self.num_speakers = num_speakers
        self.encoder_filters = encoder_filters
        
        # LayerNorm（归一化Encoder输出）
        self.layer_norm = GlobalLayerNorm(encoder_filters)
        
        # Bottleneck层：1×1卷积降维 N -> B
        self.bottleneck = nn.Conv1d(encoder_filters, bottleneck_channels, 1)
        
        # R个TCN块
        self.tcn_blocks = nn.ModuleList()
        for r in range(num_repeats):
            tcn_block = TCNBlock(
                num_blocks=num_blocks,
                in_channels=bottleneck_channels,
                hidden_channels=hidden_channels,
                skip_channels=skip_channels,
                kernel_size=kernel_size,
                norm_type=norm_type,
                causal=causal
            )
            self.tcn_blocks.append(tcn_block)
        
        # Mask生成：skip_channels -> C * N
        # 论文使用ReLU而非Sigmoid，允许掩码>1以实现信号放大
        self.mask_conv = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(skip_channels, num_speakers * encoder_filters, 1),
            nn.ReLU()  # 掩码值在 [0, +∞)，允许信号放大
        )
    
    def forward(self, encoder_output):
        """
        Args:
            encoder_output: [B, N, K] - Encoder输出
        
        Returns:
            masks: [B, C, N, K] - C个说话人的掩码
        """
        batch_size = encoder_output.shape[0]
        num_filters = encoder_output.shape[1]
        seq_len = encoder_output.shape[2]
        
        # LayerNorm
        x = self.layer_norm(encoder_output)  # [B, N, K]
        
        # Bottleneck
        x = self.bottleneck(x)  # [B, N, K] -> [B, B, K]
        
        # 通过R个TCN块，累积跳跃连接
        skip_sum = 0
        for tcn_block in self.tcn_blocks:
            x, skip = tcn_block(x)  # x: [B, B, K], skip: [B, Sc, K]
            skip_sum = skip_sum + skip
        
        # 生成掩码
        masks = self.mask_conv(skip_sum)  # [B, Sc, K] -> [B, C*N, K]
        
        # Reshape: [B, C*N, K] -> [B, C, N, K]
        masks = masks.view(batch_size, self.num_speakers, num_filters, seq_len)
        
        return masks


if __name__ == "__main__":
    # 测试代码
    print("Testing Separation Module...")
    
    # 参数
    batch_size = 4
    num_speakers = 2
    encoder_filters = 512
    seq_len = 7999
    
    # 生成测试数据（Encoder输出）
    encoder_output = torch.randn(batch_size, encoder_filters, seq_len)
    
    # 创建分离模块
    separation = Separation(
        num_speakers=num_speakers,
        encoder_filters=encoder_filters,
        bottleneck_channels=128,
        hidden_channels=512,
        skip_channels=128,
        kernel_size=3,
        num_blocks=8,
        num_repeats=3,
        norm_type='gLN',
        causal=False
    )
    
    # 前向传播
    masks = separation(encoder_output)
    
    print(f"Input shape: {encoder_output.shape}")
    print(f"Output shape: {masks.shape}")
    print(f"Expected shape: [{batch_size}, {num_speakers}, {encoder_filters}, {seq_len}]")
    
    # 检查掩码值范围
    print(f"\nMask statistics:")
    print(f"  Min: {masks.min().item():.4f}")
    print(f"  Max: {masks.max().item():.4f}")
    print(f"  Mean: {masks.mean().item():.4f}")
    
    # 计算参数量
    params = sum(p.numel() for p in separation.parameters())
    print(f"\nParameters: {params:,}")
    
    # 分解参数
    bottleneck_params = sum(p.numel() for p in separation.bottleneck.parameters())
    tcn_params = sum(sum(p.numel() for p in block.parameters()) 
                     for block in separation.tcn_blocks)
    mask_params = sum(p.numel() for p in separation.mask_conv.parameters())
    
    print(f"\nParameter breakdown:")
    print(f"  Bottleneck: {bottleneck_params:,}")
    print(f"  TCN blocks: {tcn_params:,}")
    print(f"  Mask generation: {mask_params:,}")
    
    print("\nSeparation Module test passed!")
