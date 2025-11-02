"""
Encoder 模块
1D 卷积编码器，将波形转换为特征表示
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    1D 卷积编码器（无激活函数）
    
    功能：将时域波形转换为高维特征表示
    
    输入: [B, 1, T] 或 [B, T] - 混合波形
    输出: [B, N, K] - 编码特征
    
    其中:
        - B: batch size
        - T: 波形长度
        - N: 滤波器数量 (特征维度)
        - K: 特征序列长度 = (T - L) / stride + 1
    
    特点:
    - 使用线性编码器（无激活函数）
    - 相当于可学习的 STFT
    - 通过 1D 卷积实现滑动窗口
    """
    
    def __init__(self, num_filters=512, kernel_size=16, stride=8):
        """
        Args:
            num_filters (N): 滤波器数量（特征维度）
            kernel_size (L): 卷积核大小（窗口长度）
            stride: 步长（窗口移动步长）
        """
        super(Encoder, self).__init__()
        
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        
        # 1D 卷积（无偏置，无激活）
        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=num_filters,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=False
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, T] 或 [B, 1, T] - 输入波形
        
        Returns:
            w: [B, N, K] - 编码特征
        """
        # 确保输入是 3 维: [B, 1, T]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, T] -> [B, 1, T]
        
        # 编码
        w = self.conv1d(x)  # [B, 1, T] -> [B, N, K]
        
        return w
    
    def get_output_length(self, input_length):
        """
        计算输出序列长度
        
        Args:
            input_length: 输入波形长度
        
        Returns:
            output_length: 输出特征序列长度
        """
        return (input_length - self.kernel_size) // self.stride + 1


if __name__ == "__main__":
    # 测试代码
    print("Testing Encoder...")
    
    # 参数
    batch_size = 4
    sample_rate = 16000
    duration = 4  # 秒
    audio_length = sample_rate * duration  # 64000
    
    num_filters = 512
    kernel_size = 16
    stride = 8
    
    # 生成测试数据
    x = torch.randn(batch_size, audio_length)
    
    # 创建编码器
    encoder = Encoder(
        num_filters=num_filters,
        kernel_size=kernel_size,
        stride=stride
    )
    
    # 编码
    w = encoder(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {w.shape}")
    print(f"Expected output length: {encoder.get_output_length(audio_length)}")
    
    # 计算参数量
    params = sum(p.numel() for p in encoder.parameters())
    print(f"Parameters: {params:,}")
    
    # 理论参数量
    theory_params = 1 * num_filters * kernel_size
    print(f"Theoretical parameters: {theory_params:,}")
    
    # 测试不同输入形状
    print("\nTesting different input shapes:")
    x1 = torch.randn(batch_size, audio_length)  # [B, T]
    w1 = encoder(x1)
    print(f"  Input [B, T]: {x1.shape} -> Output: {w1.shape}")
    
    x2 = torch.randn(batch_size, 1, audio_length)  # [B, 1, T]
    w2 = encoder(x2)
    print(f"  Input [B, 1, T]: {x2.shape} -> Output: {w2.shape}")
    
    print("\nEncoder test passed!")
