"""
Decoder 模块
1D 转置卷积解码器，将特征转换回波形
"""

import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    1D 转置卷积解码器
    
    功能：将特征表示转换回时域波形
    
    输入: [B, N, K] - 掩码后的特征
    输出: [B, T] - 重构波形
    
    其中:
        - B: batch size
        - N: 滤波器数量 (特征维度)
        - K: 特征序列长度
        - T: 波形长度 ≈ K * stride
    
    特点:
    - 使用转置卷积（反卷积）
    - 与Encoder结构对称
    - overlap-add 重构波形
    """
    
    def __init__(self, num_filters=512, kernel_size=16, stride=8):
        """
        Args:
            num_filters (N): 滤波器数量（特征维度）
            kernel_size (L): 卷积核大小
            stride: 步长
        """
        super(Decoder, self).__init__()
        
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        
        # 1D 转置卷积（无偏置，无激活）
        self.conv_transpose1d = nn.ConvTranspose1d(
            in_channels=num_filters,
            out_channels=1,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=False
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, N, K] - 输入特征（掩码后的编码特征）
        
        Returns:
            s: [B, T] - 重构波形
        """
        # 解码
        s = self.conv_transpose1d(x)  # [B, N, K] -> [B, 1, T]
        
        # 移除通道维度
        s = s.squeeze(1)  # [B, 1, T] -> [B, T]
        
        return s
    
    def get_output_length(self, input_length):
        """
        计算输出波形长度
        
        Args:
            input_length: 输入特征序列长度
        
        Returns:
            output_length: 输出波形长度
        """
        return (input_length - 1) * self.stride + self.kernel_size


if __name__ == "__main__":
    # 测试代码
    print("Testing Decoder...")
    
    # 参数
    batch_size = 4
    num_filters = 512
    kernel_size = 16
    stride = 8
    feature_length = 7999  # 编码后的特征长度
    
    # 生成测试数据（编码后的特征）
    w = torch.randn(batch_size, num_filters, feature_length)
    
    # 创建解码器
    decoder = Decoder(
        num_filters=num_filters,
        kernel_size=kernel_size,
        stride=stride
    )
    
    # 解码
    s = decoder(w)
    
    print(f"Input shape: {w.shape}")
    print(f"Output shape: {s.shape}")
    print(f"Expected output length: {decoder.get_output_length(feature_length)}")
    
    # 计算参数量
    params = sum(p.numel() for p in decoder.parameters())
    print(f"Parameters: {params:,}")
    
    # 理论参数量
    theory_params = num_filters * 1 * kernel_size
    print(f"Theoretical parameters: {theory_params:,}")
    
    # 测试 Encoder-Decoder 配对
    print("\nTesting Encoder-Decoder pair:")
    from encoder import Encoder
    
    # 创建编码器
    encoder = Encoder(
        num_filters=num_filters,
        kernel_size=kernel_size,
        stride=stride
    )
    
    # 原始音频
    sample_rate = 16000
    duration = 4
    audio_length = sample_rate * duration
    x_original = torch.randn(batch_size, audio_length)
    
    # 编码
    w_encoded = encoder(x_original)
    
    # 解码
    x_reconstructed = decoder(w_encoded)
    
    print(f"  Original audio: {x_original.shape}")
    print(f"  Encoded features: {w_encoded.shape}")
    print(f"  Reconstructed audio: {x_reconstructed.shape}")
    
    # 检查长度匹配
    if x_reconstructed.shape[-1] != x_original.shape[-1]:
        print(f"  Warning: Length mismatch!")
        print(f"    Original: {x_original.shape[-1]}")
        print(f"    Reconstructed: {x_reconstructed.shape[-1]}")
        print(f"    Difference: {abs(x_reconstructed.shape[-1] - x_original.shape[-1])}")
    else:
        print(f"  ✓ Length matched!")
    
    print("\nDecoder test passed!")
