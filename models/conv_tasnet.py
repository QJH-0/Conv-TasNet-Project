"""
Conv-TasNet 主模型
完整的端到端语音分离模型
"""

import torch
import torch.nn as nn
from models.encoder import Encoder
from models.decoder import Decoder
from models.separation import Separation


class ConvTasNet(nn.Module):
    """
    Conv-TasNet: 完全卷积的时域语音分离网络
    
    架构:
        混合波形 x(t)
            ↓
        Encoder (1D Conv)
            ↓
        编码特征 w
            ↓
        Separation (TCN) → 掩码 m1, m2, ..., mC
            ↓
        掩码相乘 d1=w⊙m1, d2=w⊙m2, ...
            ↓
        Decoder (1D TransConv)
            ↓
        分离波形 ŝ1(t), ŝ2(t), ..., ŝC(t)
    
    特点:
    - 端到端训练
    - 完全卷积（无RNN）
    - 使用掩码估计
    - PIT训练避免排列问题
    """
    
    def __init__(self, num_speakers=2, encoder_filters=512, encoder_kernel_size=16, 
                 encoder_stride=8, bottleneck_channels=128, hidden_channels=512,
                 skip_channels=128, kernel_size=3, num_blocks=8, num_repeats=3,
                 norm_type='gLN', causal=False):
        """
        Args:
            num_speakers (C): 说话人数量
            encoder_filters (N): Encoder滤波器数量
            encoder_kernel_size (L): Encoder卷积核大小
            encoder_stride: Encoder步长
            bottleneck_channels (B): 瓶颈层通道数
            hidden_channels (H): TCN隐藏层通道数
            skip_channels (Sc): 跳跃连接通道数
            kernel_size (P): TCN卷积核大小
            num_blocks (M): 每个TCN块的层数
            num_repeats (R): TCN块重复次数
            norm_type: 归一化类型 ('gLN', 'cLN', 'BN')
            causal: 是否因果卷积
        """
        super(ConvTasNet, self).__init__()
        
        self.num_speakers = num_speakers
        
        # Encoder: 波形 → 特征
        self.encoder = Encoder(
            num_filters=encoder_filters,
            kernel_size=encoder_kernel_size,
            stride=encoder_stride
        )
        
        # Separation: 特征 → 掩码
        self.separation = Separation(
            num_speakers=num_speakers,
            encoder_filters=encoder_filters,
            bottleneck_channels=bottleneck_channels,
            hidden_channels=hidden_channels,
            skip_channels=skip_channels,
            kernel_size=kernel_size,
            num_blocks=num_blocks,
            num_repeats=num_repeats,
            norm_type=norm_type,
            causal=causal
        )
        
        # Decoder: 掩码特征 → 波形
        self.decoder = Decoder(
            num_filters=encoder_filters,
            kernel_size=encoder_kernel_size,
            stride=encoder_stride
        )
    
    def forward(self, mixture):
        """
        前向传播
        
        Args:
            mixture: [B, T] - 混合波形
        
        Returns:
            separated_sources: [B, C, T] - 分离后的C个说话人波形
        """
        batch_size = mixture.shape[0]
        mixture_length = mixture.shape[-1]
        
        # 1. Encoder: 波形 → 特征
        encoder_output = self.encoder(mixture)  # [B, T] -> [B, N, K]
        
        # 2. Separation: 特征 → 掩码
        masks = self.separation(encoder_output)  # [B, N, K] -> [B, C, N, K]
        
        # 3. 掩码相乘: w ⊙ m_i
        masked_features = encoder_output.unsqueeze(1) * masks  # [B, C, N, K]
        
        # 4. Decoder: 掩码特征 → 波形
        separated_sources = []
        for i in range(self.num_speakers):
            # 对每个说话人的掩码特征进行解码
            source = self.decoder(masked_features[:, i, :, :])  # [B, N, K] -> [B, T']
            separated_sources.append(source)
        
        # Stack: List[[B, T']] -> [B, C, T']
        separated_sources = torch.stack(separated_sources, dim=1)
        
        # 5. 裁剪或填充到原始长度
        separated_sources = self._pad_or_trim(separated_sources, mixture_length)
        
        return separated_sources
    
    def _pad_or_trim(self, sources, target_length):
        """
        裁剪或填充到目标长度
        
        Args:
            sources: [B, C, T']
            target_length: T
        
        Returns:
            sources: [B, C, T]
        """
        current_length = sources.shape[-1]
        
        if current_length > target_length:
            # 裁剪
            sources = sources[:, :, :target_length]
        elif current_length < target_length:
            # 填充
            pad_length = target_length - current_length
            sources = torch.nn.functional.pad(sources, (0, pad_length))
        
        return sources
    
    @classmethod
    def from_config(cls, config):
        """
        从配置字典创建模型
        
        Args:
            config: 配置字典
        
        Returns:
            model: Conv-TasNet模型
        """
        model_config = config['model']
        
        return cls(
            num_speakers=model_config['separation']['num_speakers'],
            encoder_filters=model_config['encoder']['num_filters'],
            encoder_kernel_size=model_config['encoder']['kernel_size'],
            encoder_stride=model_config['encoder']['stride'],
            bottleneck_channels=model_config['separation']['bottleneck_channels'],
            hidden_channels=model_config['separation']['hidden_channels'],
            skip_channels=model_config['separation']['skip_channels'],
            kernel_size=model_config['separation']['kernel_size'],
            num_blocks=model_config['separation']['num_blocks'],
            num_repeats=model_config['separation']['num_repeats'],
            norm_type=model_config['separation']['norm_type'],
            causal=model_config['separation']['causal']
        )


if __name__ == "__main__":
    # 测试代码
    print("Testing Conv-TasNet Model...")
    
    # 参数
    batch_size = 2
    sample_rate = 16000
    duration = 4
    audio_length = sample_rate * duration  # 64000
    num_speakers = 2
    
    # 生成测试数据（混合波形）
    mixture = torch.randn(batch_size, audio_length)
    
    # 创建模型
    model = ConvTasNet(
        num_speakers=num_speakers,
        encoder_filters=512,
        encoder_kernel_size=16,
        encoder_stride=8,
        bottleneck_channels=128,
        hidden_channels=512,
        skip_channels=128,
        kernel_size=3,
        num_blocks=8,
        num_repeats=3,
        norm_type='gLN',
        causal=False
    )
    
    print(f"\nModel created successfully!")
    print(f"Input shape: {mixture.shape}")
    
    # 前向传播
    with torch.no_grad():
        separated_sources = model(mixture)
    
    print(f"Output shape: {separated_sources.shape}")
    print(f"Expected shape: [{batch_size}, {num_speakers}, {audio_length}]")
    
    # 验证输出长度
    assert separated_sources.shape == (batch_size, num_speakers, audio_length), \
        f"Output shape mismatch!"
    print("✓ Output shape correct!")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    separation_params = sum(p.numel() for p in model.separation.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Encoder parameters: {encoder_params:,}")
    print(f"  Separation parameters: {separation_params:,}")
    print(f"  Decoder parameters: {decoder_params:,}")
    
    # 估算模型大小
    model_size_mb = total_params * 4 / (1024 ** 2)  # 假设float32
    print(f"  Model size: {model_size_mb:.2f} MB")
    
    # 测试从配置创建
    print("\nTesting from_config...")
    config = {
        'model': {
            'encoder': {
                'num_filters': 512,
                'kernel_size': 16,
                'stride': 8
            },
            'separation': {
                'num_speakers': 2,
                'bottleneck_channels': 128,
                'hidden_channels': 512,
                'skip_channels': 128,
                'kernel_size': 3,
                'num_blocks': 8,
                'num_repeats': 3,
                'norm_type': 'gLN',
                'causal': False
            }
        }
    }
    
    model_from_config = ConvTasNet.from_config(config)
    with torch.no_grad():
        output = model_from_config(mixture)
    print(f"✓ Model created from config successfully!")
    print(f"  Output shape: {output.shape}")
    
    print("\nConv-TasNet Model test passed!")
