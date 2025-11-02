"""
SI-SNR (Scale-Invariant Signal-to-Noise Ratio) 损失函数
"""

import torch
import torch.nn as nn


class SI_SNR_Loss(nn.Module):
    """
    Scale-Invariant Signal-to-Noise Ratio (SI-SNR) 损失函数
    
    SI-SNR 是一种尺度不变的信噪比指标，用于衡量语音分离质量。
    相比传统 SNR，SI-SNR 对信号的缩放不敏感。
    
    公式:
        s_target = <ŝ, s> / ||s||² * s
        e_noise = ŝ - s_target
        SI-SNR = 10 * log10(||s_target||² / ||e_noise||²)
    
    其中:
        - ŝ: 预测信号
        - s: 真实信号
        - <·,·>: 内积
        - ||·||: L2范数
    """
    
    def __init__(self, eps=1e-8):
        """
        Args:
            eps: 用于数值稳定性的小常数
        """
        super(SI_SNR_Loss, self).__init__()
        self.eps = eps
    
    def forward(self, estimation, target):
        """
        计算 SI-SNR 损失
        
        Args:
            estimation: [B, T] - 预测的语音信号
            target: [B, T] - 真实的语音信号
        
        Returns:
            loss: 负的 SI-SNR（用于最小化）
        """
        # 确保输入形状正确
        assert estimation.shape == target.shape, \
            f"Shape mismatch: estimation {estimation.shape} vs target {target.shape}"
        
        # 零均值化（去除直流分量）
        estimation = estimation - torch.mean(estimation, dim=-1, keepdim=True)
        target = target - torch.mean(target, dim=-1, keepdim=True)
        
        # 计算目标信号的投影
        # s_target = <ŝ, s> / ||s||² * s
        inner_product = torch.sum(estimation * target, dim=-1, keepdim=True)  # [B, 1]
        target_norm_square = torch.sum(target ** 2, dim=-1, keepdim=True) + self.eps  # [B, 1]
        s_target = (inner_product / target_norm_square) * target  # [B, T]
        
        # 计算噪声信号
        # e_noise = ŝ - s_target
        e_noise = estimation - s_target  # [B, T]
        
        # 计算 SI-SNR
        target_power = torch.sum(s_target ** 2, dim=-1) + self.eps  # [B]
        noise_power = torch.sum(e_noise ** 2, dim=-1) + self.eps  # [B]
        si_snr = 10 * torch.log10(target_power / noise_power)  # [B]
        
        # 返回负的 SI-SNR 作为损失（因为我们要最小化损失）
        return -torch.mean(si_snr)
    
    def calculate_si_snr(self, estimation, target):
        """
        计算 SI-SNR 指标（正值，用于评估）
        
        Args:
            estimation: [B, T] or [T] - 预测的语音信号
            target: [B, T] or [T] - 真实的语音信号
        
        Returns:
            si_snr: SI-SNR 值（dB）
        """
        if estimation.dim() == 1:
            estimation = estimation.unsqueeze(0)
            target = target.unsqueeze(0)
        
        # 零均值化
        estimation = estimation - torch.mean(estimation, dim=-1, keepdim=True)
        target = target - torch.mean(target, dim=-1, keepdim=True)
        
        # 计算目标信号的投影
        inner_product = torch.sum(estimation * target, dim=-1, keepdim=True)
        target_norm_square = torch.sum(target ** 2, dim=-1, keepdim=True) + self.eps
        s_target = (inner_product / target_norm_square) * target
        
        # 计算噪声信号
        e_noise = estimation - s_target
        
        # 计算 SI-SNR
        target_power = torch.sum(s_target ** 2, dim=-1) + self.eps
        noise_power = torch.sum(e_noise ** 2, dim=-1) + self.eps
        si_snr = 10 * torch.log10(target_power / noise_power)
        
        return torch.mean(si_snr).item()


if __name__ == "__main__":
    # 测试代码
    print("Testing SI-SNR Loss...")
    
    # 创建损失函数
    criterion = SI_SNR_Loss()
    
    # 生成测试数据
    batch_size = 4
    length = 16000  # 1秒音频 @ 16kHz
    
    # 情况1: 完美预测（SI-SNR应该很高）
    target = torch.randn(batch_size, length)
    estimation = target.clone()
    loss = criterion(estimation, target)
    si_snr = criterion.calculate_si_snr(estimation, target)
    print(f"Perfect prediction - Loss: {loss.item():.4f}, SI-SNR: {si_snr:.4f} dB")
    
    # 情况2: 加入噪声（SI-SNR应该下降）
    noise = torch.randn_like(target) * 0.1
    estimation_noisy = target + noise
    loss = criterion(estimation_noisy, target)
    si_snr = criterion.calculate_si_snr(estimation_noisy, target)
    print(f"With noise - Loss: {loss.item():.4f}, SI-SNR: {si_snr:.4f} dB")
    
    # 情况3: 随机预测（SI-SNR应该很低）
    estimation_random = torch.randn_like(target)
    loss = criterion(estimation_random, target)
    si_snr = criterion.calculate_si_snr(estimation_random, target)
    print(f"Random prediction - Loss: {loss.item():.4f}, SI-SNR: {si_snr:.4f} dB")
    
    print("\nSI-SNR Loss test passed!")
