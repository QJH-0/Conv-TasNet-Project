"""
音频处理工具函数
"""

import torch
import torchaudio
import numpy as np


def load_audio(file_path, sample_rate=16000):
    """
    加载音频文件
    
    Args:
        file_path: 音频文件路径
        sample_rate: 目标采样率
    
    Returns:
        audio: [T] - 音频信号
        sr: 采样率
    """
    # 加载音频
    audio, sr = torchaudio.load(file_path)
    
    # 转换为单声道
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    
    # 重采样
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        audio = resampler(audio)
        sr = sample_rate
    
    audio = audio.squeeze(0)  # [T]
    
    return audio, sr


def save_audio(file_path, audio, sample_rate=16000):
    """
    保存音频文件
    
    Args:
        file_path: 保存路径
        audio: [T] 或 [1, T] - 音频信号
        sample_rate: 采样率
    """
    # 确保是 2D tensor
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    
    # 转换为 CPU tensor
    if audio.is_cuda:
        audio = audio.cpu()
    
    # 保存
    torchaudio.save(file_path, audio, sample_rate)


def normalize_audio(audio, target_level=-25.0):
    """
    归一化音频到目标响度
    
    Args:
        audio: [T] - 音频信号
        target_level: 目标响度（dBFS）
    
    Returns:
        audio: [T] - 归一化后的音频
    """
    # 计算当前RMS
    rms = torch.sqrt(torch.mean(audio ** 2))
    
    # 计算当前响度
    current_level = 20 * torch.log10(rms + 1e-8)
    
    # 计算增益
    gain = target_level - current_level
    gain = 10 ** (gain / 20)
    
    # 应用增益
    audio = audio * gain
    
    # 防止削波
    max_val = torch.max(torch.abs(audio))
    if max_val > 0.99:
        audio = audio * 0.99 / max_val
    
    return audio


def mix_audio(audio1, audio2, snr_db):
    """
    混合两个音频信号（旧版本，保留向后兼容）
    
    Args:
        audio1: [T] - 音频信号1
        audio2: [T] - 音频信号2
        snr_db: 信噪比（dB）
    
    Returns:
        mixture: [T] - 混合音频
    """
    # 确保长度相同
    min_len = min(len(audio1), len(audio2))
    audio1 = audio1[:min_len]
    audio2 = audio2[:min_len]
    
    # 计算audio1的能量
    energy1 = torch.sum(audio1 ** 2)
    
    # 计算audio2需要的能量（根据SNR）
    target_energy2 = energy1 / (10 ** (snr_db / 10))
    
    # 缩放audio2
    current_energy2 = torch.sum(audio2 ** 2)
    scale = torch.sqrt(target_energy2 / (current_energy2 + 1e-8))
    audio2_scaled = audio2 * scale
    
    # 混合
    mixture = audio1 + audio2_scaled
    
    # 归一化防止削波
    max_val = torch.max(torch.abs(mixture))
    if max_val > 0.99:
        mixture = mixture * 0.99 / max_val
    
    return mixture


def mix_audio_with_snr(audio1, audio2, snr_db):
    """
    根据SNR混合两个音频（正确实现，保持SNR不被破坏）
    
    SNR_dB = 10 * log10(E1 / E2)
    其中 E1, E2 分别是 audio1 和 audio2 的能量
    
    Args:
        audio1: [T] - 音频信号1（参考信号）
        audio2: [T] - 音频信号2（将被缩放）
        snr_db: 信噪比（dB），audio1相对于audio2的能量比
    
    Returns:
        mixture: [T] - 混合音频
        audio1: [T] - 原始audio1
        audio2_scaled: [T] - 缩放后的audio2
    """
    # 确保长度相同
    min_len = min(len(audio1), len(audio2))
    audio1 = audio1[:min_len]
    audio2 = audio2[:min_len]
    
    # 计算能量
    energy1 = torch.sum(audio1 ** 2)
    energy2 = torch.sum(audio2 ** 2)
    
    # 根据SNR计算audio2的目标能量
    # SNR_dB = 10 * log10(E1 / E2)
    # => E2_target = E1 / 10^(SNR_dB / 10)
    energy2_target = energy1 / (10 ** (snr_db / 10))
    
    # 缩放audio2到目标能量
    scale = torch.sqrt(energy2_target / (energy2 + 1e-8))
    audio2_scaled = audio2 * scale
    
    # 混合（简单相加）
    mixture = audio1 + audio2_scaled
    
    # 计算缩放因子以避免削波，同时缩放所有信号以保持SNR
    max_val = torch.max(torch.abs(mixture))
    if max_val > 0.99:
        scale_factor = 0.99 / max_val
        mixture = mixture * scale_factor
        audio1 = audio1 * scale_factor
        audio2_scaled = audio2_scaled * scale_factor
    
    return mixture, audio1, audio2_scaled


def normalize_mixture(mixture, sources, target_level=-25.0):
    """
    归一化混合信号和源信号（保持它们之间的相对关系）
    
    关键：同时缩放mixture和sources，保持SNR不变
    
    Args:
        mixture: [T] - 混合信号
        sources: [C, T] 或 list of [T] - 源信号列表
        target_level: 目标dB级别（相对于满量程）
    
    Returns:
        mixture_norm: [T] - 归一化的混合信号
        sources_norm: [C, T] - 归一化的源信号
    """
    # 转换sources为tensor
    if isinstance(sources, list):
        sources = torch.stack(sources)
    
    # 计算mixture的RMS
    rms = torch.sqrt(torch.mean(mixture ** 2))
    
    # 计算目标RMS
    target_rms = 10 ** (target_level / 20)
    
    # 计算缩放因子
    scale = target_rms / (rms + 1e-8)
    
    # 同时缩放mixture和sources（保持相对关系）
    mixture_norm = mixture * scale
    sources_norm = sources * scale
    
    # 防止削波（同时检查mixture和sources）
    max_val = max(
        torch.max(torch.abs(mixture_norm)).item(),
        torch.max(torch.abs(sources_norm)).item()
    )
    
    if max_val > 1.0:
        clip_scale = 0.99 / max_val
        mixture_norm = mixture_norm * clip_scale
        sources_norm = sources_norm * clip_scale
    
    return mixture_norm, sources_norm


def pad_or_trim(audio, target_length):
    """
    填充或裁剪音频到目标长度
    
    Args:
        audio: [T] - 音频信号
        target_length: 目标长度
    
    Returns:
        audio: [target_length] - 处理后的音频
    """
    current_length = len(audio)
    
    if current_length > target_length:
        # 裁剪（从中间开始）
        start = (current_length - target_length) // 2
        audio = audio[start:start + target_length]
    elif current_length < target_length:
        # 填充
        padding = target_length - current_length
        audio = torch.nn.functional.pad(audio, (0, padding))
    
    return audio


if __name__ == "__main__":
    print("Testing audio utils...")
    
    # 创建测试音频
    sample_rate = 16000
    duration = 2  # 秒
    audio1 = torch.randn(sample_rate * duration)
    audio2 = torch.randn(sample_rate * duration)
    
    print(f"Audio1 shape: {audio1.shape}")
    print(f"Audio2 shape: {audio2.shape}")
    
    # 测试归一化
    audio_normalized = normalize_audio(audio1)
    print(f"\nNormalized audio:")
    print(f"  Max: {audio_normalized.max().item():.4f}")
    print(f"  Min: {audio_normalized.min().item():.4f}")
    print(f"  RMS: {torch.sqrt(torch.mean(audio_normalized**2)).item():.4f}")
    
    # 测试混合
    snr_db = 0
    mixture = mix_audio(audio1, audio2, snr_db)
    print(f"\nMixed audio (SNR={snr_db}dB):")
    print(f"  Shape: {mixture.shape}")
    print(f"  Max: {mixture.max().item():.4f}")
    
    # 测试填充/裁剪
    target_length = sample_rate * 3
    audio_padded = pad_or_trim(audio1, target_length)
    print(f"\nPadded audio:")
    print(f"  Original length: {len(audio1)}")
    print(f"  Target length: {target_length}")
    print(f"  Result length: {len(audio_padded)}")
    
    print("\nAudio utils test passed!")
