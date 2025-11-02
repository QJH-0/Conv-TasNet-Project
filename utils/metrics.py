"""
语音分离评估指标计算 - SI-SDR

本模块实现了语音分离任务中的核心评价指标：

SI-SDR (Scale-Invariant Signal-to-Distortion Ratio) - 尺度不变信号失真比
   - 核心思想: 消除信号幅度缩放的影响，专注于波形结构相似性
   - 适用场景: 端到端语音分离，对幅度不敏感的场景
   - 计算特点: 零均值化 + 尺度不变投影
   - 计算步骤: 先零均值化 → 投影 → 计算 SDR

评判标准:
- SI-SDR: 越高越好，通常 > 10 dB 表示良好分离

使用建议:
- SI-SDR 是现代语音分离研究中最广泛使用的指标
- Conv-TasNet 等现代方法主要使用 SI-SDR 作为评估标准
"""

import torch
import numpy as np
import itertools


def calculate_si_sdr(estimation, target, eps=1e-8):
    """
    计算 SI-SDR (Scale-Invariant Signal-to-Distortion Ratio) - 尺度不变信号失真比
    
    **核心思想**: 消除信号幅度缩放的影响，专注于信号波形结构的相似性
    
    **计算过程**:
    1. 对分离信号和参考信号进行零均值化处理
    2. 将分离信号投影到参考信号上，得到最优的尺度因子
    3. 用这个尺度因子调整分离信号的幅度
    4. 计算调整后的信号与参考信号之间的信号失真比
    
    公式: SI-SDR = 10 * log10(||s_target||² / ||e_noise||²)
    其中: s_target = (<estimation, target> / ||target||²) * target
         e_noise = estimation - s_target
    
    **关键特点**: 对信号的绝对幅度不敏感，只关注信号波形的相对形状
    
    Args:
        estimation: [T] - 预测信号（分离后的信号）
        target: [T] - 真实信号（纯净的目标语音）
        eps: 数值稳定性常数
    
    Returns:
        si_sdr: SI-SDR 值（dB），越大越好
    """
    # 转换为 torch tensor
    if isinstance(estimation, np.ndarray):
        estimation = torch.from_numpy(estimation).float()
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).float()
    
    # 步骤1: 零均值化（去除直流分量）
    # 这是尺度不变性的关键步骤，确保只关注波形形状
    estimation = estimation - torch.mean(estimation)
    target = target - torch.mean(target)
    
    # 步骤2: 将分离信号投影到参考信号上，得到最优尺度因子
    # s_target = (<estimation, target> / ||target||²) * target
    # 这里的尺度因子 α = <estimation, target> / ||target||²
    inner_product = torch.sum(estimation * target)
    target_norm_square = torch.sum(target ** 2) + eps
    scale_factor = inner_product / target_norm_square
    
    # 步骤3: 用尺度因子调整后得到目标信号分量
    s_target = scale_factor * target
    
    # 步骤4: 计算噪声/失真分量（分离信号与目标分量的差异）
    e_noise = estimation - s_target
    
    # 步骤5: 计算信号失真比（目标信号能量 vs 噪声能量）
    target_power = torch.sum(s_target ** 2) + eps
    noise_power = torch.sum(e_noise ** 2) + eps
    si_sdr = 10 * torch.log10(target_power / noise_power)
    
    return si_sdr.item()


def calculate_sdr(estimation, target, eps=1e-8):
    """
    计算 SDR (Signal-to-Distortion Ratio) - 信号失真比
    
    Args:
        estimation: [T] - 预测信号
        target: [T] - 真实信号
        eps: 数值稳定性常数
    
    Returns:
        sdr: SDR 值（dB），越大越好
    """
    # 转换为 torch tensor
    if isinstance(estimation, np.ndarray):
        estimation = torch.from_numpy(estimation).float()
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).float()
    
    # 投影（不进行零均值化）
    inner_product = torch.sum(estimation * target)
    target_norm_square = torch.sum(target ** 2) + eps
    scale_factor = inner_product / target_norm_square
    s_target = scale_factor * target
    
    # 噪声
    e_noise = estimation - s_target
    
    # SDR
    target_power = torch.sum(s_target ** 2) + eps
    noise_power = torch.sum(e_noise ** 2) + eps
    sdr = 10 * torch.log10(target_power / noise_power)
    
    return sdr.item()


def calculate_si_sdri(estimation, target, mixture, eps=1e-8):
    """
    计算 SI-SDRi (SI-SDR improvement) - SI-SDR 改善量
    
    SI-SDRi = SI-SDR(分离信号, 目标信号) - SI-SDR(混合信号, 目标信号)
    
    Args:
        estimation: [T] - 预测信号（分离后的信号）
        target: [T] - 真实信号（纯净的目标语音）
        mixture: [T] - 混合信号（输入信号）
        eps: 数值稳定性常数
    
    Returns:
        si_sdri: SI-SDRi 值（dB），越大越好
    """
    # 计算分离后的 SI-SDR
    si_sdr_separated = calculate_si_sdr(estimation, target, eps)
    
    # 计算混合信号的 SI-SDR（基准）
    si_sdr_mixture = calculate_si_sdr(mixture, target, eps)
    
    # 计算改善量
    si_sdri = si_sdr_separated - si_sdr_mixture
    
    return si_sdri





def calculate_pit_si_sdr(estimations, targets):
    """
    使用 PIT 方式计算 SI-SDR 评估指标
    
    Args:
        estimations: [C, T] - C 个分离的语音
        targets: [C, T] - C 个真实语音
    
    Returns:
        best_si_sdr: 最优排列下的平均 SI-SDR
        best_perm: 最优排列
    """
    num_speakers = estimations.shape[0]
    perms = list(itertools.permutations(range(num_speakers)))
    
    perm_metrics = []
    for perm in perms:
        perm_metric = 0
        for est_idx, tgt_idx in enumerate(perm):
            metric = calculate_si_sdr(
                estimations[est_idx], 
                targets[tgt_idx]
            )
            perm_metric += metric
        perm_metrics.append(perm_metric / num_speakers)
    
    # SI-SDR 越大越好，选择最大值
    best_metric = max(perm_metrics)
    best_perm_idx = perm_metrics.index(best_metric)
    
    return best_metric, perms[best_perm_idx]


def evaluate_separation(model, dataloader, device='cuda', metrics=['si_sdr']):
    """
    评估模型在数据集上的表现（使用 PIT + 多指标）
    
    本函数对数据集中的每个样本进行评估，使用排列不变训练(PIT)方法
    自动找到最优的输出-目标匹配，然后计算多种评估指标。
    
    支持的指标:
       - SI-SDR: 尺度不变信号失真比，对幅度缩放不敏感
       - SDR: 信号失真比
       - SI-SDRi: SI-SDR改善量（相对于混合信号）
    
    Args:
        model: Conv-TasNet 模型（或其他语音分离模型）
        dataloader: 数据加载器，产生 (mixtures, targets) 批次
        device: 计算设备 ('cuda' 或 'cpu')
        metrics: 要计算的指标列表 ['si_sdr', 'sdr', 'si_sdri']
    
    Returns:
        metrics_dict: {
            'si_sdr': 平均 SI-SDR（dB），越大越好
            'sdr': 平均 SDR（dB），越大越好（如果请求）
            'si_sdri': 平均 SI-SDRi（dB），越大越好（如果请求）
        }
    """
    model.eval()
    
    # 初始化指标累加器
    metric_accumulators = {metric: 0.0 for metric in metrics}
    num_samples = 0
    
    with torch.no_grad():
        for mixtures, targets in dataloader:
            mixtures = mixtures.to(device)
            targets = targets.to(device)
            
            # 模型预测
            estimations = model(mixtures)  # [B, C, T]
            
            # 对每个样本计算 PIT 指标
            for i in range(mixtures.shape[0]):
                est = estimations[i].cpu()   # [C, T]
                tgt = targets[i].cpu()       # [C, T]
                mix = mixtures[i].cpu()      # [T]
                
                # 使用 PIT 找到最优排列
                num_speakers = est.shape[0]
                perms = list(itertools.permutations(range(num_speakers)))
                
                # 计算所有排列的 SI-SDR（用于找最优排列）
                perm_si_sdrs = []
                for perm in perms:
                    perm_si_sdr = 0
                    for est_idx, tgt_idx in enumerate(perm):
                        si_sdr = calculate_si_sdr(est[est_idx], tgt[tgt_idx])
                        perm_si_sdr += si_sdr
                    perm_si_sdrs.append(perm_si_sdr / num_speakers)
                
                # 找到最优排列
                best_perm_idx = perm_si_sdrs.index(max(perm_si_sdrs))
                best_perm = perms[best_perm_idx]
                
                # 根据最优排列计算各种指标
                sample_metrics = {metric: 0.0 for metric in metrics}
                
                for est_idx, tgt_idx in enumerate(best_perm):
                    if 'si_sdr' in metrics:
                        sample_metrics['si_sdr'] += calculate_si_sdr(
                            est[est_idx], tgt[tgt_idx]
                        )
                    if 'sdr' in metrics:
                        sample_metrics['sdr'] += calculate_sdr(
                            est[est_idx], tgt[tgt_idx]
                        )
                    if 'si_sdri' in metrics:
                        sample_metrics['si_sdri'] += calculate_si_sdri(
                            est[est_idx], tgt[tgt_idx], mix
                        )
                
                # 平均并累加
                for metric in metrics:
                    metric_accumulators[metric] += sample_metrics[metric] / num_speakers
                
                num_samples += 1
    
    # 计算平均值
    avg_metrics = {
        metric: metric_accumulators[metric] / num_samples 
        for metric in metrics
    }
    
    return avg_metrics


if __name__ == "__main__":
    print("=" * 80)
    print("语音分离评估指标测试 - SI-SDR")
    print("=" * 80)
    print("SI-SDR: 尺度不变信号失真比 - 衡量波形结构相似性")
    print("=" * 80)
    
    # 创建测试数据
    length = 16000
    
    # 模拟两个说话人的语音
    speaker1 = torch.randn(length)
    speaker2 = torch.randn(length)
    mixture = speaker1 + speaker2
    
    print("\n场景 1: 完美分离")
    print("-" * 80)
    estimation = speaker1.clone()
    si_sdr = calculate_si_sdr(estimation, speaker1)
    print(f"  SI-SDR: {si_sdr:.2f} dB (应该非常高)")
    
    print("\n场景 2: 部分分离（70% 目标 + 30% 干扰）")
    print("-" * 80)
    estimation_partial = 0.7 * speaker1 + 0.3 * speaker2
    si_sdr = calculate_si_sdr(estimation_partial, speaker1)
    print(f"  SI-SDR: {si_sdr:.2f} dB")
    
    print("\n场景 3: 无分离（直接使用混合信号）")
    print("-" * 80)
    estimation_none = mixture.clone()
    si_sdr = calculate_si_sdr(estimation_none, speaker1)
    print(f"  SI-SDR: {si_sdr:.2f} dB (应该较低)")
    
    print("\n场景 4: 错误分离（输出错误的说话人）")
    print("-" * 80)
    estimation_wrong = speaker2.clone()
    si_sdr = calculate_si_sdr(estimation_wrong, speaker1)
    print(f"  SI-SDR: {si_sdr:.2f} dB (应该很低)")
    
    print("\n场景 5: PIT 多说话人指标")
    print("-" * 80)
    num_speakers = 2
    estimations = torch.stack([speaker1 * 0.9 + speaker2 * 0.1, 
                               speaker2 * 0.9 + speaker1 * 0.1])
    targets = torch.stack([speaker1, speaker2])
    
    si_sdr_pit, best_perm = calculate_pit_si_sdr(estimations, targets)
    
    print(f"  PIT SI-SDR: {si_sdr_pit:.2f} dB, 最优排列: {best_perm}")
    
    print("\n" + "=" * 80)
    print("✓ SI-SDR 指标测试完成！")
    print("=" * 80)
    print("\n指标解读:")
    print("  SI-SDR: 尺度不变信号失真比，专注于波形形状相似性")
    print("     - 零均值化 + 尺度不变投影")
    print("     - 对幅度缩放不敏感")
    print()
    print("  评判标准:")
    print("     - SI-SDR > 10 dB: 良好分离")
    print("     - SI-SDR > 15 dB: 优秀分离")
    print()
    print("  应用建议:")
    print("     - SI-SDR 是现代语音分离研究中最广泛使用的指标")
    print("     - Conv-TasNet 等现代方法主要使用 SI-SDR")
    print()
