"""
语音分离评估指标计算 - 基于 Asteroid 库

本模块使用 asteroid 库实现语音分离任务的评估指标：

SI-SDR (Scale-Invariant Signal-to-Distortion Ratio) - 尺度不变信号失真比
   - 核心思想: 消除信号幅度缩放的影响，专注于波形结构相似性
   - 适用场景: 端到端语音分离，对幅度不敏感的场景
   
SDR (Signal-to-Distortion Ratio) - 信号失真比
   - 标准的 BSS Eval 指标
   
SIR (Source-to-Interference Ratio) - 信号干扰比
   - 衡量其他声源的干扰程度
   
SAR (Sources-to-Artifacts Ratio) - 信号伪影比
   - 衡量分离算法引入的失真

评判标准:
- SI-SDR: 越高越好，通常 > 10 dB 表示良好分离
- SDR: 越高越好
- SIR: 越高越好，衡量串扰抑制能力
- SAR: 越高越好，衡量算法失真

优势:
- 自动 PIT (排列不变训练) 匹配
- 经过充分测试和优化
- 支持批量处理
- 完整的 BSS Eval 误差分解
"""

import torch
import numpy as np
from typing import Dict, List, Union, Optional

# Asteroid 库导入
try:
    from asteroid.metrics import get_metrics as asteroid_get_metrics
    ASTEROID_AVAILABLE = True
except ImportError:
    ASTEROID_AVAILABLE = False
    print("警告: asteroid 库未安装，将使用简化版指标计算")
    print("请运行: pip install asteroid")

# STOI 库导入（纯 Python，无需编译）
try:
    from pystoi import stoi as compute_stoi
    STOI_AVAILABLE = True
except ImportError:
    STOI_AVAILABLE = False
    # 不打印警告，因为 STOI 是可选的


# 注意：旧的手动实现已移除
# 现在统一使用 Asteroid 库进行指标计算
# 如需手动实现，请参考 git 历史或 Asteroid 源码


def calculate_stoi(estimation, target, sample_rate=16000):
    """
    计算 STOI (Short-Time Objective Intelligibility) - 短时客观可懂度
    
    STOI 是一个纯 Python 实现的语音质量指标，无需编译。
    它衡量语音的可懂度（intelligibility），范围 0-1，越高越好。
    
    Args:
        estimation: [T] - 预测信号
        target: [T] - 真实信号
        sample_rate: 采样率
    
    Returns:
        stoi: STOI 值（0-1），越大越好，通常 > 0.9 表示良好
    """
    if not STOI_AVAILABLE:
        return 0.0
    
    # 转换为 numpy
    if isinstance(estimation, torch.Tensor):
        estimation = estimation.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    try:
        # pystoi 需要 1D 数组
        stoi_score = compute_stoi(target, estimation, sample_rate, extended=False)
        return float(stoi_score)
    except Exception as e:
        print(f"警告: STOI 计算失败 - {e}")
        return 0.0


def calculate_metrics_asteroid(estimations, targets, mixtures=None, 
                               sample_rate=16000,
                               metrics_list=['si_sdr', 'sdr', 'sir', 'sar'],
                               include_stoi=False):
    """
    使用 Asteroid 库计算语音分离评估指标（自动 PIT + 误差分解）
    
    Asteroid 的优势:
       - 自动处理 PIT（排列不变训练）匹配
       - 完整的 BSS Eval 误差分解 (SDR = SIR + SAR + ...)
       - 经过充分测试和优化
       - 支持批量处理
    
    Args:
        estimations: [B, C, T] - 分离后的语音
        targets: [B, C, T] - 真实目标语音
        mixtures: [B, T] - 混合信号（可选，用于计算SI-SDRi）
        sample_rate: 采样率
        metrics_list: 要计算的指标列表，可选 ['si_sdr', 'sdr', 'sir', 'sar']
        include_stoi: 是否计算 STOI 指标（纯 Python，无需编译）
        
    Returns:
        metrics_dict: {
            'si_sdr': 平均SI-SDR (dB),
            'sdr': 平均SDR (dB),
            'sir': 平均SIR (dB),
            'sar': 平均SAR (dB),
            'stoi': 平均STOI (0-1, 可选),
            ...
        }
    """
    if not ASTEROID_AVAILABLE:
        raise ImportError("Asteroid 库未安装。请运行: pip install asteroid")
    
    # 转换为 numpy (asteroid 需要 numpy 格式)
    if isinstance(estimations, torch.Tensor):
        estimations = estimations.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if mixtures is not None and isinstance(mixtures, torch.Tensor):
        mixtures = mixtures.cpu().numpy()
    
    batch_size = estimations.shape[0]
    
    # 对每个样本计算指标
    all_metrics = {metric: [] for metric in metrics_list}
    if include_stoi and STOI_AVAILABLE:
        all_metrics['stoi'] = []
    
    for i in range(batch_size):
        est = estimations[i]  # [C, T]
        tgt = targets[i]      # [C, T]
        
        # 准备混合信号
        if mixtures is not None:
            mix = mixtures[i]  # [T]
        else:
            # 如果没有提供混合信号，使用目标信号的和作为近似
            mix = np.sum(tgt, axis=0)
        
        # 使用 asteroid 计算指标（自动 PIT）
        # 注意: 不在 metrics_list 中包含 'stoi'，因为我们手动计算
        asteroid_metrics_list = [m for m in metrics_list if m != 'stoi']
        metrics = asteroid_get_metrics(
            mix=mix,
            clean=tgt,
            estimate=est,
            sample_rate=sample_rate,
            metrics_list=asteroid_metrics_list,
            average=True  # 自动平均多说话人
        )
        
        # 累积 asteroid 指标结果
        for metric in asteroid_metrics_list:
            if metric in metrics:
                all_metrics[metric].append(metrics[metric])
        
        # 手动计算 STOI（纯 Python，无需编译）
        if include_stoi and STOI_AVAILABLE:
            # 对每个说话人计算 STOI 后平均
            stoi_scores = []
            num_speakers = est.shape[0]
            for spk in range(num_speakers):
                stoi_score = calculate_stoi(est[spk], tgt[spk], sample_rate)
                stoi_scores.append(stoi_score)
            all_metrics['stoi'].append(np.mean(stoi_scores))
    
    # 计算批次平均
    avg_metrics = {
        metric: np.mean(values) if values else 0.0
        for metric, values in all_metrics.items()
    }
    
    # 如果提供了混合信号，添加 SI-SDRi
    if mixtures is not None and 'si_sdr' in metrics_list:
        si_sdri_list = []
        for i in range(batch_size):
            mix = mixtures[i]
            tgt = targets[i]
            
            # 计算混合信号的基准 SI-SDR
            mix_metrics = asteroid_get_metrics(
                mix=mix,
                clean=tgt,
                estimate=np.stack([mix] * tgt.shape[0]),  # 重复混合信号
                sample_rate=sample_rate,
                metrics_list=['si_sdr'],
                average=True
            )
            
            si_sdri = avg_metrics['si_sdr'] - mix_metrics['si_sdr']
            si_sdri_list.append(si_sdri)
        
        avg_metrics['si_sdri'] = np.mean(si_sdri_list)
    
    return avg_metrics


def calculate_metrics(estimations, targets, mixtures=None, 
                     metrics_list=['si_sdr', 'sdr', 'sir', 'sar'],
                     use_asteroid=True, include_stoi=False):
    """
    计算语音分离评估指标（统一接口）
    
    **强烈推荐使用 Asteroid 库**：自动 PIT + 完整误差分解
    
    Args:
        estimations: [B, C, T] - 分离后的语音
        targets: [B, C, T] - 真实目标语音
        mixtures: [B, T] - 混合信号（可选，用于计算SI-SDRi）
        metrics_list: 要计算的指标列表，默认 ['si_sdr', 'sdr', 'sir', 'sar']
        use_asteroid: 是否使用 asteroid 库（默认 True，强烈推荐）
        include_stoi: 是否计算 STOI 指标（纯 Python，无需编译）
    
    Returns:
        metrics_dict: {
            'si_sdr': 平均SI-SDR (dB),
            'sdr': 平均SDR (dB),
            'sir': 平均SIR (dB),
            'sar': 平均SAR (dB),
            'stoi': 平均STOI (0-1, 可选),
            'si_sdri': 平均SI-SDRi (dB, 如果提供mixtures)
        }
    """
    if not use_asteroid or not ASTEROID_AVAILABLE:
        raise ImportError(
            "Asteroid 库未安装或被禁用。\n"
            "请安装: pip install asteroid\n"
            "旧的手动实现已移除，请使用 Asteroid 库以获得：\n"
            "  - 自动 PIT 匹配\n"
            "  - 完整的 BSS Eval 误差分解 (SIR, SAR)\n"
            "  - 经过充分测试和优化的实现"
        )
    
    # 使用 asteroid 库（推荐）
    return calculate_metrics_asteroid(
        estimations, targets, mixtures,
        metrics_list=metrics_list,
        include_stoi=include_stoi
    )


# 旧的手动实现已移除
# 现在统一使用 Asteroid 库，提供：
# - 自动 PIT 匹配
# - 完整的 BSS Eval 误差分解（SIR, SAR）
# - 经过充分测试和优化
# - 如果需要回退，请使用 use_asteroid=False（会抛出错误提示安装 asteroid）


def evaluate_separation(model, dataloader, device='cuda', 
                        metrics=['si_sdr', 'sdr', 'sir', 'sar'],
                        use_asteroid=True,
                        include_stoi=False):
    """
    评估模型在数据集上的表现（使用 Asteroid 自动 PIT + 多指标）
    
    本函数对数据集中的每个样本进行评估，使用 asteroid 库自动处理
    排列不变训练(PIT)和误差分解。
    
    支持的指标:
       - SI-SDR: 尺度不变信号失真比
       - SDR: 信号失真比
       - SIR: 信号干扰比（衡量串扰）
       - SAR: 信号伪影比（衡量算法失真）
       - STOI: 短时客观可懂度（纯 Python，无需编译）
       - SI-SDRi: SI-SDR改善量
    
    Args:
        model: 语音分离模型
        dataloader: 数据加载器，产生 (mixtures, targets) 批次
        device: 计算设备 ('cuda' 或 'cpu')
        metrics: 要计算的指标列表
        use_asteroid: 是否使用 asteroid 库
        include_stoi: 是否计算 STOI 指标（纯 Python，无需编译）
    
    Returns:
        metrics_dict: {
            'si_sdr': 平均 SI-SDR（dB）
            'sdr': 平均 SDR（dB）
            'sir': 平均 SIR（dB）
            'sar': 平均 SAR（dB）
            'stoi': 平均 STOI（0-1，可选）
            'si_sdri': 平均 SI-SDRi（dB）
        }
    """
    model.eval()
    
    # 初始化指标累加器
    metric_accumulators = {metric: [] for metric in metrics}
    if 'si_sdr' in metrics:
        metric_accumulators['si_sdri'] = []
    if include_stoi:
        metric_accumulators['stoi'] = []
    
    with torch.no_grad():
        for batch in dataloader:
            # 数据格式转换
            mixtures = batch['mix'].to(device)
            targets = torch.stack(batch['ref'], dim=1).to(device)
            
            # 模型预测
            estimations = model(mixtures)  # [B, C, T]
            
            # 计算指标（使用 Asteroid）
            if not use_asteroid or not ASTEROID_AVAILABLE:
                raise ImportError("Asteroid 库未安装。请运行: pip install asteroid")
            
            batch_metrics = calculate_metrics_asteroid(
                estimations.cpu(),
                targets.cpu(),
                mixtures.cpu(),
                metrics_list=metrics,
                include_stoi=include_stoi
            )
            
            # 累积结果
            for metric, value in batch_metrics.items():
                if metric in metric_accumulators:
                    metric_accumulators[metric].append(value)
    
    # 计算平均值
    avg_metrics = {
        metric: np.mean(values) if values else 0.0
        for metric, values in metric_accumulators.items()
    }
    
    return avg_metrics

