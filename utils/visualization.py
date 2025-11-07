"""
可视化工具 - 支持完整指标（SI-SDR, SDR, SIR, SAR, STOI）
"""

import matplotlib.pyplot as plt
import numpy as np
import os


def plot_loss_curves(train_losses, val_losses, save_path):
    """
    绘制训练和验证损失曲线
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (Negative SI-SNR)', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Loss curves saved to {save_path}")


def plot_metrics(metrics_history, save_path, plot_metrics=None):
    """
    绘制评估指标曲线（支持所有指标：SI-SDR, SDR, SIR, SAR, STOI, SI-SDRi）
    
    Args:
        metrics_history: {
            'si_sdr': [...],   # SI-SDR指标
            'sdr': [...],      # SDR指标
            'sir': [...],      # SIR指标（串扰抑制）
            'sar': [...],      # SAR指标（算法失真）
            'stoi': [...],     # STOI指标（可懂度，0-1）
            'si_sdri': [...]   # SI-SDRi指标
        }
        save_path: 保存路径
        plot_metrics: 要绘制的指标列表（None表示全部）
    """
    # 检查有哪些指标
    available_metrics = []
    for metric in ['si_sdr', 'sdr', 'sir', 'sar', 'stoi', 'si_sdri']:
        if metric in metrics_history and len(metrics_history[metric]) > 0:
            # 过滤掉全为0的指标（未计算）
            if any(v != 0 for v in metrics_history[metric]):
                available_metrics.append(metric)
    
    # 如果指定了要绘制的指标，过滤
    if plot_metrics:
        available_metrics = [m for m in available_metrics if m in plot_metrics]
    
    if not available_metrics:
        print("Warning: No metrics to plot")
        return
    
    # 指标信息
    metric_info = {
        'si_sdr': {
            'label': 'SI-SDR (dB)', 
            'color': 'b', 
            'title': 'SI-SDR over Epochs',
            'ylabel': 'dB'
        },
        'sdr': {
            'label': 'SDR (dB)', 
            'color': 'g', 
            'title': 'SDR over Epochs',
            'ylabel': 'dB'
        },
        'sir': {
            'label': 'SIR (dB)', 
            'color': 'c', 
            'title': 'SIR (Source-to-Interference) over Epochs',
            'ylabel': 'dB'
        },
        'sar': {
            'label': 'SAR (dB)', 
            'color': 'm', 
            'title': 'SAR (Sources-to-Artifacts) over Epochs',
            'ylabel': 'dB'
        },
        'stoi': {
            'label': 'STOI', 
            'color': 'orange', 
            'title': 'STOI (Intelligibility) over Epochs',
            'ylabel': 'Score (0-1)'
        },
        'si_sdri': {
            'label': 'SI-SDRi (dB)', 
            'color': 'r', 
            'title': 'SI-SDRi (Improvement) over Epochs',
            'ylabel': 'dB'
        }
    }
    
    # 确定子图布局
    num_metrics = len(available_metrics)
    
    # 智能布局：少于3个用垂直，多于3个用网格
    if num_metrics <= 2:
        fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 5 * num_metrics))
        axes = [axes] if num_metrics == 1 else axes
    else:
        # 2列网格布局
        rows = (num_metrics + 1) // 2
        fig, axes = plt.subplots(rows, 2, figsize=(15, 5 * rows))
        axes = axes.flatten()
        # 如果是奇数个指标，隐藏最后一个空子图
        if num_metrics % 2 == 1:
            axes[-1].set_visible(False)
    
    # 确定epochs
    epochs = range(1, len(metrics_history[available_metrics[0]]) + 1)
    
    # 绘制每个指标
    for idx, metric_name in enumerate(available_metrics):
        ax = axes[idx]
        info = metric_info[metric_name]
        
        ax.plot(epochs, metrics_history[metric_name], 
                color=info['color'], linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(info['ylabel'], fontsize=12)
        ax.set_title(info['title'], fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 为 STOI 设置 y 轴范围 [0, 1]
        if metric_name == 'stoi':
            ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Metrics plot saved to {save_path}")


def plot_distillation_metrics(train_losses, val_losses, save_path, plot_metrics=None):
    """
    绘制蒸馏训练的指标曲线（支持所有指标）
    
    Args:
        train_losses: 训练损失历史 (list of dicts)
        val_losses: 验证损失历史 (list of dicts)
        save_path: 保存路径
        plot_metrics: 要绘制的指标列表
    """
    if not train_losses or not val_losses:
        print("Warning: No data to plot")
        return
    
    epochs = range(1, len(train_losses) + 1)
    
    # 检测可用指标
    available_metrics = []
    for metric in ['si_sdr', 'sdr', 'sir', 'sar', 'stoi', 'si_sdri']:
        if metric in val_losses[0] and any(d.get(metric, 0) != 0 for d in val_losses):
            available_metrics.append(metric)
    
    # 过滤指标
    if plot_metrics:
        available_metrics = [m for m in available_metrics if m in plot_metrics]
    
    # 总是包含总损失
    num_plots = 1 + len(available_metrics)
    
    # 智能布局
    if num_plots <= 4:
        rows, cols = 2, 2
    else:
        rows = (num_plots + 2) // 3
        cols = 3
    
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows))
    axes = axes.flatten() if num_plots > 1 else [axes]
    
    # 隐藏多余的子图
    for i in range(num_plots, len(axes)):
        axes[i].set_visible(False)
    
    # 1. 总损失
    axes[0].plot(epochs, [d['total_loss'] for d in train_losses], 
                'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, [d['total_loss'] for d in val_losses], 
                'r-', label='Val', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Total Loss', fontsize=12)
    axes[0].set_title('Total Distillation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 指标配置
    metric_configs = {
        'si_sdr': ('SI-SDR (dB)', 'g'),
        'sdr': ('SDR (dB)', 'c'),
        'sir': ('SIR (dB)', 'b'),
        'sar': ('SAR (dB)', 'm'),
        'stoi': ('STOI (0-1)', 'orange'),
        'si_sdri': ('SI-SDRi (dB)', 'r')
    }
    
    # 绘制其他指标
    for idx, metric in enumerate(available_metrics, start=1):
        label, color = metric_configs[metric]
        axes[idx].plot(epochs, [d.get(metric, 0) for d in val_losses], 
                      color=color, linewidth=2, marker='o', markersize=4)
        axes[idx].set_xlabel('Epoch', fontsize=12)
        axes[idx].set_ylabel(label, fontsize=12)
        axes[idx].set_title(f'{metric.upper()} Metric', fontsize=14, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
        
        # STOI 特殊处理
        if metric == 'stoi':
            axes[idx].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Distillation metrics plot saved to {save_path}")


def plot_waveforms(mixture, separated, targets, sample_rate=16000, save_path=None):
    """
    绘制波形对比图
    
    Args:
        mixture: [T] - 混合音频
        separated: [C, T] - 分离后的音频
        targets: [C, T] - 真实音频
        sample_rate: 采样率
        save_path: 保存路径
    """
    num_speakers = separated.shape[0]
    time = np.arange(len(mixture)) / sample_rate
    
    fig, axes = plt.subplots(num_speakers + 1, 1, figsize=(15, 3 * (num_speakers + 1)))
    
    # 混合音频
    axes[0].plot(time, mixture, 'k-', linewidth=0.5)
    axes[0].set_title('Mixture', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, time[-1]])
    
    # 每个说话人
    for i in range(num_speakers):
        ax = axes[i + 1]
        ax.plot(time, targets[i], 'b-', alpha=0.5, linewidth=0.5, label='Target')
        ax.plot(time, separated[i], 'r-', linewidth=0.5, label='Separated')
        ax.set_title(f'Speaker {i+1}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Amplitude')
        ax.set_xlabel('Time (s)' if i == num_speakers - 1 else '')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, time[-1]])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Waveform plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    print("Testing visualization with new metrics...")
    
    # 创建测试目录
    os.makedirs('utils/test_plots', exist_ok=True)
    
    # 测试损失曲线
    train_losses = [-5 + i * 0.1 + np.random.randn() * 0.5 for i in range(50)]
    val_losses = [-4 + i * 0.08 + np.random.randn() * 0.5 for i in range(50)]
    plot_loss_curves(train_losses, val_losses, 'utils/test_plots/loss_curves.png')
    
    # 测试完整指标曲线（包括新指标）
    metrics_history = {
        'si_sdr': [8 + i * 0.18 + np.random.randn() * 0.4 for i in range(50)],
        'sdr': [8.5 + i * 0.17 + np.random.randn() * 0.4 for i in range(50)],
        'sir': [15 + i * 0.2 + np.random.randn() * 0.5 for i in range(50)],
        'sar': [12 + i * 0.15 + np.random.randn() * 0.4 for i in range(50)],
        'stoi': [0.7 + i * 0.005 + np.random.randn() * 0.02 for i in range(50)],
        'si_sdri': [6 + i * 0.15 + np.random.randn() * 0.3 for i in range(50)]
    }
    # 确保 STOI 在 [0, 1] 范围内
    metrics_history['stoi'] = np.clip(metrics_history['stoi'], 0, 1).tolist()
    
    plot_metrics(metrics_history, 'utils/test_plots/all_metrics.png')
    
    # 测试部分指标
    plot_metrics(metrics_history, 'utils/test_plots/selected_metrics.png', 
                plot_metrics=['si_sdr', 'sdr', 'sir', 'sar'])
    
    print("\nVisualization test completed. Check utils/test_plots/ directory.")
