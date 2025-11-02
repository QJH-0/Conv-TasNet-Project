"""
可视化工具
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


def plot_metrics(metrics_history, save_path):
    """
    绘制评估指标曲线（SI-SDR）
    
    Args:
        metrics_history: {
            'si_sdr': [...]  # SI-SDR指标
        }
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 6))
    
    # 确定epochs
    epochs = range(1, len(metrics_history['si_sdr']) + 1)
    
    # SI-SDR
    plt.plot(epochs, metrics_history['si_sdr'], 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('SI-SDR (dB)', fontsize=12)
    plt.title('SI-SDR over Epochs', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Metrics plot saved to {save_path}")


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
    print("Testing visualization...")
    
    # 创建测试目录
    os.makedirs('test_plots', exist_ok=True)
    
    # 测试损失曲线
    train_losses = [-5 + i * 0.1 + np.random.randn() * 0.5 for i in range(50)]
    val_losses = [-4 + i * 0.08 + np.random.randn() * 0.5 for i in range(50)]
    plot_loss_curves(train_losses, val_losses, 'test_plots/loss_curves.png')
    
    # 测试指标曲线（SI-SDR）
    metrics_history = {
        'si_sdr': [8 + i * 0.18 + np.random.randn() * 0.4 for i in range(50)]
    }
    plot_metrics(metrics_history, 'test_plots/metrics.png')
    
    # 测试波形图
    sample_rate = 16000
    duration = 2
    mixture = np.random.randn(sample_rate * duration) * 0.5
    separated = np.random.randn(2, sample_rate * duration) * 0.3
    targets = np.random.randn(2, sample_rate * duration) * 0.3
    plot_waveforms(mixture, separated, targets, sample_rate, 'test_plots/waveforms.png')
    
    print("\nVisualization test completed. Check test_plots/ directory.")
