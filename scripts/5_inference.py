"""
推理脚本
对新的混合音频进行语音分离
"""

import os
import sys
import argparse
import yaml
import torch

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
# 切换到项目根目录,确保所有相对路径正确
os.chdir(project_root)

from models.conv_tasnet import ConvTasNet
from utils.audio_utils import load_audio, save_audio


def load_config(config_path):
    """加载配置文件"""
    # 如果是相对路径，转换为相对于项目根目录的绝对路径
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 规范化配置中的所有路径为基于项目根目录的绝对路径
    if 'dataset' in config and 'processed_data_path' in config['dataset']:
        if not os.path.isabs(config['dataset']['processed_data_path']):
            config['dataset']['processed_data_path'] = os.path.join(
                project_root, config['dataset']['processed_data_path']
            )
    
    if 'logging' in config:
        for key in ['log_dir', 'checkpoint_dir', 'result_dir']:
            if key in config['logging'] and not os.path.isabs(config['logging'][key]):
                config['logging'][key] = os.path.join(
                    project_root, config['logging'][key]
                )
    
    return config


def load_model(checkpoint_path, config, device):
    """加载模型"""
    model = ConvTasNet.from_config(config)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only= False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


def separate_audio(model, mixture_path, output_dir, sample_rate, device):
    """
    分离音频
    
    Args:
        model: Conv-TasNet 模型
        mixture_path: 混合音频路径
        output_dir: 输出目录
        sample_rate: 采样率
        device: 设备
    """
    # 加载音频
    print(f"Loading audio: {mixture_path}")
    mixture, sr = load_audio(mixture_path, sample_rate)
    
    # 转换为 tensor
    mixture = mixture.to(device).unsqueeze(0)  # [1, T]
    
    # 推理
    print("Separating...")
    with torch.no_grad():
        separated = model(mixture)  # [1, C, T]
    
    # 保存分离后的音频
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(mixture_path))[0]
    num_speakers = separated.shape[1]
    
    output_paths = []
    for i in range(num_speakers):
        output_path = os.path.join(output_dir, f"{base_name}_speaker{i+1}.wav")
        save_audio(output_path, separated[0, i], sample_rate)
        output_paths.append(output_path)
        print(f"Saved: {output_path}")
    
    return output_paths


def find_best_checkpoint(config):
    """查找最佳模型检查点"""
    checkpoint_dir = config['logging']['checkpoint_dir']
    if not os.path.isabs(checkpoint_dir):
        checkpoint_dir = os.path.join(project_root, checkpoint_dir)
    
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        return best_model_path
    
    # 如果没有best_model.pth，查找最新的检查点
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        if checkpoints:
            checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
            return os.path.join(checkpoint_dir, checkpoints[0])
    
    return None


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Separate audio with Conv-TasNet')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input mixture audio file')
    parser.add_argument('--output', type=str, default='outputs/separated_audio',
                       help='Output directory for separated audio')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint (default: auto-find best model)')
    parser.add_argument('--config', type=str, default='config/config.yml',
                       help='Path to config file')
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return
    
    # 加载配置
    config = load_config(args.config)
    
    # 自动查找检查点
    if args.checkpoint is None:
        print("未指定checkpoint，自动查找最佳模型...")
        args.checkpoint = find_best_checkpoint(config)
        if args.checkpoint is None:
            print("错误: 未找到任何模型检查点!")
            print(f"请先训练模型或指定checkpoint路径")
            return
        print(f"找到模型: {args.checkpoint}")
    
    # 检查检查点
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return
    
    # 加载配置
    config = load_config(args.config)
    sample_rate = config['dataset']['sample_rate']
    
    # 设置设备
    device_config = config['device']
    if device_config['use_cuda'] and torch.cuda.is_available():
        device = f"cuda:{device_config['gpu_ids'][0]}"
    else:
        device = 'cpu'
    
    print("=" * 80)
    print("Conv-TasNet Audio Separation")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Sample rate: {sample_rate} Hz")
    
    # 加载模型
    print("\nLoading model...")
    model = load_model(args.checkpoint, config, device)
    print("Model loaded successfully")
    
    # 分离音频
    print("\n" + "-" * 80)
    output_paths = separate_audio(
        model=model,
        mixture_path=args.input,
        output_dir=args.output,
        sample_rate=sample_rate,
        device=device
    )
    
    print("-" * 80)
    print("\nSeparation completed successfully!")
    print(f"Output files:")
    for path in output_paths:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
