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
from models.bconv_tasnet import BConvTasNet
from utils.audio_utils import load_audio, save_audio
from utils.pretrained_model_utils import load_asteroid_model, load_speechbrain_model, print_available_models


def load_config(config_path):
    """加载配置文件 - 先从config_loadpath.yml获取实际配置文件路径"""
    # 首先加载config_loadpath.yml获取实际配置文件路径
    loadpath_config_file = os.path.join(project_root, 'config', 'config_loadpath.yml')
    
    if os.path.exists(loadpath_config_file):
        with open(loadpath_config_file, 'r', encoding='utf-8') as f:
            loadpath_config = yaml.safe_load(f)
        
        # 从config_loadpath.yml获取实际配置文件路径
        if 'loadPath' in loadpath_config and 'config' in loadpath_config['loadPath']:
            actual_config_path = loadpath_config['loadPath']['config']
            print(f"从 config_loadpath.yml 加载配置文件路径: {actual_config_path}")
        else:
            # 如果config_loadpath.yml格式不正确，使用传入的参数
            print(f"警告: config_loadpath.yml 格式不正确，使用默认配置路径")
            actual_config_path = config_path
    else:
        # 如果config_loadpath.yml不存在，使用传入的参数
        print(f"警告: config_loadpath.yml 不存在，使用默认配置路径")
        actual_config_path = config_path
    
    # 如果是相对路径，转换为相对于项目根目录的绝对路径
    if not os.path.isabs(actual_config_path):
        actual_config_path = os.path.join(project_root, actual_config_path)
    
    print(f"加载配置文件: {actual_config_path}")
    
    with open(actual_config_path, 'r', encoding='utf-8') as f:
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


def load_model(checkpoint_path, config, device, model_type='conv_tasnet'):
    """加载模型（兼容单GPU和多GPU训练的checkpoint）"""
    from collections import OrderedDict
    
    # 根据模型类型创建模型
    if model_type == 'bconv_tasnet':
        model = BConvTasNet.from_config(config)
    else:
        model = ConvTasNet.from_config(config)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    
    # 处理DataParallel的key不匹配问题
    # 如果checkpoint是多GPU训练的（有module.前缀），但当前模型没有DataParallel包装
    if list(state_dict.keys())[0].startswith('module.'):
        # 移除'module.'前缀
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k  # 移除'module.'
            new_state_dict[name] = v
        state_dict = new_state_dict
        print("✓ Converted multi-GPU checkpoint to single-GPU format")
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    return model


def load_pretrained_model(source, model_name, device, cache_dir=None):
    """
    加载预训练模型（Asteroid 或 SpeechBrain）
    
    Args:
        source: 'asteroid' 或 'speechbrain'
        model_name: 模型名称
        device: 设备
        cache_dir: 缓存目录
        
    Returns:
        model: 加载的模型
    """
    if source == 'asteroid':
        model, info = load_asteroid_model(model_name, device, cache_dir)
        print(f"\n模型信息:")
        print(f"  来源: Asteroid")
        print(f"  模型名: {model_name}")
        print(f"  参数量: {info['total_parameters_M']}M")
        return model
    elif source == 'speechbrain':
        separator, info = load_speechbrain_model(model_name, device, cache_dir)
        print(f"\n模型信息:")
        print(f"  来源: SpeechBrain")
        print(f"  模型名: {model_name}")
        return separator
    else:
        raise ValueError(f"不支持的模型来源: {source}。请使用 'asteroid' 或 'speechbrain'")


def separate_audio(model, mixture_path, output_dir, sample_rate, device, is_speechbrain=False):
    """
    分离音频
    
    Args:
        model: Conv-TasNet 模型或 SpeechBrain separator
        mixture_path: 混合音频路径
        output_dir: 输出目录
        sample_rate: 采样率
        device: 设备
        is_speechbrain: 是否是 SpeechBrain 模型
    """
    # 加载音频
    print(f"Loading audio: {mixture_path}")
    
    # SpeechBrain 使用自己的分离方法
    if is_speechbrain:
        import torchaudio
        # SpeechBrain separator 直接处理文件
        est_sources = model.separate_file(path=mixture_path)
        
        # 保存分离后的音频
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(mixture_path))[0]
        
        output_paths = []
        for i in range(est_sources.shape[-1]):
            output_path = os.path.join(output_dir, f"{base_name}_speaker{i+1}.wav")
            torchaudio.save(output_path, est_sources[:, :, i].cpu(), sample_rate)
            output_paths.append(output_path)
            print(f"Saved: {output_path}")
        
        return output_paths
    
    # Asteroid/本地模型使用标准方法
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
    parser = argparse.ArgumentParser(description='Separate audio with Conv-TasNet / Pretrained Models')
    parser.add_argument('--input', type=str, required=False,
                       help='Path to input mixture audio file')
    parser.add_argument('--output', type=str, default='outputs/separated_audio',
                       help='Output directory for separated audio')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint (default: auto-find best model)')
    parser.add_argument('--config', type=str, default='config/config.yml',
                       help='Path to config file')
    parser.add_argument('--model-type', type=str, default='conv_tasnet',
                       choices=['conv_tasnet', 'bconv_tasnet'],
                       help='Model type for local checkpoint')
    
    # 预训练模型参数
    parser.add_argument('--pretrained', type=str, default=None,
                       choices=['asteroid', 'speechbrain'],
                       help='Use pretrained model from Asteroid or SpeechBrain')
    parser.add_argument('--pretrained-name', type=str, default=None,
                       help='Pretrained model name (e.g., mpariente/ConvTasNet_WHAM!_sepclean)')
    parser.add_argument('--cache-dir', type=str, default=None,
                       help='Cache directory for pretrained models')
    parser.add_argument('--list-models', action='store_true',
                       help='List available pretrained models and exit')
    
    args = parser.parse_args()
    
    # 如果只是列出模型，打印后退出
    if args.list_models:
        print_available_models()
        return
    
    # 检查输入文件
    if not args.input:
        print("Error: --input is required")
        return
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
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
    print("语音分离推理")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    
    # 判断使用预训练模型还是本地checkpoint
    is_speechbrain = False
    if args.pretrained:
        # 使用预训练模型
        if not args.pretrained_name:
            print(f"Error: --pretrained-name is required when using --pretrained")
            print("\n使用 --list-models 查看可用的预训练模型")
            return
        
        print(f"模型来源: {args.pretrained} 预训练模型")
        print(f"模型名称: {args.pretrained_name}")
        
        # 设置缓存目录
        if args.cache_dir:
            cache_dir = args.cache_dir
        else:
            cache_dir = f"pretrained_models/{args.pretrained}"
        print(f"缓存目录: {cache_dir}")
        
        # 加载预训练模型
        print("\n正在加载预训练模型...")
        model = load_pretrained_model(args.pretrained, args.pretrained_name, device, cache_dir)
        is_speechbrain = (args.pretrained == 'speechbrain')
        print("✓ 预训练模型加载成功")
        
    else:
        # 使用本地checkpoint
        # 自动查找检查点
        if args.checkpoint is None:
            print("未指定checkpoint，自动查找最佳模型...")
            args.checkpoint = find_best_checkpoint(config)
            if args.checkpoint is None:
                print("错误: 未找到任何模型检查点!")
                print(f"请先训练模型或指定checkpoint路径，或使用 --pretrained 加载预训练模型")
                return
            print(f"找到模型: {args.checkpoint}")
        
        # 检查检查点
        if not os.path.exists(args.checkpoint):
            print(f"Error: Checkpoint not found: {args.checkpoint}")
            return
        
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Model type: {args.model_type}")
        
        # 加载模型
        print("\nLoading model...")
        model = load_model(args.checkpoint, config, device, args.model_type)
        print("Model loaded successfully")
    
    print(f"Sample rate: {sample_rate} Hz")
    
    # 分离音频
    print("\n" + "-" * 80)
    output_paths = separate_audio(
        model=model,
        mixture_path=args.input,
        output_dir=args.output,
        sample_rate=sample_rate,
        device=device,
        is_speechbrain=is_speechbrain
    )
    
    print("-" * 80)
    print("\nSeparation completed successfully!")
    print(f"Output files:")
    for path in output_paths:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
