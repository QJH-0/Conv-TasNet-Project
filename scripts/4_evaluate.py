"""
评估脚本
评估训练好的模型
"""

import os
import sys
import argparse
import yaml
import torch
import json

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
# 切换到项目根目录,确保所有相对路径正确
os.chdir(project_root)

from models.conv_tasnet import ConvTasNet
from models.bconv_tasnet import BConvTasNet
from dataset.dataloader import create_dataloader
from utils.metrics import evaluate_separation
from utils.logger import setup_logger
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
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
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
        is_speechbrain: 是否是 SpeechBrain 模型
    """
    if source == 'asteroid':
        model, info = load_asteroid_model(model_name, device, cache_dir)
        print(f"\n模型信息:")
        print(f"  来源: Asteroid")
        print(f"  模型名: {model_name}")
        print(f"  参数量: {info['total_parameters_M']}M")
        return model, False
    elif source == 'speechbrain':
        separator, info = load_speechbrain_model(model_name, device, cache_dir)
        print(f"\n模型信息:")
        print(f"  来源: SpeechBrain")
        print(f"  模型名: {model_name}")
        return separator, True
    else:
        raise ValueError(f"不支持的模型来源: {source}。请使用 'asteroid' 或 'speechbrain'")


def find_best_checkpoint(config):
    """查找最佳模型检查点"""
    checkpoint_dir = config['logging']['checkpoint_dir_load']
    if not os.path.isabs(checkpoint_dir):
        checkpoint_dir = os.path.join(project_root, checkpoint_dir)
    
    # best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    best_model_path = checkpoint_dir
    if os.path.exists(best_model_path):
        return best_model_path
    
    # 如果没有best_model.pth，查找最新的检查点
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if checkpoints:
        # 按修改时间排序，返回最新的
        checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
        return os.path.join(checkpoint_dir, checkpoints[0])
    
    return None


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Evaluate Conv-TasNet / Pretrained Models')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint (default: auto-find best model)')
    parser.add_argument('--config', type=str, default='config/config.yml',
                       help='Path to config file')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Path to test data directory (override config)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results (override config)')
    parser.add_argument('--gpu-id', type=int, default=None,
                       help='GPU ID to use (override config)')
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
    
    # 加载配置
    config = load_config(args.config)
    
    # 应用命令行参数覆盖
    if args.output_dir:
        config['logging']['result_dir'] = os.path.join(args.output_dir, 'results')
        config['logging']['checkpoint_dir'] = os.path.join(args.output_dir, 'checkpoints')
    if args.gpu_id is not None:
        config['device']['gpu_ids'] = [args.gpu_id]
    
    # 设置设备
    device_config = config['device']
    if device_config['use_cuda'] and torch.cuda.is_available():
        device = f"cuda:{device_config['gpu_ids'][0]}"
    else:
        device = 'cpu'
    
    print("=" * 80)
    print("语音分离模型评估")
    print("=" * 80)
    print(f"Device: {device}")
    
    # 设置日志
    log_dir = config['logging']['log_dir']
    logger = setup_logger('eval', log_dir, 'evaluation.log')
    
    # 判断使用预训练模型还是本地checkpoint
    is_speechbrain = False
    if args.pretrained:
        # 使用预训练模型
        if not args.pretrained_name:
            print(f"Error: --pretrained-name is required when using --pretrained")
            print("\n使用 --list-models 查看可用的预训练模型")
            sys.exit(1)
        
        logger.info(f"模型来源: {args.pretrained} 预训练模型")
        logger.info(f"模型名称: {args.pretrained_name}")
        print(f"模型来源: {args.pretrained} 预训练模型")
        print(f"模型名称: {args.pretrained_name}")
        
        # 设置缓存目录
        if args.cache_dir:
            cache_dir = args.cache_dir
        else:
            cache_dir = f"pretrained_models/{args.pretrained}"
        logger.info(f"缓存目录: {cache_dir}")
        print(f"缓存目录: {cache_dir}")
        
        # 加载预训练模型
        logger.info("正在加载预训练模型...")
        model, is_speechbrain = load_pretrained_model(args.pretrained, args.pretrained_name, device, cache_dir)
        logger.info("✓ 预训练模型加载成功")
        
    else:
        # 使用本地checkpoint
        # 自动查找检查点
        if args.checkpoint is None:
            print("未指定checkpoint，自动查找最佳模型...")
            args.checkpoint = find_best_checkpoint(config)
            if args.checkpoint is None:
                print("错误: 未找到任何模型检查点!")
                print(f"请检查目录: {config['logging']['checkpoint_dir']}")
                print("或使用 --pretrained 加载预训练模型")
                sys.exit(1)
            print(f"找到模型: {args.checkpoint}")
        
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Model type: {args.model_type}")
        logger.info(f"Checkpoint: {args.checkpoint}")
        logger.info(f"Model type: {args.model_type}")
        
        # 加载模型
        logger.info("Loading model...")
        model = load_model(args.checkpoint, config, device, args.model_type)
        logger.info("Model loaded successfully")
    
    # 创建数据加载器（wsj0-2mix格式）
    if args.data_dir:
        test_dir = args.data_dir
    else:
        # wsj0-2mix 标准路径
        if 'data_path' in config['dataset']:
            # 使用预处理好的数据集（如 Libri2Mix）
            test_dir = os.path.join(config['dataset']['data_path'], 'test')
        else:
            # 使用自己生成的数据
            sr_folder = f"wav{config['dataset']['sample_rate']//1000}k"
            test_dir = os.path.join(
                config['dataset']['processed_data_path'],
                sr_folder, 'test'
            )
    
    logger.info(f"Test data directory: {test_dir}")
    
    # 新API：简化的参数
    use_all_chunks = config['dataset'].get('use_all_chunks', True)  # 默认True保持向后兼容
    
    test_loader = create_dataloader(
        data_dir=test_dir,
        is_train=False,  # 测试模式：固定起点，不shuffle
        batch_size=config['validation']['batch_size'],
        num_workers=config['device']['num_workers'],
        sample_rate=config['dataset']['sample_rate'],
        chunk_size=config['dataset']['chunk_size'],
        use_all_chunks=use_all_chunks
    )
    
    logger.info(f"Use all chunks: {use_all_chunks}")
    logger.info(f"Test batches: {len(test_loader)}")
    
    # 从配置读取评估设置
    eval_config = config.get('evaluation', {})
    enabled_metrics = eval_config.get('metrics', {}).get('enabled', ['si_sdr', 'sdr', 'sir', 'sar'])
    use_asteroid = eval_config.get('use_asteroid', True)
    include_stoi = eval_config.get('include_stoi', False)
    
    # 评估（使用 Asteroid 自动 PIT + 误差分解）
    logger.info("\nEvaluating...")
    logger.info(f"Enabled metrics: {enabled_metrics}")
    
    # 如果是 SpeechBrain 模型，需要特殊处理
    if is_speechbrain:
        logger.warning("SpeechBrain 模型评估当前不支持标准评估流程")
        logger.warning("建议使用 Asteroid 模型或本地训练的模型进行标准评估")
        print("\n警告: SpeechBrain 模型评估功能尚未完全实现")
        print("建议使用 --pretrained asteroid 或本地训练模型进行评估")
        sys.exit(0)
    
    metrics = evaluate_separation(
        model, 
        test_loader, 
        device,
        metrics=enabled_metrics,
        use_asteroid=use_asteroid,
        include_stoi=include_stoi
    )
    
    # 打印结果（动态显示配置的指标）
    print("\n" + "=" * 80)
    print("Evaluation Results (Asteroid 自动 PIT + 误差分解)")
    print("=" * 80)
    
    # 指标描述
    metric_names = {
        'si_sdr': 'SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)',
        'sdr': 'SDR (Signal-to-Distortion Ratio)',
        'sir': 'SIR (Source-to-Interference Ratio)',
        'sar': 'SAR (Sources-to-Artifacts Ratio)',
        'stoi': 'STOI (Short-Time Objective Intelligibility)',
        'si_sdri': 'SI-SDRi (SI-SDR Improvement)'
    }
    
    for metric_name, value in metrics.items():
        full_name = metric_names.get(metric_name, metric_name.upper())
        if metric_name == 'stoi':
            print(f"  {full_name}: {value:.4f}")
        else:
            print(f"  {full_name}: {value:.2f} dB")
    
    print("=" * 80)
    print("指标说明:")
    print("  • SI-SDR: 尺度不变信号失真比 (目标 > 10 dB, 优秀 > 15 dB)")
    print("  • SDR: 信号失真比 (BSS Eval 标准)")
    if 'sir' in metrics:
        print("  • SIR: 信号干扰比 (衡量串扰抑制能力，越高越好)")
    if 'sar' in metrics:
        print("  • SAR: 信号伪影比 (衡量算法失真，越高越好)")
    if 'stoi' in metrics:
        print("  • STOI: 语音可懂度 (0-1，> 0.9 表示良好)")
    if 'sir' in metrics and 'sar' in metrics:
        print("  • 误差分解: SDR ≈ SIR + SAR (能量分解)")
    print("=" * 80)
    
    # 记录日志
    logger.info("\nEvaluation Results (Asteroid 自动 PIT + 误差分解):")
    for metric_name, value in metrics.items():
        if metric_name == 'stoi':
            logger.info(f"  {metric_name.upper()}: {value:.4f}")
        elif metric_name == 'sir':
            logger.info(f"  {metric_name.upper()}: {value:.2f} dB (串扰抑制)")
        elif metric_name == 'sar':
            logger.info(f"  {metric_name.upper()}: {value:.2f} dB (算法失真)")
        else:
            logger.info(f"  {metric_name.upper()}: {value:.2f} dB")
    
    # 保存结果
    result_dir = config['logging']['result_dir']
    os.makedirs(result_dir, exist_ok=True)
    
    result_file = os.path.join(result_dir, 'evaluation_metrics.json')
    with open(result_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"\nResults saved to: {result_file}")
    print(f"\nResults saved to: {result_file}")


if __name__ == "__main__":
    main()
