"""
BTCN评估脚本
评估训练好的二值化Conv-TasNet模型
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

from models.bconv_tasnet import BConvTasNet
from dataset.dataloader import create_dataloader
from utils.metrics import evaluate_separation
from utils.logger import setup_logger


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


def load_model(checkpoint_path, config, device):
    """加载模型（兼容单GPU和多GPU训练的checkpoint）"""
    from collections import OrderedDict
    
    model = BConvTasNet.from_config(config)
    
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


def find_best_checkpoint(config):
    """查找最佳模型检查点"""
    checkpoint_dir = config['logging']['checkpoint_dir']
    if not os.path.isabs(checkpoint_dir):
        checkpoint_dir = os.path.join(project_root, checkpoint_dir)
    
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        return best_model_path
    
    # 如果没有best_model.pth，查找最新的检查点
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if checkpoints:
        # 按修改时间排序，返回最新的
        checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
        return os.path.join(checkpoint_dir, checkpoints[0])
    
    return None


def count_binary_params(model):
    """统计二值化参数和全精度参数"""
    binary_params = 0
    full_precision_params = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        
        # 检查所有二值化层：binary_conv1x1_1, binary_conv1x1_2, binary_dw_conv, binary_skip_conv
        if any(keyword in name for keyword in ['binary_conv1x1', 'binary_dw_conv', 'binary_skip_conv']):
            binary_params += param_count
        else:
            full_precision_params += param_count
    
    return binary_params, full_precision_params, total_params


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Evaluate Binary Conv-TasNet (BTCN)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint (default: auto-find best model)')
    parser.add_argument('--config', type=str, default='config/bconfig.yml',
                       help='Path to config file')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Path to test data directory (override config)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results (override config)')
    parser.add_argument('--gpu-id', type=int, default=None,
                       help='GPU ID to use (override config)')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 应用命令行参数覆盖
    if args.output_dir:
        config['logging']['result_dir'] = os.path.join(args.output_dir, 'results')
        config['logging']['checkpoint_dir'] = os.path.join(args.output_dir, 'checkpoints')
    if args.gpu_id is not None:
        config['device']['gpu_ids'] = [args.gpu_id]
    
    # 自动查找检查点
    if args.checkpoint is None:
        print("未指定checkpoint，自动查找最佳模型...")
        args.checkpoint = find_best_checkpoint(config)
        if args.checkpoint is None:
            print("错误: 未找到任何模型检查点!")
            print(f"请检查目录: {config['logging']['checkpoint_dir']}")
            sys.exit(1)
        print(f"找到模型: {args.checkpoint}")
    
    # 设置设备
    device_config = config['device']
    if device_config['use_cuda'] and torch.cuda.is_available():
        device = f"cuda:{device_config['gpu_ids'][0]}"
    else:
        device = 'cpu'
    
    print("=" * 80)
    print("Binary Conv-TasNet (BTCN) Evaluation")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    
    # 设置日志
    log_dir = config['logging']['log_dir']
    logger = setup_logger('eval', log_dir, 'evaluation.log')
    
    # 加载模型
    logger.info("Loading BTCN model...")
    model = load_model(args.checkpoint, config, device)
    logger.info("Model loaded successfully")
    
    # 统计模型参数
    binary_params, full_precision_params, total_params = count_binary_params(model)
    
    logger.info("\nModel Statistics:")
    logger.info(f"  Total Parameters: {total_params:,}")
    logger.info(f"  Binary Parameters: {binary_params:,} ({binary_params/total_params*100:.1f}%)")
    logger.info(f"  Full-Precision Parameters: {full_precision_params:,} ({full_precision_params/total_params*100:.1f}%)")
    
    # 估算模型大小
    model_size_fp32 = total_params * 4 / (1024 ** 2)  # FP32
    model_size_binary = (binary_params / 32 + full_precision_params) * 4 / (1024 ** 2)  # 二值化
    
    logger.info(f"\nModel Size:")
    logger.info(f"  FP32: {model_size_fp32:.2f} MB")
    logger.info(f"  Binary (1-bit for binary params): {model_size_binary:.2f} MB")
    logger.info(f"  Compression Ratio: {model_size_fp32/model_size_binary:.2f}x")
    
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
    
    logger.info(f"\nTest data directory: {test_dir}")
    
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
    logger.info("\nEvaluating BTCN model...")
    logger.info(f"Enabled metrics: {enabled_metrics}")
    metrics = evaluate_separation(
        model, 
        test_loader, 
        device,
        metrics=enabled_metrics,
        use_asteroid=use_asteroid,
        include_stoi=include_stoi
    )
    
    # 打印结果（包含误差分解）
    print("\n" + "=" * 80)
    print("BTCN Evaluation Results (Asteroid 自动 PIT + 误差分解)")
    print("=" * 80)
    print(f"  SI-SDR (Scale-Invariant Signal-to-Distortion Ratio): {metrics.get('si_sdr', 0):.2f} dB")
    print(f"  SDR (Signal-to-Distortion Ratio): {metrics.get('sdr', 0):.2f} dB")
    print(f"  SIR (Source-to-Interference Ratio): {metrics.get('sir', 0):.2f} dB")
    print(f"  SAR (Sources-to-Artifacts Ratio): {metrics.get('sar', 0):.2f} dB")
    if 'si_sdri' in metrics:
        print(f"  SI-SDRi (SI-SDR Improvement): {metrics['si_sdri']:.2f} dB")
    print("=" * 80)
    print("Model Efficiency:")
    print(f"  Model Size (Binary): {model_size_binary:.2f} MB")
    print(f"  Compression Ratio: {model_size_fp32/model_size_binary:.2f}x")
    print(f"  Binary Parameters: {binary_params/total_params*100:.1f}%")
    print("=" * 80)
    print("指标说明:")
    print("  • SI-SDR: 尺度不变信号失真比 (目标 > 10 dB, 优秀 > 15 dB)")
    print("  • SDR: 信号失真比 (BSS Eval 标准)")
    print("  • SIR: 信号干扰比 (衡量串扰抑制能力)")
    print("  • SAR: 信号伪影比 (衡量算法失真)")
    print("  • 误差分解: SDR ≈ SIR + SAR")
    print("  • 二值化模型性能通常略低于全精度模型")
    print("  • 可以通过知识蒸馏进一步提升性能")
    print("=" * 80)
    
    # 记录日志
    logger.info("\nEvaluation Results (Asteroid 自动 PIT + 误差分解):")
    logger.info(f"  SI-SDR: {metrics.get('si_sdr', 0):.2f} dB")
    logger.info(f"  SDR: {metrics.get('sdr', 0):.2f} dB")
    logger.info(f"  SIR: {metrics.get('sir', 0):.2f} dB (串扰抑制)")
    logger.info(f"  SAR: {metrics.get('sar', 0):.2f} dB (算法失真)")
    if 'si_sdri' in metrics:
        logger.info(f"  SI-SDRi: {metrics['si_sdri']:.2f} dB")
    
    # 保存结果
    result_dir = config['logging']['result_dir']
    os.makedirs(result_dir, exist_ok=True)
    
    # 添加模型统计信息到metrics
    metrics['model_stats'] = {
        'total_params': total_params,
        'binary_params': binary_params,
        'full_precision_params': full_precision_params,
        'binary_params_ratio': binary_params / total_params,
        'model_size_fp32_mb': model_size_fp32,
        'model_size_binary_mb': model_size_binary,
        'compression_ratio': model_size_fp32 / model_size_binary
    }
    
    result_file = os.path.join(result_dir, 'evaluation_metrics.json')
    with open(result_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"\nResults saved to: {result_file}")
    print(f"\nResults saved to: {result_file}")
    
    # 与全精度模型对比说明
    print("\n" + "=" * 80)
    print("Next Steps:")
    print("=" * 80)
    print("  1. Compare with full-precision Conv-TasNet performance")
    print("  2. Analyze the trade-off between model size and accuracy")
    print("  3. Consider applying knowledge distillation to improve performance")
    print("  4. Test on real-world audio samples")
    print("=" * 80)


if __name__ == "__main__":
    main()


