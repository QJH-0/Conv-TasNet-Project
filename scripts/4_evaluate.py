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
from dataset.dataloader import create_dataloader
from utils.metrics import evaluate_separation
from utils.logger import setup_logger


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
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
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


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Evaluate Conv-TasNet')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint (default: auto-find best model)')
    parser.add_argument('--config', type=str, default='config/config.yml',
                       help='Path to config file')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Path to test data directory')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
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
    print("Conv-TasNet Evaluation")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    
    # 设置日志
    log_dir = config['logging']['log_dir']
    logger = setup_logger('eval', log_dir, 'evaluation.log')
    
    # 加载模型
    logger.info("Loading model...")
    model = load_model(args.checkpoint, config, device)
    logger.info("Model loaded successfully")
    
    # 创建数据加载器
    if args.data_dir:
        test_dir = args.data_dir
    else:
        test_dir = os.path.join(
            config['dataset']['processed_data_path'],
            'mixed/test'
        )
    
    logger.info(f"Test data directory: {test_dir}")
    
    test_loader = create_dataloader(
        data_dir=test_dir,
        batch_size=config['validation']['batch_size'],
        num_workers=config['device']['num_workers'],
        sample_rate=config['dataset']['sample_rate'],
        segment_length=config['dataset']['segment_length'],
        shuffle=False,
        pin_memory=False
    )
    
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    logger.info(f"Test batches: {len(test_loader)}")
    
    # 评估
    logger.info("\nEvaluating...")
    metrics = evaluate_separation(model, test_loader, device)
    
    # 打印结果
    print("\n" + "=" * 80)
    print("Evaluation Results - SI-SDR")
    print("=" * 80)
    print(f"  SI-SDR (Scale-Invariant Signal-to-Distortion Ratio): {metrics['si_sdr']:.2f} dB")
    print("=" * 80)
    print("说明:")
    print("  • SI-SDR: 衡量分离信号的质量，越大越好")
    print("  • SI-SDR > 10 dB 表示良好分离")
    print("  • SI-SDR > 15 dB 表示优秀分离")
    print("=" * 80)
    
    # 记录日志
    logger.info("\nEvaluation Results - SI-SDR:")
    logger.info(f"  SI-SDR (Scale-Invariant Signal-to-Distortion Ratio): {metrics['si_sdr']:.2f} dB")
    
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
