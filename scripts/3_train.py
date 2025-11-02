"""
训练脚本
使用配置文件训练 Conv-TasNet 模型
"""

import os
import sys
import argparse
import yaml
import torch
import random
import numpy as np

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
# 切换到项目根目录,确保所有相对路径正确
os.chdir(project_root)

from models.conv_tasnet import ConvTasNet
from dataset.dataloader import create_dataloader
from trainer.trainer import Trainer
from utils.logger import setup_logger


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Train Conv-TasNet')
    parser.add_argument('--config', type=str, default='config/config.yml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from (overrides config)')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    print("=" * 80)
    print("Configuration loaded successfully")
    print("=" * 80)
    
    # 确定resume路径: 命令行参数优先,否则使用配置文件
    resume_checkpoint = args.resume
    if resume_checkpoint is None:
        resume_checkpoint = config['training'].get('resume_checkpoint', None)
    
    if resume_checkpoint:
        # 如果是相对路径,转换为绝对路径
        if not os.path.isabs(resume_checkpoint):
            resume_checkpoint = os.path.join(project_root, resume_checkpoint)
        print(f"Will resume from checkpoint: {resume_checkpoint}")
    
    # 设置随机种子
    seed = config['training'].get('seed', 42)
    set_seed(seed)
    print(f"Random seed set to: {seed}")
    
    # 设置设备
    device_config = config['device']
    if device_config['use_cuda'] and torch.cuda.is_available():
        device = f"cuda:{device_config['gpu_ids'][0]}"
        print(f"Using device: {device}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print(f"Using device: {device}")
    
    # 设置日志
    log_dir = config['logging']['log_dir']
    logger = setup_logger('train', log_dir)
    
    logger.info("=" * 80)
    logger.info("Conv-TasNet Training")
    logger.info("=" * 80)
    
    # 创建模型
    logger.info("Creating model...")
    model = ConvTasNet.from_config(config)
    
    # 输出模型配置
    logger.info("\nModel Configuration:")
    logger.info("-" * 80)
    logger.info(f"Model Name: {config['model']['name']}")
    logger.info(f"\nEncoder:")
    logger.info(f"  - Filters: {config['model']['encoder']['num_filters']}")
    logger.info(f"  - Kernel Size: {config['model']['encoder']['kernel_size']}")
    logger.info(f"  - Stride: {config['model']['encoder']['stride']}")
    logger.info(f"\nSeparation (TCN):")
    logger.info(f"  - Speakers: {config['model']['separation']['num_speakers']}")
    logger.info(f"  - Bottleneck Channels: {config['model']['separation']['bottleneck_channels']}")
    logger.info(f"  - Hidden Channels: {config['model']['separation']['hidden_channels']}")
    logger.info(f"  - Skip Channels: {config['model']['separation']['skip_channels']}")
    logger.info(f"  - Kernel Size: {config['model']['separation']['kernel_size']}")
    logger.info(f"  - Blocks per Repeat: {config['model']['separation']['num_blocks']}")
    logger.info(f"  - Num Repeats: {config['model']['separation']['num_repeats']}")
    logger.info(f"  - Normalization: {config['model']['separation']['norm_type']}")
    logger.info(f"  - Causal: {config['model']['separation']['causal']}")
    logger.info(f"\nDecoder:")
    logger.info(f"  - Filters: {config['model']['decoder']['num_filters']}")
    logger.info(f"  - Kernel Size: {config['model']['decoder']['kernel_size']}")
    logger.info(f"  - Stride: {config['model']['decoder']['stride']}")
    
    # 计算参数量
    logger.info("\n" + "-" * 80)
    logger.info("Model Parameters:")
    logger.info("-" * 80)
    
    # 按模块统计参数
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    separation_params = sum(p.numel() for p in model.separation.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Encoder Parameters: {encoder_params:,} ({encoder_params/total_params*100:.1f}%)")
    logger.info(f"Separation Parameters: {separation_params:,} ({separation_params/total_params*100:.1f}%)")
    logger.info(f"Decoder Parameters: {decoder_params:,} ({decoder_params/total_params*100:.1f}%)")
    logger.info(f"\nTotal Parameters: {total_params:,}")
    logger.info(f"Trainable Parameters: {trainable_params:,}")
    logger.info(f"Total Parameters: {total_params / 1e6:.2f} M")  # 参数数量（百万）
    logger.info(f"Model Size (FP32): {total_params * 4 / (1024 ** 2):.2f} MB")  # 存储大小
    logger.info("-" * 80)
    
    # 输出训练配置
    # logger.info("\nTraining Configuration:")
    # logger.info("-" * 80)
    # logger.info(f"Batch Size: {config['training']['batch_size']}")
    # logger.info(f"Num Epochs: {config['training']['num_epochs']}")
    # logger.info(f"Learning Rate: {config['training']['learning_rate']}")
    # logger.info(f"Optimizer: {config['training']['optimizer']}")
    # logger.info(f"Gradient Clip: {config['training']['gradient_clip']}")
    # logger.info(f"Early Stopping Patience: {config['training']['early_stopping_patience']}")
    # logger.info(f"LR Scheduler: {config['training']['scheduler']['type']}")
    # logger.info(f"  - Mode: {config['training']['scheduler']['mode']}")
    # logger.info(f"  - Patience: {config['training']['scheduler']['patience']}")
    # logger.info(f"  - Factor: {config['training']['scheduler']['factor']}")
    # logger.info(f"  - Min LR: {config['training']['scheduler']['min_lr']}")
    #
    # logger.info("\nDataset Configuration:")
    # logger.info(f"Dataset: {config['dataset']['name']}")
    # logger.info(f"Sample Rate: {config['dataset']['sample_rate']} Hz")
    # logger.info(f"Audio Length: {config['dataset']['audio_length']} seconds")
    # logger.info(f"Segment Length: {config['dataset']['segment_length']} samples")
    # logger.info(f"SNR Range: {config['dataset']['snr_range']} dB")
    #
    # logger.info("\nPaths Configuration:")
    # logger.info(f"Project Root: {project_root}")
    # logger.info(f"Config File: {args.config}")
    # logger.info(f"Data Path: {config['dataset']['processed_data_path']}")
    # logger.info(f"Log Dir: {config['logging']['log_dir']}")
    # logger.info(f"Checkpoint Dir: {config['logging']['checkpoint_dir']}")
    # logger.info(f"Result Dir: {config['logging']['result_dir']}")
    # if resume_checkpoint:
    #     logger.info(f"Resume From: {resume_checkpoint}")
    # logger.info("-" * 80)
    
    # 创建数据加载器
    logger.info("\nCreating dataloaders...")
    
    # 路径已在load_config中规范化为绝对路径
    train_dir = os.path.join(
        config['dataset']['processed_data_path'],
        'mixed', 'train'
    )
    val_dir = os.path.join(
        config['dataset']['processed_data_path'],
        'mixed', 'test'
    )
    
    train_loader = create_dataloader(
        data_dir=train_dir,
        batch_size=config['training']['batch_size'],
        num_workers=config['device']['num_workers'],
        sample_rate=config['dataset']['sample_rate'],
        segment_length=config['dataset']['segment_length'],
        shuffle=True,
        pin_memory=True,
        use_cache=config['dataset'].get('use_cache', True),
        # ✅ 优化配置：避免重复处理
        normalize=False,       # 数据生成时已归一化，不需要再次处理
        augmentation=False,    # 数据已固定长度，随机裁剪无效
        dynamic_mixing=False   # 保护数据生成时的精确SNR控制
    )
    
    val_loader = create_dataloader(
        data_dir=val_dir,
        batch_size=config['validation']['batch_size'],
        num_workers=config['device']['num_workers'],
        sample_rate=config['dataset']['sample_rate'],
        segment_length=config['dataset']['segment_length'],
        shuffle=False,
        pin_memory=True,
        use_cache=config['dataset'].get('use_cache', True),
        # ✅ 验证集同样使用优化配置
        normalize=False,
        augmentation=False,
        dynamic_mixing=False
    )
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    
    # 创建训练器
    logger.info("Creating trainer...")
    trainer = Trainer(
        model=model,
        config=config,
        logger=logger,
        device=device
    )
    
    # 开始训练
    trainer.run(
        train_loader=train_loader,
        val_loader=val_loader,
        resume_from=resume_checkpoint
    )
    
    logger.info("Training finished!")


if __name__ == "__main__":
    main()
