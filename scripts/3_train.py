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


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Train Conv-TasNet')
    parser.add_argument('--config', type=str, default='config/config.yml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from (overrides config)')
    
    # 数据路径参数（可覆盖config）
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Processed data directory (override config)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for logs/checkpoints (override config)')
    
    # 训练参数（可覆盖config）
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (override config)')
    parser.add_argument('--num-epochs', type=int, default=None,
                       help='Number of epochs (override config)')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate (override config)')
    parser.add_argument('--use-amp', action='store_true', default=None,
                       help='Use automatic mixed precision (override config)')
    parser.add_argument('--no-amp', action='store_true',
                       help='Disable automatic mixed precision')
    
    # GPU参数
    parser.add_argument('--gpu-ids', type=str, default=None,
                       help='GPU IDs to use (comma separated, e.g., 0,1)')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 应用命令行参数覆盖
    if args.data_dir:
        config['dataset']['processed_data_path'] = args.data_dir
    if args.output_dir:
        # 同时覆盖所有输出路径
        config['logging']['log_dir'] = os.path.join(args.output_dir, 'logs')
        config['logging']['checkpoint_dir'] = os.path.join(args.output_dir, 'checkpoints')
        config['logging']['result_dir'] = os.path.join(args.output_dir, 'results')
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.num_epochs:
        config['training']['num_epochs'] = args.num_epochs
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if args.use_amp:
        config['training']['use_amp'] = True
    if args.no_amp:
        config['training']['use_amp'] = False
    if args.gpu_ids:
        gpu_ids = [int(i) for i in args.gpu_ids.split(',')]
        config['device']['gpu_ids'] = gpu_ids
    
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
    gpu_ids = device_config['gpu_ids']
    if device_config['use_cuda'] and torch.cuda.is_available():
        device = f"cuda:{gpu_ids[0]}"
        print(f"Using device: {device}")
        if len(gpu_ids) > 1:
            print(f"Using {len(gpu_ids)} GPUs: {gpu_ids}")
        for gpu_id in gpu_ids:
            print(f"  GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
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
    
    # 多GPU支持
    if len(gpu_ids) > 1 and torch.cuda.is_available():
        logger.info(f"Using DataParallel with {len(gpu_ids)} GPUs: {gpu_ids}")
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    
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
    
    # 获取原始模型（如果使用了DataParallel，需要访问model.module）
    base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    
    # 按模块统计参数
    encoder_params = sum(p.numel() for p in base_model.encoder.parameters())
    separation_params = sum(p.numel() for p in base_model.separation.parameters())
    decoder_params = sum(p.numel() for p in base_model.decoder.parameters())
    
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
