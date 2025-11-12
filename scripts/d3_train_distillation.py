"""
知识蒸馏训练脚本
教师模型: Conv-TasNet (5.1M) → 学生模型: BTCN (1.3M)

使用方法:
    python scripts/d3_train_distillation.py --config config/distillation_config.yml
"""

import os
import sys
import argparse
import yaml
import torch
import random
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.conv_tasnet import ConvTasNet
from models.bconv_tasnet import BConvTasNet
from dataset.dataloader import get_dataloader
from trainer.distillation_trainer import DistillationTrainer
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
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config




def create_teacher_model(config, device, logger):
    """创建并加载教师模型（从本地checkpoint加载）"""
    teacher_config = config['teacher']
    local_config = teacher_config.get('local', {})
    arch_config = local_config.get('architecture', {})
    checkpoint_path = local_config.get('checkpoint_path')
    
    logger.info("="*80)
    logger.info("加载教师模型")
    logger.info("="*80)
    
    if not checkpoint_path:
        raise ValueError("使用本地模型时必须指定 checkpoint_path")
    
    # 确保checkpoint路径是绝对路径
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(project_root, checkpoint_path)
    
    # 创建教师模型
    logger.info("创建 Conv-TasNet 模型...")
    teacher = ConvTasNet(
        num_speakers=arch_config['num_speakers'],
        encoder_filters=arch_config['encoder_filters'],
        encoder_kernel_size=arch_config['encoder_kernel_size'],
        encoder_stride=arch_config['encoder_stride'],
        bottleneck_channels=arch_config['bottleneck_channels'],
        hidden_channels=arch_config['hidden_channels'],
        skip_channels=arch_config['skip_channels'],
        kernel_size=arch_config['kernel_size'],
        num_blocks=arch_config['num_blocks'],
        num_repeats=arch_config['num_repeats'],
        norm_type=arch_config['norm_type'],
        causal=arch_config['causal']
    )
    
    # 加载本地checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"教师模型检查点不存在: {checkpoint_path}")
    
    logger.info(f"加载本地checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 兼容不同的checkpoint格式
    if 'model_state_dict' in checkpoint:
        teacher.load_state_dict(checkpoint['model_state_dict'])
        logger.info("✓ 从 'model_state_dict' 加载权重")
    elif 'state_dict' in checkpoint:
        teacher.load_state_dict(checkpoint['state_dict'])
        logger.info("✓ 从 'state_dict' 加载权重")
    else:
        teacher.load_state_dict(checkpoint)
        logger.info("✓ 直接加载权重")
    
    teacher = teacher.to(device)
    teacher.eval()
    
    # 统计参数
    total_params = sum(p.numel() for p in teacher.parameters())
    trainable_params = sum(p.numel() for p in teacher.parameters() if p.requires_grad)
    
    logger.info(f"教师模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    logger.info(f"可训练参数: {trainable_params:,}")
    
    # 测试前向传播
    logger.info("测试教师模型前向传播...")
    with torch.no_grad():
        test_input = torch.randn(1, 16000).to(device)
        try:
            test_output = teacher(test_input)
            logger.info(f"✓ 前向传播成功: {test_input.shape} -> {test_output.shape}")
        except Exception as e:
            logger.error(f"✗ 前向传播失败: {str(e)}")
            raise
    
    return teacher


def create_student_model(config, device, logger):
    """创建学生模型"""
    student_config = config['student']
    arch_config = student_config['architecture']
    
    # 创建学生模型
    student = BConvTasNet(
        num_speakers=arch_config['num_speakers'],
        encoder_filters=arch_config['encoder_filters'],
        encoder_kernel_size=arch_config['encoder_kernel_size'],
        encoder_stride=arch_config['encoder_stride'],
        bottleneck_channels=arch_config['bottleneck_channels'],
        hidden_channels=arch_config['hidden_channels'],
        skip_channels=arch_config['skip_channels'],
        kernel_size=arch_config['kernel_size'],
        num_blocks=arch_config['num_blocks'],
        num_repeats=arch_config['num_repeats'],
        norm_type=arch_config['norm_type'],
        causal=arch_config['causal']
    )
    
    student = student.to(device)
    
    # 统计参数
    total_params = sum(p.numel() for p in student.parameters())
    binary_params = sum(p.numel() for n, p in student.named_parameters() 
                       if 'binary' in n.lower())
    
    logger.info(f"学生模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    logger.info(f"二值化参数: {binary_params:,} ({binary_params/total_params*100:.2f}%)")
    
    return student


def create_dataloaders(config, logger):
    """创建数据加载器"""
    dataset_config = config['dataset']
    train_config = config['training']
    val_config = config['validation']
    device_config = config['device']
    
    # 确保数据路径是绝对路径
    data_path = dataset_config['data_path']
    if not os.path.isabs(data_path):
        data_path = os.path.join(project_root, data_path)
    
    logger.info(f"数据路径: {data_path}")
    
    # 训练集
    train_loader = get_dataloader(
        data_path=data_path,
        batch_size=train_config['batch_size'],
        num_workers=device_config['num_workers'],
        shuffle=True,
        split='train',
        chunk_size=dataset_config.get('chunk_size', 32000),
        use_all_chunks=dataset_config.get('use_all_chunks', True)
    )
    
    # 验证集
    val_loader = get_dataloader(
        data_path=data_path,
        batch_size=val_config['batch_size'],
        num_workers=device_config['num_workers'],
        shuffle=False,
        split='dev',
        chunk_size=dataset_config.get('chunk_size', 32000),
        use_all_chunks=False
    )
    
    logger.info(f"训练集: {len(train_loader)} batches")
    logger.info(f"验证集: {len(val_loader)} batches")
    
    return train_loader, val_loader


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='知识蒸馏训练脚本')
    parser.add_argument('--config', type=str, default='config/distillation_config.yml',
                       help='配置文件路径')
    parser.add_argument('--mode', type=str, default=None, choices=['basic', 'full'],
                       help='蒸馏模式: basic(L_spec+L_task) 或 full(全部5个损失) [默认使用配置文件中的设置]')
    args = parser.parse_args()
    
    # 确保配置文件路径是相对于项目根目录的
    config_path = args.config
    if not os.path.isabs(config_path):
        # 如果是相对路径，转换为相对于项目根目录的绝对路径
        config_path = os.path.join(project_root, config_path)
    
    # 加载配置
    config = load_config(config_path)
    
    # 只有明确指定时才覆盖配置中的模式
    if args.mode is not None:
        config['distillation']['mode'] = args.mode
    
    # 创建日志目录（确保使用绝对路径）
    log_dir = config['logging']['log_dir']
    checkpoint_dir = config['logging']['checkpoint_dir']
    result_dir = config['logging']['result_dir']
    
    # 转换为绝对路径
    if not os.path.isabs(log_dir):
        log_dir = os.path.join(project_root, log_dir)
    if not os.path.isabs(checkpoint_dir):
        checkpoint_dir = os.path.join(project_root, checkpoint_dir)
    if not os.path.isabs(result_dir):
        result_dir = os.path.join(project_root, result_dir)
    
    # 更新配置中的路径
    config['logging']['log_dir'] = log_dir
    config['logging']['checkpoint_dir'] = checkpoint_dir
    config['logging']['result_dir'] = result_dir
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    
    # 设置日志
    logger = setup_logger(
        'distillation_training',
        os.path.join(log_dir, 'training.log')
    )
    
    logger.info("="*80)
    logger.info("知识蒸馏训练开始")
    logger.info("="*80)
    logger.info(f"配置文件: {args.config}")
    logger.info(f"蒸馏模式: {config['distillation']['mode']}")
    logger.info(f"实验名称: {config['logging']['experiment_name']}")
    
    # 设置随机种子
    seed = config['training'].get('seed', 42)
    set_seed(seed)
    logger.info(f"随机种子: {seed}")
    
    # 设置设备
    device_config = config['device']
    if device_config['use_cuda'] and torch.cuda.is_available():
        device = torch.device(f"cuda:{device_config['gpu_ids'][0]}")
        logger.info(f"使用设备: {device} ({torch.cuda.get_device_name(0)})")
        logger.info(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        logger.info("使用设备: CPU")
    
    # 创建数据加载器
    logger.info("\n" + "="*80)
    logger.info("创建数据加载器")
    logger.info("="*80)
    train_loader, val_loader = create_dataloaders(config, logger)
    
    # 创建教师模型
    logger.info("\n" + "="*80)
    logger.info("创建教师模型 (Conv-TasNet)")
    logger.info("="*80)
    teacher = create_teacher_model(config, device, logger)
    
    # 创建学生模型
    logger.info("\n" + "="*80)
    logger.info("创建学生模型 (BTCN)")
    logger.info("="*80)
    student = create_student_model(config, device, logger)
    
    # 创建蒸馏训练器
    logger.info("\n" + "="*80)
    logger.info("创建蒸馏训练器")
    logger.info("="*80)
    trainer = DistillationTrainer(
        teacher_model=teacher,
        student_model=student,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        logger=logger
    )
    
    # 开始训练
    logger.info("\n" + "="*80)
    logger.info("开始蒸馏训练")
    logger.info("="*80)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("\n训练被用户中断")
    except Exception as e:
        logger.error(f"训练过程中出现错误: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    logger.info("\n" + "="*80)
    logger.info("知识蒸馏训练完成！")
    logger.info("="*80)
    logger.info(f"最佳SI-SDR: {trainer.best_si_sdr:.2f} dB")
    logger.info(f"模型保存目录: {checkpoint_dir}")
    logger.info(f"日志保存目录: {log_dir}")


if __name__ == "__main__":
    main()

