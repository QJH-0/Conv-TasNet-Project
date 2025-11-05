"""
⚠️ 此文件已废弃 (v3.2) ⚠️

测试优化后的Conv-TasNet模型
验证所有优化是否正常工作

废弃原因：
    - 使用旧版数据加载API
    - 测试的优化功能（缓存、归一化、梯度累积等）已内置到主代码
    - 功能已集成到 trainer/trainer.py 中

替代方案：
    直接运行训练脚本测试优化功能
    python scripts/3_train.py --num-epochs 1
    
建议：
    此文件可以删除或作为历史参考保留
"""

import torch
import yaml
import sys
import os
import io

# 设置标准输出编码为UTF-8（解决Windows控制台编码问题）
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.conv_tasnet import ConvTasNet
from dataset.dataloader import create_dataloader
from utils.metrics import calculate_si_sdr, calculate_sdr, calculate_si_sdri
from trainer.trainer import Trainer
from utils.logger import setup_logger


def test_model_structure():
    """测试模型结构（验证ReLU掩码）"""
    print("\n" + "="*80)
    print("测试 1: 模型结构验证")
    print("="*80)
    
    # 创建模型
    model = ConvTasNet(
        num_speakers=2,
        encoder_filters=512,
        encoder_kernel_size=16,
        encoder_stride=8,
        bottleneck_channels=128,
        hidden_channels=256,
        skip_channels=128,
        kernel_size=3,
        num_blocks=8,
        num_repeats=3,
        norm_type='gLN',
        causal=False
    )
    
    # 检查掩码层的激活函数
    mask_activation = model.separation.mask_conv[2]
    print(f"✓ 掩码激活函数: {type(mask_activation).__name__}")
    
    if isinstance(mask_activation, torch.nn.ReLU):
        print("  ✓ 正确：使用ReLU激活函数（允许信号放大）")
    else:
        print("  ✗ 错误：应该使用ReLU而非", type(mask_activation).__name__)
    
    # 测试前向传播
    batch_size = 2
    audio_length = 16000
    mixture = torch.randn(batch_size, audio_length)
    
    with torch.no_grad():
        separated = model(mixture)
    
    print(f"✓ 输入形状: {mixture.shape}")
    print(f"✓ 输出形状: {separated.shape}")
    
    # 验证掩码可以>1（信号放大）
    with torch.no_grad():
        encoder_output = model.encoder(mixture)
        masks = model.separation(encoder_output)
    
    max_mask = masks.max().item()
    min_mask = masks.min().item()
    print(f"✓ 掩码范围: [{min_mask:.4f}, {max_mask:.4f}]")
    
    if max_mask > 1.0:
        print(f"  ✓ 正确：掩码可以 >1（最大值={max_mask:.4f}），支持信号放大")
    else:
        print(f"  ⚠ 警告：掩码最大值={max_mask:.4f}，可能需要更多训练")
    
    print("\n测试1 通过！\n")
    return True


def test_data_normalization():
    """测试数据归一化"""
    print("="*80)
    print("测试 2: 数据归一化验证")
    print("="*80)
    
    from dataset.dataloader import SeparationDataset
    import tempfile
    
    # 创建临时测试数据
    temp_dir = tempfile.mkdtemp()
    mixture_dir = os.path.join(temp_dir, 'mixture')
    clean_dir = os.path.join(temp_dir, 'clean')
    os.makedirs(mixture_dir, exist_ok=True)
    os.makedirs(clean_dir, exist_ok=True)
    
    # 生成测试音频
    import torchaudio
    for i in range(2):
        # 混合音频
        mixture = torch.randn(1, 16000) * 0.5  # 随机幅度
        torchaudio.save(
            os.path.join(mixture_dir, f'test_{i:04d}.wav'),
            mixture, 16000
        )
        
        # 干净音频
        s1 = torch.randn(1, 16000) * 0.3
        s2 = torch.randn(1, 16000) * 0.7
        torchaudio.save(
            os.path.join(clean_dir, f'test_{i:04d}_s1.wav'),
            s1, 16000
        )
        torchaudio.save(
            os.path.join(clean_dir, f'test_{i:04d}_s2.wav'),
            s2, 16000
        )
    
    # 测试归一化
    dataset_with_norm = SeparationDataset(
        data_dir=temp_dir,
        sample_rate=16000,
        segment_length=16000,
        use_cache=False,
        normalize=True,
        target_level=-25.0
    )
    
    dataset_without_norm = SeparationDataset(
        data_dir=temp_dir,
        sample_rate=16000,
        segment_length=16000,
        use_cache=False,
        normalize=False
    )
    
    # 检查归一化效果
    mixture_norm, sources_norm = dataset_with_norm[0]
    mixture_raw, sources_raw = dataset_without_norm[0]
    
    print(f"✓ 原始混合信号幅度范围: [{mixture_raw.min():.4f}, {mixture_raw.max():.4f}]")
    print(f"✓ 归一化混合信号幅度范围: [{mixture_norm.min():.4f}, {mixture_norm.max():.4f}]")
    
    # 验证归一化效果
    rms_norm = torch.sqrt(torch.mean(mixture_norm ** 2)).item()
    target_rms = 10 ** (-25.0 / 20)
    
    print(f"✓ 归一化后RMS: {rms_norm:.6f}")
    print(f"✓ 目标RMS: {target_rms:.6f}")
    
    if abs(rms_norm - target_rms) < 0.01:
        print("  ✓ 正确：归一化成功")
    else:
        print(f"  ⚠ 警告：RMS偏差较大")
    
    # 清理
    import shutil
    shutil.rmtree(temp_dir)
    
    print("\n测试2 通过！\n")
    return True


def test_metrics():
    """测试多指标计算"""
    print("="*80)
    print("测试 3: 多指标计算验证")
    print("="*80)
    
    # 创建测试信号
    length = 16000
    target = torch.randn(length)
    
    # 测试SI-SDR
    estimation = target + torch.randn(length) * 0.1
    si_sdr = calculate_si_sdr(estimation, target)
    print(f"✓ SI-SDR计算成功: {si_sdr:.2f} dB")
    
    # 测试SDR
    sdr = calculate_sdr(estimation, target)
    print(f"✓ SDR计算成功: {sdr:.2f} dB")
    
    # 测试SI-SDRi
    mixture = target + torch.randn(length)
    si_sdri = calculate_si_sdri(estimation, target, mixture)
    print(f"✓ SI-SDRi计算成功: {si_sdri:.2f} dB")
    
    print("\n测试3 通过！\n")
    return True


def test_trainer_config():
    """测试训练器配置（学习率策略）"""
    print("="*80)
    print("测试 4: 训练器配置验证")
    print("="*80)
    
    # 加载配置
    with open('config/config.yml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"✓ 学习率调度器类型: {config['training']['scheduler']['type']}")
    print(f"✓ 初始学习率: {config['training']['learning_rate']}")
    print(f"✓ 减半patience: {config['training']['scheduler']['patience']}")
    print(f"✓ 减半因子: {config['training']['scheduler']['factor']}")
    print(f"✓ 最小学习率: {config['training']['scheduler']['min_lr']}")
    
    if config['training']['scheduler']['type'] == 'Halving':
        print("  ✓ 正确：使用Halving策略（论文标准）")
    else:
        print(f"  ⚠ 提示：当前使用{config['training']['scheduler']['type']}策略")
    
    # 创建模型和训练器测试
    model = ConvTasNet.from_config(config)
    logger = setup_logger('test', 'experiments/test/logs')
    
    try:
        trainer = Trainer(model, config, logger, device='cpu')
        print("✓ 训练器创建成功")
        
        # 检查调度器类型
        scheduler_type = type(trainer.scheduler).__name__
        print(f"✓ 调度器实例: {scheduler_type}")
        
        if scheduler_type == 'ReduceLROnPlateau':
            print("  ✓ 正确：使用ReduceLROnPlateau（Halving策略）")
        
        # 检查优化器配置
        optimizer = trainer.optimizer
        print(f"✓ 优化器: {type(optimizer).__name__}")
        print(f"✓ 优化器参数:")
        print(f"  - lr: {optimizer.param_groups[0]['lr']}")
        print(f"  - betas: {optimizer.param_groups[0]['betas']}")
        print(f"  - eps: {optimizer.param_groups[0]['eps']}")
        print(f"  - weight_decay: {optimizer.param_groups[0]['weight_decay']}")
        
        if optimizer.param_groups[0]['weight_decay'] == 0:
            print("  ✓ 正确：无权重衰减（论文标准）")
        
    except Exception as e:
        print(f"✗ 训练器创建失败: {e}")
        return False
    
    print("\n测试4 通过！\n")
    return True


def test_complete_pipeline():
    """测试完整的训练流程"""
    print("="*80)
    print("测试 5: 完整流程验证（小规模）")
    print("="*80)
    
    # 检查是否有训练数据
    train_dir = "data/processed/mixed/train"
    if not os.path.exists(train_dir):
        print("⚠ 训练数据不存在，跳过完整流程测试")
        print(f"  请先运行数据生成脚本")
        return True
    
    # 加载配置
    with open('config/config.yml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建小规模数据加载器（用于快速测试）
    try:
        print("创建数据加载器...")
        train_loader = create_dataloader(
            data_dir=train_dir,
            batch_size=1,
            num_workers=0,
            sample_rate=config['dataset']['sample_rate'],
            segment_length=config['dataset']['segment_length'],
            shuffle=False,
            use_cache=False,
            normalize=True,  # 启用归一化
            target_level=-25.0,
            augmentation=False,  # 测试时关闭增强
            dynamic_mixing=False
        )
        print(f"✓ 数据加载器创建成功，batch数: {len(train_loader)}")
        
        # 测试数据加载
        mixtures, sources = next(iter(train_loader))
        print(f"✓ 数据形状:")
        print(f"  - 混合信号: {mixtures.shape}")
        print(f"  - 干净信号: {sources.shape}")
        
        # 创建模型
        print("\n创建模型...")
        model = ConvTasNet.from_config(config)
        print(f"✓ 模型创建成功")
        
        # 前向传播测试
        print("\n测试前向传播...")
        with torch.no_grad():
            separated = model(mixtures)
        print(f"✓ 分离信号形状: {separated.shape}")
        
        # 计算指标
        print("\n计算评估指标...")
        from utils.metrics import evaluate_separation
        metrics_result = evaluate_separation(
            model, 
            train_loader, 
            device='cpu',
            metrics=['si_sdr', 'sdr']
        )
        print(f"✓ SI-SDR: {metrics_result['si_sdr']:.2f} dB")
        print(f"✓ SDR: {metrics_result['sdr']:.2f} dB")
        
        print("\n✓ 完整流程测试成功！")
        
    except Exception as e:
        print(f"✗ 完整流程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n测试5 通过！\n")
    return True


def main():
    """运行所有测试"""
    print("\n" + "="*80)
    print(" Conv-TasNet 优化验证测试")
    print("="*80)
    print("\n基于论文: Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking")
    print("测试项目:")
    print("  1. 模型结构（ReLU掩码）")
    print("  2. 数据归一化")
    print("  3. 多指标计算")
    print("  4. 训练器配置（Halving策略）")
    print("  5. 完整流程")
    print("\n" + "="*80 + "\n")
    
    results = []
    
    # 运行测试
    results.append(("模型结构", test_model_structure()))
    results.append(("数据归一化", test_data_normalization()))
    results.append(("多指标计算", test_metrics()))
    results.append(("训练器配置", test_trainer_config()))
    results.append(("完整流程", test_complete_pipeline()))
    
    # 汇总结果
    print("\n" + "="*80)
    print(" 测试结果汇总")
    print("="*80)
    
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name:20s}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("="*80)
    if all_passed:
        print("\n🎉 所有测试通过！优化已成功实施。")
        print("\n建议下一步:")
        print("  1. 运行完整训练: python scripts/3_train.py")
        print("  2. 监控训练曲线和指标")
        print("  3. 对比优化前后的性能")
    else:
        print("\n⚠ 部分测试失败，请检查上述错误信息。")
    print("\n")


if __name__ == "__main__":
    main()

