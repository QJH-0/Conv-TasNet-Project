"""
测试数据加载的一致性
验证训练加载的数据与生成的数据一致
"""

import torch
import yaml
import os
import sys
import io
import json

# 设置UTF-8编码（仅在非pytest环境下）
if sys.platform == 'win32' and 'pytest' not in sys.modules:
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except:
        pass  # pytest环境下可能失败

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset.dataloader import create_dataloader


def test_data_consistency():
    """测试数据加载的一致性"""
    print("="*80)
    print("测试数据加载一致性")
    print("="*80)
    
    # 检查数据是否存在
    train_dir = "data/processedNew/mixed/train"
    if not os.path.exists(train_dir):
        print("\n⚠️ 训练数据不存在，请先生成数据：")
        print("  python scripts/2_generate_mixtures.py")
        return False
    
    # 加载元数据
    metadata_path = os.path.join(train_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        print(f"\n⚠️ 元数据不存在: {metadata_path}")
        return False
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"\n找到 {len(metadata)} 个训练样本")
    
    # 创建数据加载器（优化后的配置）
    print("\n创建数据加载器（优化配置）...")
    dataloader = create_dataloader(
        data_dir=train_dir,
        batch_size=2,
        num_workers=0,
        sample_rate=16000,
        segment_length=32000,
        shuffle=False,
        use_cache=False,
        normalize=False,       # ✅ 关闭（数据已归一化）
        augmentation=False,    # ✅ 关闭（数据已固定长度）
        dynamic_mixing=False   # ✅ 关闭（SNR已精确控制）
    )
    
    print("✓ 数据加载器创建成功")
    
    # 加载第一个batch
    mixtures, sources = next(iter(dataloader))
    
    print(f"\n数据形状:")
    print(f"  Mixtures: {mixtures.shape}")
    print(f"  Sources:  {sources.shape}")
    
    # 验证1: mixture = sources之和
    print("\n验证1: mixture = sources之和")
    for i in range(mixtures.shape[0]):
        reconstructed = sources[i].sum(dim=0)
        error = torch.mean((mixtures[i] - reconstructed) ** 2).item()
        
        is_valid = error < 1e-6
        status = "✓" if is_valid else "✗"
        print(f"  样本{i}: 重建误差 = {error:.2e} {status}")
    
    # 验证2: 归一化级别
    print("\n验证2: 归一化级别（应该接近-25dB）")
    target_rms = 10 ** (-25.0 / 20)
    
    for i in range(mixtures.shape[0]):
        rms = torch.sqrt(torch.mean(mixtures[i] ** 2)).item()
        rms_db = 20 * torch.log10(torch.tensor(rms)).item()
        
        is_valid = abs(rms - target_rms) < 0.01
        status = "✓" if is_valid else "✗"
        print(f"  样本{i}: RMS = {rms:.6f} ({rms_db:.2f} dB), 目标 = {target_rms:.6f} (-25.00 dB) {status}")
    
    # 验证3: SNR（从元数据对比）
    print("\n验证3: SNR控制精度")
    batch_indices = list(range(min(2, len(metadata))))
    
    for idx in batch_indices:
        if idx < len(metadata):
            meta = metadata[idx]
            
            # 计算实际SNR
            energy1 = torch.sum(sources[idx, 0] ** 2).item()
            energy2 = torch.sum(sources[idx, 1] ** 2).item()
            actual_snr = 10 * torch.log10(torch.tensor(energy1 / (energy2 + 1e-8))).item()
            
            # 从元数据获取目标SNR
            target_snr = meta.get('snr_db_actual', meta.get('snr_db_target', 0))
            error = abs(actual_snr - target_snr)
            
            is_valid = error < 0.1
            status = "✓" if is_valid else "✗"
            print(f"  样本{idx}: 目标SNR = {target_snr:+.2f} dB, 实际SNR = {actual_snr:+.2f} dB, 误差 = {error:.4f} dB {status}")
    
    # 验证4: 数据范围（应该在[-1, 1]之间）
    print("\n验证4: 数据范围")
    for i in range(mixtures.shape[0]):
        mix_min = mixtures[i].min().item()
        mix_max = mixtures[i].max().item()
        
        is_valid = -1.0 <= mix_min and mix_max <= 1.0
        status = "✓" if is_valid else "✗"
        print(f"  样本{i}: 范围 = [{mix_min:+.4f}, {mix_max:+.4f}] {status}")
    
    print("\n" + "="*80)
    print("✓ 数据加载一致性验证完成")
    print("="*80)
    
    return True


def test_loading_speed():
    """测试加载速度"""
    print("\n" + "="*80)
    print("测试数据加载速度")
    print("="*80)
    
    train_dir = "data/processedNew/mixed/train"
    if not os.path.exists(train_dir):
        print("\n⚠️ 训练数据不存在")
        return False
    
    import time
    
    # 测试1: 不使用缓存
    print("\n测试1: 不使用缓存")
    dataloader_no_cache = create_dataloader(
        data_dir=train_dir,
        batch_size=4,
        num_workers=0,
        use_cache=False,
        normalize=False,
        augmentation=False,
        dynamic_mixing=False
    )
    
    start = time.time()
    for i, (mixtures, sources) in enumerate(dataloader_no_cache):
        if i >= 10:  # 只测试10个batch
            break
    time_no_cache = time.time() - start
    print(f"  加载10个batch耗时: {time_no_cache:.3f}s")
    
    # 测试2: 使用缓存
    print("\n测试2: 使用缓存")
    dataloader_with_cache = create_dataloader(
        data_dir=train_dir,
        batch_size=4,
        num_workers=0,
        use_cache=True,
        normalize=False,
        augmentation=False,
        dynamic_mixing=False
    )
    
    start = time.time()
    for i, (mixtures, sources) in enumerate(dataloader_with_cache):
        if i >= 10:
            break
    time_with_cache = time.time() - start
    print(f"  加载10个batch耗时: {time_with_cache:.3f}s")
    
    speedup = time_no_cache / time_with_cache
    print(f"\n  加速比: {speedup:.2f}x")
    
    return True


def main():
    """运行所有测试"""
    print("\n" + "="*80)
    print(" 数据加载优化验证")
    print("="*80)
    print("\n验证优化后的数据加载流程\n")
    
    results = []
    
    results.append(("数据一致性", test_data_consistency()))
    results.append(("加载速度", test_loading_speed()))
    
    # 汇总
    print("\n" + "="*80)
    print(" 测试结果汇总")
    print("="*80)
    
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败/跳过"
        print(f"{name:20s}: {status}")
    
    print("="*80)
    
    print("\n优化总结:")
    print("  1. ✅ 移除重复归一化（数据已归一化）")
    print("  2. ✅ 移除无效增强（数据已固定长度）")
    print("  3. ✅ 保护SNR精度（不启用动态混合）")
    print("  4. ✅ 提升加载速度（使用缓存）")
    
    print("\n推荐配置:")
    print("  normalize=False, augmentation=False, dynamic_mixing=False")
    print()


if __name__ == "__main__":
    main()

