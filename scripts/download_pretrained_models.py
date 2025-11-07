"""
下载预训练模型工具脚本
自动下载并保存模型信息
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.pretrained_model_utils import (
    load_asteroid_model, 
    load_speechbrain_model, 
    print_available_models,
    list_available_models
)


def download_model(source, model_name, cache_dir=None, device='cpu'):
    """
    下载预训练模型并保存信息
    
    Args:
        source: 'asteroid' 或 'speechbrain'
        model_name: 模型名称
        cache_dir: 缓存目录
        device: 设备（默认 cpu，下载时不需要 GPU）
    """
    print("="*80)
    print(f"下载预训练模型")
    print("="*80)
    print(f"来源: {source}")
    print(f"模型名: {model_name}")
    
    # 设置默认缓存目录
    if cache_dir is None:
        cache_dir = os.path.join(project_root, f"pretrained_models/{source}")
    
    print(f"缓存目录: {cache_dir}")
    print("\n开始下载...")
    
    try:
        if source == 'asteroid':
            model, info = load_asteroid_model(model_name, device, cache_dir, save_info=True)
        elif source == 'speechbrain':
            model, info = load_speechbrain_model(model_name, device, cache_dir, save_info=True)
        else:
            print(f"错误: 不支持的来源 '{source}'")
            return False
        
        print("\n" + "="*80)
        print("下载成功！")
        print("="*80)
        
        if info:
            print("\n模型信息:")
            print(f"  模型名称: {info.get('model_name', 'N/A')}")
            print(f"  模型类型: {info.get('model_class', 'N/A')}")
            if 'total_parameters_M' in info:
                print(f"  参数量: {info['total_parameters_M']}M")
            if 'file_size_MB' in info:
                print(f"  文件大小: {info['file_size_MB']}MB")
            print(f"  保存目录: {info.get('model_dir', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"\n错误: 下载失败")
        print(f"详细信息: {str(e)}")
        return False


def download_all_asteroid_models(cache_dir=None, device='cpu'):
    """下载所有推荐的 Asteroid 模型"""
    models = list_available_models()
    asteroid_models = models.get('Asteroid', [])
    
    print("="*80)
    print(f"批量下载 Asteroid 模型（共 {len(asteroid_models)} 个）")
    print("="*80)
    
    success_count = 0
    for i, model_info in enumerate(asteroid_models, 1):
        print(f"\n[{i}/{len(asteroid_models)}] 下载: {model_info['name']}")
        if download_model('asteroid', model_info['name'], cache_dir, device):
            success_count += 1
        print()
    
    print("="*80)
    print(f"批量下载完成: 成功 {success_count}/{len(asteroid_models)}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='下载预训练模型工具')
    
    parser.add_argument('--list', action='store_true',
                       help='列出所有可用模型')
    parser.add_argument('--source', type=str, choices=['asteroid', 'speechbrain'],
                       help='模型来源: asteroid 或 speechbrain')
    parser.add_argument('--name', type=str,
                       help='模型名称（例如: mpariente/ConvTasNet_WHAM!_sepclean）')
    parser.add_argument('--cache-dir', type=str, default=None,
                       help='缓存目录（默认: pretrained_models/{source}）')
    parser.add_argument('--device', type=str, default='cpu',
                       help='设备（默认: cpu，下载时推荐使用 cpu）')
    parser.add_argument('--download-all-asteroid', action='store_true',
                       help='下载所有推荐的 Asteroid 模型')
    
    args = parser.parse_args()
    
    # 列出模型
    if args.list:
        print_available_models()
        return
    
    # 批量下载 Asteroid 模型
    if args.download_all_asteroid:
        download_all_asteroid_models(args.cache_dir, args.device)
        return
    
    # 下载单个模型
    if args.source and args.name:
        success = download_model(args.source, args.name, args.cache_dir, args.device)
        sys.exit(0 if success else 1)
    
    # 如果没有指定参数，显示帮助
    parser.print_help()
    print("\n" + "="*80)
    print("示例用法:")
    print("="*80)
    print("\n1. 列出所有可用模型:")
    print("   python scripts/download_pretrained_models.py --list")
    print("\n2. 下载单个模型:")
    print("   python scripts/download_pretrained_models.py \\")
    print("     --source asteroid \\")
    print("     --name mpariente/ConvTasNet_WHAM!_sepclean")
    print("\n3. 批量下载所有 Asteroid 模型:")
    print("   python scripts/download_pretrained_models.py --download-all-asteroid")
    print("\n4. 指定缓存目录:")
    print("   python scripts/download_pretrained_models.py \\")
    print("     --source asteroid \\")
    print("     --name mpariente/ConvTasNet_WHAM!_sepclean \\")
    print("     --cache-dir ./my_models")
    print()


if __name__ == "__main__":
    main()

