"""
预训练模型工具
用于下载、保存和加载预训练模型（Asteroid/SpeechBrain）
"""

import os
import json
import torch
from pathlib import Path
from typing import Optional, Dict, Any




def load_asteroid_model(model_name: str, device: str = 'cpu', 
                       cache_dir: Optional[str] = None,
                       save_info: bool = True):
    """从 Asteroid 加载预训练模型并保存到指定目录"""
    try:
        from asteroid.models import BaseModel
        import torch
    except ImportError:
        raise RuntimeError("Asteroid 库未安装！请安装: pip install asteroid")
    
    # 设置保存目录
    if cache_dir is None:
        cache_dir = "pretrained_models/asteroid"
    
    # 清理模型名称作为子目录
    safe_name = model_name.replace('/', '_').replace('!', '_')
    model_dir = os.path.join(cache_dir, safe_name)
    os.makedirs(model_dir, exist_ok=True)
    
    model_file = os.path.join(model_dir, f"{safe_name}.pth")
    
    print(f"正在加载模型: {model_name}")
    print(f"保存目录: {model_dir}")
    
    # 加载模型（从系统缓存）
    model = BaseModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # 保存完整模型到项目目录
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_name': model_name,
        'model_class': type(model).__name__,
        'total_parameters': total_params,
    }
    torch.save(checkpoint, model_file)
    file_size = os.path.getsize(model_file) / 1024 / 1024
    print(f"模型已保存: {model_file} ({file_size:.2f} MB)")
    
    # 保存模型信息
    info = None
    if save_info:
        info = {
            "model_name": model_name,
            "source": "asteroid",
            "model_dir": model_dir,
            "model_file": model_file,
            "total_parameters": total_params,
            "total_parameters_M": round(total_params / 1e6, 2),
            "model_class": type(model).__name__,
            "file_size_MB": round(file_size, 2),
        }
        info_file = os.path.join(model_dir, "model_info.json")
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=4, ensure_ascii=False)
        print(f"信息已保存: {info_file}")
    
    return model, info


def load_speechbrain_model(model_name: str, device: str = 'cpu',
                          cache_dir: Optional[str] = None,
                          save_info: bool = True):
    """从 SpeechBrain 加载预训练模型"""
    try:
        try:
            from speechbrain.inference.separation import SepformerSeparation
        except ImportError:
            from speechbrain.pretrained import SepformerSeparation
    except ImportError:
        raise RuntimeError("SpeechBrain 库未安装！请安装: pip install speechbrain")
    
    # 设置保存目录
    if cache_dir is None:
        cache_dir = "pretrained_models/speechbrain"
    
    # 清理模型名称作为子目录
    safe_name = model_name.replace('/', '_')
    model_dir = os.path.join(cache_dir, safe_name)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"正在加载模型: {model_name}")
    print(f"保存目录: {model_dir}")
    
    # 加载模型
    separator = SepformerSeparation.from_hparams(
        source=model_name,
        savedir=model_dir,
        run_opts={"device": str(device)}
    )
    
    print(f"模型已加载: {type(separator).__name__}")
    
    # 保存模型信息
    info = None
    if save_info:
        info = {
            "model_name": model_name,
            "source": "speechbrain",
            "model_dir": model_dir,
            "model_class": type(separator).__name__,
        }
        info_file = os.path.join(model_dir, "model_info.json")
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=4, ensure_ascii=False)
        print(f"信息已保存: {info_file}")
    
    return separator, info


def list_available_models():
    """列出可用的预训练模型"""
    models = {
        "Asteroid": [
            {
                "name": "mpariente/ConvTasNet_WHAM!_sepclean",
                "description": "Conv-TasNet 在 WHAM! 数据集上训练（干净语音分离）",
                "sample_rate": "8kHz",
                "num_speakers": 2,
                "params": "~5.1M"
            },
            {
                "name": "mpariente/ConvTasNet_Libri2Mix_sepclean_16k",
                "description": "Conv-TasNet 在 Libri2Mix 数据集上训练",
                "sample_rate": "16kHz",
                "num_speakers": 2,
                "params": "~5.1M"
            },
            {
                "name": "JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k",
                "description": "Conv-TasNet 在 Libri2Mix 数据集上训练（含噪声）",
                "sample_rate": "16kHz",
                "num_speakers": 2,
                "params": "~5.1M"
            }
        ],
        "SpeechBrain": [
            {
                "name": "speechbrain/sepformer-wsj02mix",
                "description": "Sepformer 在 WSJ0-2Mix 数据集上训练",
                "sample_rate": "8kHz",
                "num_speakers": 2,
                "params": "~26M"
            },
            {
                "name": "speechbrain/sepformer-wham",
                "description": "Sepformer 在 WHAM! 数据集上训练",
                "sample_rate": "8kHz",
                "num_speakers": 2,
                "params": "~26M"
            },
            {
                "name": "speechbrain/sepformer-libri2mix",
                "description": "Sepformer 在 Libri2Mix 数据集上训练",
                "sample_rate": "16kHz",
                "num_speakers": 2,
                "params": "~26M"
            }
        ]
    }
    
    return models


def print_available_models():
    """打印可用的预训练模型列表"""
    models = list_available_models()
    
    print("="*80)
    print("可用的预训练模型")
    print("="*80)
    
    for source, model_list in models.items():
        print(f"\n【{source}】")
        for i, model in enumerate(model_list, 1):
            print(f"\n{i}. {model['name']}")
            print(f"   描述: {model['description']}")
            print(f"   采样率: {model['sample_rate']}")
            print(f"   说话人数: {model['num_speakers']}")
            print(f"   参数量: {model['params']}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    # 示例：打印可用模型
    print_available_models()

