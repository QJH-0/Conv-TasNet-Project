"""
PyTorch DataLoader 实现
语音分离数据集加载（支持数据缓存）
"""

import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from tqdm import tqdm


class SeparationDataset(Dataset):
    """
    语音分离数据集（支持数据缓存）
    
    加载混合音频和对应的干净音频
    可选择预加载所有数据到内存，大幅提升训练速度
    """
    
    def __init__(self, data_dir, sample_rate=16000, segment_length=64000, 
                 use_cache=True, cache_file=None, normalize=False, 
                 target_level=-25.0, augmentation=False, dynamic_mixing=False):
        """
        Args:
            data_dir: 数据目录路径（train或test）
            sample_rate: 采样率
            segment_length: 音频片段长度（samples）
            use_cache: 是否使用缓存
            cache_file: 缓存文件路径（默认为data_dir下的dataset_cache.pkl）
            normalize: 是否归一化音频（默认False，数据生成时已归一化）
            target_level: 目标dB级别（默认-25.0）
            augmentation: 是否使用数据增强-随机裁剪（默认False，数据已固定长度）
            dynamic_mixing: 是否动态混合-随机调整SNR（默认False，会破坏精确SNR控制）
            
        注意:
            - 数据生成时已经归一化到-25dB，训练时通常不需要再归一化
            - 数据生成时已经固定长度，随机裁剪通常无效
            - 数据生成时已精确控制SNR，动态混合会破坏SNR控制
            - 推荐配置: normalize=False, augmentation=False, dynamic_mixing=False
        """
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.use_cache = use_cache
        self.normalize = normalize
        self.target_level = target_level
        self.augmentation = augmentation
        self.dynamic_mixing = dynamic_mixing
        
        # 混合音频和干净音频路径
        self.mixture_dir = os.path.join(data_dir, 'mixture')
        self.clean_dir = os.path.join(data_dir, 'clean')
        
        # 缓存文件路径
        if cache_file is None:
            cache_file = os.path.join(data_dir, 'dataset_cache.pkl')
        self.cache_file = cache_file
        
        # 获取所有混合音频文件
        self.mixture_files = sorted([
            f for f in os.listdir(self.mixture_dir) 
            if f.endswith('.wav')
        ])
        
        # 初始化缓存
        self.cached_data = None
        
        # 尝试加载缓存
        if use_cache and os.path.exists(cache_file):
            print(f"Loading cached dataset from {cache_file}...")
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.cached_data = cache_data['data']
                    cached_files = cache_data['files']
                    
                # 验证缓存是否匹配当前文件列表
                if cached_files == self.mixture_files:
                    print(f"✓ Loaded {len(self.cached_data)} cached samples")
                else:
                    print("⚠ Cache file mismatch, will rebuild cache")
                    self.cached_data = None
            except Exception as e:
                print(f"⚠ Failed to load cache: {e}, will rebuild")
                self.cached_data = None
        
        # 如果需要缓存但缓存不存在，则预加载数据
        if use_cache and self.cached_data is None:
            self._preload_data()
        
        print(f"Found {len(self.mixture_files)} samples in {data_dir}")
    
    def __len__(self):
        return len(self.mixture_files)
    
    def _preload_data(self):
        """预加载所有数据到内存并保存缓存"""
        print("Preloading dataset to memory...")
        self.cached_data = []
        
        for idx in tqdm(range(len(self.mixture_files)), desc="Loading audio"):
            mixture, sources = self._load_sample(idx)
            self.cached_data.append({
                'mixture': mixture,
                'sources': sources
            })
        
        # 保存缓存到磁盘
        print(f"Saving cache to {self.cache_file}...")
        try:
            cache_data = {
                'data': self.cached_data,
                'files': self.mixture_files
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print("✓ Cache saved successfully!")
        except Exception as e:
            print(f"⚠ Failed to save cache: {e}")
    
    def _load_sample(self, idx):
        """
        加载单个样本（原始逻辑）
        
        Returns:
            mixture: [T] - 混合音频
            sources: [C, T] - C个说话人的干净音频
        """
        # 加载混合音频
        mixture_file = self.mixture_files[idx]
        mixture_path = os.path.join(self.mixture_dir, mixture_file)
        mixture, sr = torchaudio.load(mixture_path)
        
        # 确保单声道
        if mixture.shape[0] > 1:
            mixture = torch.mean(mixture, dim=0, keepdim=True)
        mixture = mixture.squeeze(0)  # [T]
        
        # 重采样（如果需要）
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            mixture = resampler(mixture)
        
        # 加载干净音频（假设有s1和s2两个文件）
        base_name = os.path.splitext(mixture_file)[0]
        s1_path = os.path.join(self.clean_dir, f"{base_name}_s1.wav")
        s2_path = os.path.join(self.clean_dir, f"{base_name}_s2.wav")
        
        sources = []
        for source_path in [s1_path, s2_path]:
            if os.path.exists(source_path):
                source, sr = torchaudio.load(source_path)
                if source.shape[0] > 1:
                    source = torch.mean(source, dim=0, keepdim=True)
                source = source.squeeze(0)
                
                # 重采样
                if sr != self.sample_rate:
                    resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                    source = resampler(source)
                
                sources.append(source)
            else:
                # 如果文件不存在，创建零信号
                sources.append(torch.zeros_like(mixture))
        
        sources = torch.stack(sources)  # [C, T]
        
        # 裁剪或填充到固定长度
        mixture = self._fix_length(mixture, self.segment_length)
        sources = torch.stack([
            self._fix_length(s, self.segment_length) 
            for s in sources
        ])
        
        return mixture, sources
    
    def _normalize_audio(self, audio):
        """
        归一化音频到目标dB级别
        
        Args:
            audio: [T] - 音频信号
        
        Returns:
            normalized: [T] - 归一化后的音频
        """
        # 计算当前RMS能量
        rms = torch.sqrt(torch.mean(audio ** 2))
        
        # 计算缩放因子
        target_rms = 10 ** (self.target_level / 20)
        scale = target_rms / (rms + 1e-8)
        
        # 归一化
        normalized = audio * scale
        
        # 防止削波
        normalized = torch.clamp(normalized, -1.0, 1.0)
        
        return normalized
    
    def __getitem__(self, idx):
        """
        获取一个样本
        
        优化说明:
        - 数据生成时已经完成归一化、混合、长度调整
        - 训练时默认直接使用，避免重复处理
        - 可选的增强功能通常应该关闭，以保持数据一致性
        
        Returns:
            mixture: [T] - 混合音频
            sources: [C, T] - C个说话人的干净音频
        """
        if self.cached_data is not None:
            # 从缓存读取（快速）
            data = self.cached_data[idx]
            mixture = data['mixture'].clone()
            sources = data['sources'].clone()
        else:
            # 动态加载（慢速）
            mixture, sources = self._load_sample(idx)
        
        # ⚠️ 可选增强1: 随机裁剪
        # 注意：只有当数据长度>segment_length时才有效
        # 推荐：数据生成时生成更长音频(如6s)，训练时裁剪到需要的长度(如4s)
        if self.augmentation and mixture.shape[0] > self.segment_length:
            start = torch.randint(0, mixture.shape[0] - self.segment_length + 1, (1,)).item()
            mixture = mixture[start:start + self.segment_length]
            sources = sources[:, start:start + self.segment_length]
        
        # ⚠️ 可选增强2: 动态混合（不推荐）
        # 警告：这会破坏数据生成时精确控制的SNR，通常应该关闭
        # 如需SNR多样性，建议在数据生成时创建多个SNR版本
        if self.dynamic_mixing and torch.rand(1).item() < 0.5:
            snr_db = torch.FloatTensor(1).uniform_(-3, 3).item()
            scale = 10 ** (snr_db / 20)
            sources_scaled = sources.clone()
            sources_scaled[0] = sources_scaled[0] * scale
            mixture = sources_scaled.sum(dim=0)
        
        # ⚠️ 可选增强3: 归一化（通常不需要）
        # 注意：数据生成时已经归一化到-25dB，通常不需要再次归一化
        # 只有当数据未归一化或需要不同的归一化级别时才启用
        if self.normalize:
            mixture = self._normalize_audio(mixture)
            sources = torch.stack([self._normalize_audio(s) for s in sources])
        
        return mixture, sources
    
    def _fix_length(self, audio, target_length):
        """
        裁剪或填充音频到目标长度
        
        Args:
            audio: [T] - 音频信号
            target_length: 目标长度
        
        Returns:
            audio: [target_length] - 固定长度的音频
        """
        current_length = audio.shape[0]
        
        if current_length > target_length:
            # 随机裁剪
            start = torch.randint(0, current_length - target_length + 1, (1,)).item()
            audio = audio[start:start + target_length]
        elif current_length < target_length:
            # 填充零
            padding = target_length - current_length
            audio = torch.nn.functional.pad(audio, (0, padding))
        
        return audio


def collate_fn(batch):
    """
    自定义collate函数
    
    Args:
        batch: List of (mixture, sources)
    
    Returns:
        mixtures: [B, T]
        sources: [B, C, T]
    """
    mixtures = []
    sources = []
    
    for mixture, source in batch:
        mixtures.append(mixture)
        sources.append(source)
    
    mixtures = torch.stack(mixtures)  # [B, T]
    sources = torch.stack(sources)    # [B, C, T]
    
    return mixtures, sources


def create_dataloader(data_dir, batch_size=4, num_workers=4, 
                     sample_rate=16000, segment_length=64000, 
                     shuffle=True, pin_memory=True, use_cache=True,
                     normalize=False, target_level=-25.0, 
                     augmentation=False, dynamic_mixing=False):
    """
    创建 DataLoader（支持数据缓存、归一化和数据增强）
    
    优化说明:
    - normalize默认False（数据生成时已归一化）
    - augmentation默认False（数据已固定长度）
    - dynamic_mixing默认False（会破坏SNR控制）
    
    推荐配置:
    - 训练集: normalize=False, augmentation=False, dynamic_mixing=False
    - 验证集: normalize=False, augmentation=False, dynamic_mixing=False
    
    特殊情况:
    - 如果数据生成时使用了更长的音频长度，可以启用augmentation做随机裁剪
    - 如果数据未归一化，需要启用normalize
    - 通常不建议启用dynamic_mixing
    
    Args:
        data_dir: 数据目录路径
        batch_size: batch大小
        num_workers: 工作进程数
        sample_rate: 采样率
        segment_length: 音频片段长度
        shuffle: 是否打乱
        pin_memory: 是否固定内存
        use_cache: 是否使用数据缓存
        normalize: 是否归一化音频（默认False）
        target_level: 目标dB级别（默认-25.0）
        augmentation: 是否使用数据增强（默认False）
        dynamic_mixing: 是否使用动态混合（默认False，不推荐）
    
    Returns:
        dataloader: DataLoader对象
    """
    dataset = SeparationDataset(
        data_dir=data_dir,
        sample_rate=sample_rate,
        segment_length=segment_length,
        use_cache=use_cache,
        normalize=normalize,
        target_level=target_level,
        augmentation=augmentation,
        dynamic_mixing=dynamic_mixing
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=True  # 丢弃最后不完整的batch
    )
    
    return dataloader


if __name__ == "__main__":
    # 测试代码
    print("Testing DataLoader...")
    
    # 创建测试数据（需要先运行数据生成脚本）
    data_dir = "data/processed/mixed/train"
    
    if os.path.exists(data_dir):
        # 创建数据集
        dataset = SeparationDataset(
            data_dir=data_dir,
            sample_rate=16000,
            segment_length=64000
        )
        
        print(f"\nDataset size: {len(dataset)}")
        
        # 测试单个样本
        if len(dataset) > 0:
            mixture, sources = dataset[0]
            print(f"\nSample 0:")
            print(f"  Mixture shape: {mixture.shape}")
            print(f"  Sources shape: {sources.shape}")
            
            # 创建DataLoader
            dataloader = create_dataloader(
                data_dir=data_dir,
                batch_size=2,
                num_workers=0,  # 测试时使用0
                shuffle=False
            )
            
            print(f"\nDataLoader batches: {len(dataloader)}")
            
            # 测试一个batch
            mixtures, sources = next(iter(dataloader))
            print(f"\nBatch:")
            print(f"  Mixtures shape: {mixtures.shape}")
            print(f"  Sources shape: {sources.shape}")
            
            print("\nDataLoader test passed!")
        else:
            print("No data found. Please run data generation scripts first.")
    else:
        print(f"Data directory not found: {data_dir}")
        print("Please run data generation scripts first.")
