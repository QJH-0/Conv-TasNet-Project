"""
PyTorch DataLoader 实现
语音分离数据集加载（支持动态切片和wsj0-2mix格式）

核心设计思想：
    - 原始语音长度各不相同，采用固定 chunk_size 以利于 batch 训练
    - 动态切片：运行时将长音频切成固定长度 chunk，避免预处理
    - 随机起点：训练模式下引入数据增广（不同 epoch 不同起点）
    - 删除二次处理：不在数据加载阶段做归一化、动态混合等操作
    - 支持 wsj0-2mix 格式的目录结构和CSV元数据
"""

import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import torch.nn.functional as F
import csv
import random
from tqdm import tqdm


class SeparationDataset(Dataset):
    """
    语音分离数据集（wsj0-2mix格式，原始音频加载，无二次处理）
    
    设计原则：
        - 只负责加载原始音频数据
        - 不做归一化、裁剪、混合等二次处理
        - 返回完整的原始音频，切片由 Spliter 完成
        - 只支持 wsj0-2mix 标准格式
    """
    
    def __init__(self, data_dir, sample_rate=16000):
        """
        Args:
            data_dir: 数据目录路径（wsj0-2mix标准格式）:
                     示例: data/wav8k/train (或 dev, test)
            sample_rate: 采样率
        
        wsj0-2mix目录结构:
            wav8k/
            ├── train/
            ├── dev/
            ├── test/
            └── metadata/
                ├── train_metadata.csv
                ├── dev_metadata.csv
                └── test_metadata.csv
        """
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.metadata = []
        
        # 加载 wsj0-2mix 格式元数据
        print(f"加载 wsj0-2mix 格式数据集...")
        self._load_metadata_csv()
        print(f"[OK] 加载了 {len(self.metadata)} 个样本")
    
    def _load_metadata_csv(self):
        """
        从CSV文件加载元数据（wsj0-2mix标准格式）
        
        支持的split名称: train, dev, test
        目录结构: wav8k/train/, wav8k/dev/, wav8k/test/
        元数据位置: wav8k/metadata/{split}_metadata.csv
        """
        # 获取split名称 (train/dev/test)
        split_name = os.path.basename(self.data_dir)
        
        # 验证split名称
        if split_name not in ['train', 'dev', 'test']:
            raise ValueError(
                f"不支持的split名称: {split_name}。"
                f"wsj0-2mix标准格式只支持: train, dev, test"
            )
        
        # 元数据文件路径
        # data_dir 示例: data/wav8k/train
        # 元数据路径: data/wav8k/metadata/train_metadata.csv
        base_dir = os.path.dirname(self.data_dir)  # data/wav8k
        metadata_dir = os.path.join(base_dir, 'metadata')
        metadata_file = os.path.join(metadata_dir, f'{split_name}_metadata.csv')
        
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"元数据文件不存在: {metadata_file}")
        
        print(f"加载元数据: {metadata_file}")
        
        # 读取CSV（wsj0-2mix/Libri2Mix 实际格式）
        # 列名: id, mix_wav:FILE, s1_wav:FILE, s2_wav:FILE, length
        # 注意：列名使用冒号 : 而不是点 .
        with open(metadata_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 获取相对路径（可能不含.wav扩展名）
                mix_path = row['mix_wav:FILE']
                s1_path = row['s1_wav:FILE']
                s2_path = row['s2_wav:FILE']
                
                # 确保路径以.wav结尾
                if not mix_path.endswith('.wav'):
                    mix_path = mix_path + '.wav'
                if not s1_path.endswith('.wav'):
                    s1_path = s1_path + '.wav'
                if not s2_path.endswith('.wav'):
                    s2_path = s2_path + '.wav'
                
                # 转换为绝对路径
                mixture_path = os.path.join(base_dir, mix_path)
                source_1_path = os.path.join(base_dir, s1_path)
                source_2_path = os.path.join(base_dir, s2_path)
                
                self.metadata.append({
                    'mixture_ID': row['id'],  # 注意：列名是小写 id
                    'mixture_path': mixture_path,
                    'source_1_path': source_1_path,
                    'source_2_path': source_2_path,
                    'length': int(float(row['length']))  # 先转float再转int，处理 "28800.0" 格式
                })
    
    def __len__(self):
        return len(self.metadata)
    
    def _load_audio(self, audio_path):
        """
        加载单个音频文件（混合和源音频使用相同的处理方法）
        
        处理流程：
            1. 加载音频
            2. 转单声道（如果是多声道）
            3. 重采样到目标采样率（如果需要）
        
        Args:
            audio_path: 音频文件路径
        
        Returns:
            audio: Tensor [T] - 处理后的音频
        """
        # 加载音频
        audio, sr = torchaudio.load(audio_path)
        
        # 确保单声道
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        audio = audio.squeeze(0)  # [T]
        
        # 重采样（如果需要）
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)
        
        return audio
    
    def __getitem__(self, idx):
        """
        加载单个样本（原始音频，不做任何二次处理）
        
        Returns:
            dict: {
                'mix': Tensor [T] - 混合音频
                'ref': list[Tensor [T]] - 源音频列表
            }
        """
        # 从元数据读取路径
        meta = self.metadata[idx]
        mixture_path = meta['mixture_path']
        s1_path = meta['source_1_path']
        s2_path = meta['source_2_path']
        
        # 加载音频（混合和源音频使用相同的处理方法）
        mixture = self._load_audio(mixture_path)
        sources = [self._load_audio(s1_path), self._load_audio(s2_path)]
        
        # 返回字典格式，与参考代码一致
        return {
            'mix': mixture,
            'ref': sources
        }
    


class Spliter:
    """
    语音切片器：将一条样本拆分为多个 chunk
    
    参数：
        chunk_size: 每个切片长度 (采样点)
        is_train: 训练模式下随机起点，否则从 0 开始
        least: 小于此长度的样本直接忽略 (默认 chunk_size//2)
        use_all_chunks: 是否使用全部切片，False则每个样本只返回一个chunk (默认True)
    """
    
    def __init__(self, chunk_size=32000, is_train=True, least=None, use_all_chunks=True):
        super(Spliter, self).__init__()
        self.chunk_size = chunk_size
        self.is_train = is_train
        self.least = least if least is not None else chunk_size // 2
        self.use_all_chunks = use_all_chunks
    
    def count_chunks(self, audio_length):
        """
        计算给定音频长度能产生多少个chunk（不实际加载音频）
        
        Args:
            audio_length: 音频长度（采样点数）
        
        Returns:
            chunk数量
        """
        if audio_length < self.least:
            return 0
        
        if audio_length < self.chunk_size:
            # 短音频零填充，产生1个chunk
            return 1
        
        if not self.use_all_chunks:
            # 只使用一个chunk
            return 1
        else:
            # 计算能产生多少个chunk
            # 从0开始，每次步进least，直到 start + chunk_size > length
            num_chunks = 0
            start = 0
            while start + self.chunk_size <= audio_length:
                num_chunks += 1
                start += self.least
            return num_chunks
    
    def chunk_audio(self, sample, start):
        """
        根据起点裁剪一个 chunk
        
        Args:
            sample: {'mix': Tensor, 'ref': [Tensor,...]}
            start: 起始索引（采样点）
        Returns:
            同结构字典，长度=chunk_size
        """
        chunk = dict()
        chunk['mix'] = sample['mix'][start:start + self.chunk_size]
        chunk['ref'] = [r[start:start + self.chunk_size] for r in sample['ref']]
        return chunk
    
    def splits(self, sample):
        """
        核心切片逻辑
        
        规则：
            1. 长度 < least -> 空列表
            2. 长度 < chunk_size -> 右侧零填充到 chunk_size，仅返回 1 个 chunk
            3. 长度 >= chunk_size -> 从 random_start (train) 或 0 (eval) 开始，每次步进 least 生成 chunk
            4. use_all_chunks=False -> 只返回第一个chunk
        Returns:
            list[chunk_dict]
        """
        length = sample['mix'].shape[0]
        if length < self.least:
            return []
        
        audio_lists = []
        if length < self.chunk_size:
            # 零填充
            gap = self.chunk_size - length
            sample['mix'] = F.pad(sample['mix'], (0, gap), mode='constant')
            sample['ref'] = [F.pad(r, (0, gap), mode='constant') for r in sample['ref']]
            audio_lists.append(sample)
        else:
            # 随机起点（训练）或固定起点（测试）
            random_start = random.randint(0, length % self.least) if self.is_train else 0
            
            if not self.use_all_chunks:
                # 只返回一个chunk（从随机/固定起点开始）
                audio_lists.append(self.chunk_audio(sample, random_start))
            else:
                # 返回所有可能的chunk
                while True:
                    if random_start + self.chunk_size > length:
                        break
                    audio_lists.append(self.chunk_audio(sample, random_start))
                    random_start += self.least
        
        return audio_lists


class CustomDataLoader:
    """
    自定义迭代器封装（支持动态切片和批次聚合）
    
    流程：
        1. 内部使用 torch.utils.data.DataLoader 先按 batch_size//2 抓取原始样本
        2. 对每个原始样本调用 Spliter.splits 拆成多个 chunk
        3. 聚合所有 chunk 放入 mini_batch 列表
        4. 按 batch_size 再次整合为模型输入的标准批次
        5. 末尾不足一个 batch 的 chunk 留到下一轮与后续样本拼接
    
    优点：
        - 避免对特别长的语音一次性展开导致内存峰值
        - 支持随机切片增广
        - 输出结构直接可用于损失函数
    """
    
    def __init__(self, dataset, num_workers=4, chunk_size=32000, batch_size=1, 
                 is_train=True, use_all_chunks=True):
        super(CustomDataLoader, self).__init__()
        self.dataset = dataset
        self.num_workers = num_workers
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.is_train = is_train
        self.use_all_chunks = use_all_chunks
        self.data_loader = DataLoader(
            self.dataset,
            num_workers=self.num_workers,
            batch_size=max(1, self.batch_size // 2),  # 减少一次性切片爆炸
            shuffle=self.is_train,
            collate_fn=self._collate
        )
        self.spliter = Spliter(
            chunk_size=self.chunk_size,
            is_train=self.is_train,
            least=self.chunk_size // 2,
            use_all_chunks=self.use_all_chunks
        )
        
        # 预计算总chunk数（使用元数据，避免重复计算）
        self._total_chunks = self._calculate_total_chunks()
        self._num_batches = max(1, self._total_chunks // self.batch_size)
    
    def _calculate_total_chunks(self):
        """
        预计算总chunk数（使用元数据中的实际长度信息）
        
        Returns:
            总chunk数
        """
        total_chunks = 0
        chunk_counts = []  # 统计每个样本的chunk数
        
        for meta in self.dataset.metadata:
            audio_length = meta['length']
            num_chunks = self.spliter.count_chunks(audio_length)
            total_chunks += num_chunks
            chunk_counts.append(num_chunks)
        
        # 打印统计信息
        if chunk_counts:
            avg_chunks = sum(chunk_counts) / len(chunk_counts)
            print(f"[DataLoader统计]")
            print(f"  - 样本数: {len(self.dataset)}")
            print(f"  - 总chunk数: {total_chunks}")
            print(f"  - 平均每样本chunk数: {avg_chunks:.2f}")
            print(f"  - 最小chunk数: {min(chunk_counts)}")
            print(f"  - 最大chunk数: {max(chunk_counts)}")
            print(f"  - 批次大小: {self.batch_size}")
            print(f"  - 预计批次数: {total_chunks // self.batch_size}")
        
        return total_chunks
    
    def __len__(self):
        """
        返回批次数量（已在初始化时预计算）
        
        使用元数据中的实际长度信息精确计算，而非估算。
        """
        return self._num_batches
    
    def _collate(self, batch):
        """
        将原始 batch 中每条样本切片后展平为一个大的 chunk 列表
        
        输入：list[{'mix':Tensor,'ref':[Tensor,...]}]
        输出：list[chunk_dict,...]
        """
        batch_audio = []
        for b in batch:
            batch_audio += self.spliter.splits(b)
        return batch_audio
    
    def __iter__(self):
        """
        迭代器入口：按策略产出标准 batch
        
        细节：
            - mini_batch 累积当前尚未拼成完整 batch 的 chunk
            - 每次对已积累的 chunk 按 batch_size 切分打包
            - 训练模式可在积累后打乱 chunk 顺序
        """
        mini_batch = []
        for batch in self.data_loader:
            mini_batch += batch  # 扩展 chunk 列表
            length = len(mini_batch)
            if self.is_train:
                random.shuffle(mini_batch)
            collate_chunk = []
            for start in range(0, length - self.batch_size + 1, self.batch_size):
                b = default_collate(mini_batch[start:start + self.batch_size])
                collate_chunk.append(b)
            # 剩余不足一个 batch 的 chunk 缓存到下一轮
            idx = length % self.batch_size
            mini_batch = mini_batch[-idx:] if idx else []
            for m_batch in collate_chunk:
                # m_batch: {'mix': Tensor(B,L), 'ref': [Tensor(B,L), ...]}
                yield m_batch


def create_dataloader(data_dir, is_train=True, batch_size=16, num_workers=4, 
                     sample_rate=16000, chunk_size=32000, use_all_chunks=True):
    """
    创建自定义 DataLoader（支持动态切片，wsj0-2mix标准格式）
    
    设计思想：
        - 数据集只负责加载原始音频
        - 不做归一化、裁剪等二次处理
        - 使用 CustomDataLoader 进行动态切片
        - 训练时随机起点增强数据多样性
        - 只支持 wsj0-2mix 标准格式
    
    Args:
        data_dir: 数据目录路径（wsj0-2mix标准格式）
                 示例: data/wav8k/train (或 dev, test)
        is_train: 是否训练模式（控制shuffle和随机起点）
        batch_size: batch大小
        num_workers: 工作进程数
        sample_rate: 采样率
        chunk_size: 切片长度（采样点数）
        use_all_chunks: 是否使用所有切片。
                       True: 每个样本切成多个chunk，数据量翻倍
                       False: 每个样本只返回一个chunk（从随机/固定起点）
    
    Returns:
        CustomDataLoader对象
        
    输出格式:
        每个batch是一个字典 {
            'mix': Tensor [B, chunk_size],
            'ref': list[Tensor [B, chunk_size], ...]
        }
    
    wsj0-2mix目录结构:
        wav8k/
        ├── train/          # 训练集
        ├── dev/            # 验证集
        ├── test/           # 测试集
        └── metadata/
            ├── train_metadata.csv
            ├── dev_metadata.csv
            └── test_metadata.csv
    """
    dataset = SeparationDataset(
        data_dir=data_dir,
        sample_rate=sample_rate
    )
    
    dataloader = CustomDataLoader(
        dataset=dataset,
        num_workers=num_workers,
        chunk_size=chunk_size,
        batch_size=batch_size,
        is_train=is_train,
        use_all_chunks=use_all_chunks
    )
    
    return dataloader


def get_dataloader(data_path, batch_size, num_workers, shuffle, split, 
                   chunk_size=32000, use_all_chunks=True):
    """
    别名函数，用于兼容训练脚本中的调用
    
    Args:
        data_path: 数据根目录路径 (例如: 'data/processed1000')
        batch_size: batch大小
        num_workers: 工作进程数
        shuffle: 是否打乱数据
        split: 数据集划分 ('train', 'dev', 'test')
        chunk_size: 切片长度
        use_all_chunks: 是否使用所有切片
    
    Returns:
        CustomDataLoader对象
    """
    # 构建完整路径
    data_dir = os.path.join(data_path, split)
    
    # 调用create_dataloader
    return create_dataloader(
        data_dir=data_dir,
        is_train=shuffle,
        batch_size=batch_size,
        num_workers=num_workers,
        sample_rate=8000,  # 默认采样率
        chunk_size=chunk_size,
        use_all_chunks=use_all_chunks
    )


if __name__ == "__main__":
    # 测试代码
    print("Testing DataLoader...")
    
    # 测试 wsj0-2mix 格式数据
    # 注意：必须指向 train/dev/test 子目录，不是根目录
    data_dir = r"D:\Paper\datasets\Libri2Mix_8k\test"  # wsj0-2mix标准路径
    
    if os.path.exists(data_dir):
        print(f"数据目录: {data_dir}")
        # 创建DataLoader
        dataloader = create_dataloader(
            data_dir=data_dir,
            is_train=False,
            batch_size=4,
            num_workers=0,  # 测试时使用0
            chunk_size=32000,
            sample_rate=8000
        )
        
        print(f"DataLoader batches (estimated): {len(dataloader)}")
        
        # 测试一个batch
        batch = next(iter(dataloader))
        print(f"\nBatch keys: {batch.keys()}")
        print(f"  Mix shape: {batch['mix'].shape}")
        print(f"  Ref length: {len(batch['ref'])}")
        print(f"  Ref[0] shape: {batch['ref'][0].shape}")
        print(f"  Ref[1] shape: {batch['ref'][1].shape}")
        
        print("\n[OK] DataLoader test passed!")
        print("\n数据格式说明:")
        print("  - batch['mix']: 混合音频 [B, chunk_size]")
        print("  - batch['ref']: 源音频列表 [Tensor[B, chunk_size], ...]")
    else:
        print(f"数据目录不存在: {data_dir}")
        print("\nwsj0-2mix标准目录结构:")
        print("  wav8k/")
        print("  ├── train/")
        print("  ├── dev/")
        print("  ├── test/")
        print("  └── metadata/")
        print("      ├── train_metadata.csv")
        print("      ├── dev_metadata.csv")
        print("      └── test_metadata.csv")
