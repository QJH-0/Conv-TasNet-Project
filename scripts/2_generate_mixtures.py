"""
数据生成脚本
从AISHELL-3数据集生成混合语音数据
按照 wsj0-2mix 格式组织文件结构和元数据
"""

import os
import sys
import argparse
import yaml
import torch
import torchaudio
import random
import csv
import glob
from tqdm import tqdm

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
# 切换到项目根目录,确保所有相对路径正确
os.chdir(project_root)

from utils.audio_utils import mix_audio_with_snr, normalize_mixture


def load_config(config_path):
    """加载配置文件"""
    # 如果是相对路径，转换为相对于项目根目录的绝对路径
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)
    
    print(f"加载配置文件: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 规范化配置中的所有路径为基于项目根目录的绝对路径
    if 'dataset' in config:
        if 'processed_data_path' in config['dataset'] and not os.path.isabs(config['dataset']['processed_data_path']):
            config['dataset']['processed_data_path'] = os.path.join(
                project_root, config['dataset']['processed_data_path']
            )
    
    return config


def scan_aishell3_dataset(aishell3_path, num_speakers=20, num_utterances_per_speaker=50):
    """
    扫描AISHELL-3数据集，选择说话人和语音文件
    
    Args:
        aishell3_path: AISHELL-3数据集根目录
        num_speakers: 需要选择的说话人数量
        num_utterances_per_speaker: 每个说话人选择的语音数量
    
    Returns:
        speaker_utterances: {speaker_id: [audio_paths]}
    """
    train_wav_dir = os.path.join(aishell3_path, 'train', 'wav')
    
    # 获取所有说话人文件夹
    speaker_dirs = sorted(glob.glob(os.path.join(train_wav_dir, 'SSB*')))
    print(f"找到 {len(speaker_dirs)} 个说话人文件夹")
    
    # 随机选择指定数量的说话人
    if len(speaker_dirs) > num_speakers:
        selected_speakers = random.sample(speaker_dirs, num_speakers)
    else:
        selected_speakers = speaker_dirs
        print(f"警告: 可用说话人少于 {num_speakers}，使用全部 {len(speaker_dirs)} 个说话人")
    
    speaker_utterances = {}
    
    for speaker_dir in selected_speakers:
        speaker_id = os.path.basename(speaker_dir)
        
        # 获取该说话人的所有wav文件
        audio_files = sorted(glob.glob(os.path.join(speaker_dir, '*.wav')))
        
        # 随机选择指定数量的语音
        if len(audio_files) > num_utterances_per_speaker:
            selected_files = random.sample(audio_files, num_utterances_per_speaker)
        else:
            selected_files = audio_files
            print(f"警告: 说话人 {speaker_id} 的语音少于 {num_utterances_per_speaker}，使用全部 {len(audio_files)} 个")
        
        speaker_utterances[speaker_id] = selected_files
    
    return speaker_utterances


def pad_or_trim_audio(audio, target_length, fade_samples=400):
    """
    填充或截取音频到目标长度
    
    改进：添加淡入淡出避免重复时的不自然跳变
    
    Args:
        audio: [T] - 音频信号
        target_length: 目标长度
        fade_samples: 淡入淡出的样本数（用于重复边界）
    
    Returns:
        audio: [target_length] - 处理后的音频
    """
    current_length = audio.shape[0]
    
    if current_length > target_length:
        # 截取：随机选择一个起始位置
        start_idx = random.randint(0, current_length - target_length)
        audio = audio[start_idx:start_idx + target_length]
        
    elif current_length < target_length:
        # 填充：重复音频，在边界添加淡入淡出
        num_repeats = (target_length // current_length) + 1
        
        # 重复音频
        repeated = audio.repeat(num_repeats)
        
        # 在每个重复边界添加淡入淡出（如果音频足够长）
        if current_length >= fade_samples * 2:
            for i in range(1, num_repeats):
                boundary = i * current_length
                if boundary < len(repeated) - fade_samples:
                    # 创建淡入淡出曲线
                    fade_out = torch.linspace(1.0, 0.0, fade_samples)
                    fade_in = torch.linspace(0.0, 1.0, fade_samples)
                    
                    # 应用淡出到前一段的结尾
                    start = boundary - fade_samples
                    repeated[start:boundary] *= fade_out
                    
                    # 应用淡入到当前段的开头
                    repeated[boundary:boundary + fade_samples] *= fade_in
        
        # 截取到目标长度
        audio = repeated[:target_length]
    
    return audio


def load_and_resample_audio(audio_path, target_sr, target_length):
    """
    加载音频并重采样到目标采样率和长度
    
    注意：此函数不进行归一化，归一化将在混合后统一进行以保持SNR
    
    Args:
        audio_path: 音频文件路径
        target_sr: 目标采样率
        target_length: 目标长度（样本点数）
    
    Returns:
        audio: [target_length] - 处理后的音频（未归一化）
    """
    # 加载音频
    audio, sr = torchaudio.load(audio_path)
    
    # 如果是多声道，取第一个声道
    if audio.shape[0] > 1:
        audio = audio[0]
    else:
        audio = audio.squeeze(0)
    
    # 重采样
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        audio = resampler(audio)
    
    # 调整长度
    audio = pad_or_trim_audio(audio, target_length)
    
    # ✅ 不在这里归一化，将在混合后统一归一化以保持SNR
    
    return audio


def generate_mixture_dataset(config, speaker_utterances, num_samples=100, split='train'):
    """
    生成混合语音数据集（wsj0-2mix格式）
    
    Args:
        config: 配置字典
        speaker_utterances: {speaker_id: [audio_paths]}
        num_samples: 样本数量
        split: 'train' (训练), 'dev' (验证) 或 'test' (测试)
    """
    dataset_config = config['dataset']
    sample_rate = dataset_config['sample_rate']
    
    # 支持固定长度或长度范围
    if 'audio_length_range' in dataset_config:
        audio_length_range = dataset_config['audio_length_range']
        print(f"使用随机音频长度范围: {audio_length_range[0]:.1f}s - {audio_length_range[1]:.1f}s")
    else:
        # 兼容旧配置（固定长度）
        audio_length = dataset_config.get('audio_length', 4.0)
        audio_length_range = [audio_length, audio_length]
        print(f"使用固定音频长度: {audio_length:.1f}s")
    
    snr_range = dataset_config['snr_range']
    
    # 按 wsj0-2mix 格式组织输出目录
    # 新结构: wav8k/train, wav8k/dev, wav8k/test
    # sr_folder = f"wav{sample_rate//1000}k"
    # output_base = os.path.join(dataset_config['processed_data_path'], sr_folder)
    # 新结构: /train, /dev, /test
    # 每个split下包含 mix_clean, s1, s2 三个子目录
    split_dir = os.path.join(dataset_config['processed_data_path'], split)
    mix_dir = os.path.join(split_dir, 'mix_clean')
    s1_dir = os.path.join(split_dir, 's1')
    s2_dir = os.path.join(split_dir, 's2')
    
    # 元数据目录在 /metadata 下
    metadata_dir = os.path.join(dataset_config['processed_data_path'], 'metadata')
    
    # 创建所有必要的目录
    os.makedirs(mix_dir, exist_ok=True)
    os.makedirs(s1_dir, exist_ok=True)
    os.makedirs(s2_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)
    
    # 准备说话人列表
    speakers = list(speaker_utterances.keys())
    
    print(f"\nGenerating {num_samples} {split} samples...")
    print(f"Output directory: {split_dir}")
    print(f"Available speakers: {len(speakers)}")
    
    # 保存CSV元数据
    metadata = []
    
    for i in tqdm(range(num_samples)):
        try:
            # 随机选择两个不同的说话人
            speaker1, speaker2 = random.sample(speakers, 2)
            
            # 随机选择每个说话人的一个语音文件
            audio_path1 = random.choice(speaker_utterances[speaker1])
            audio_path2 = random.choice(speaker_utterances[speaker2])
            
            # 为每个样本随机选择长度
            audio_length = random.uniform(audio_length_range[0], audio_length_range[1])
            target_length = int(sample_rate * audio_length)
            
            # 加载并处理音频（不归一化）
            audio1 = load_and_resample_audio(audio_path1, sample_rate, target_length)
            audio2 = load_and_resample_audio(audio_path2, sample_rate, target_length)
            
            # 随机 SNR
            snr_db = random.uniform(snr_range[0], snr_range[1])
            
            # ✅ 使用新的混合函数（保持SNR）
            mixture, audio1_scaled, audio2_scaled = mix_audio_with_snr(
                audio1, audio2, snr_db
            )
            
            # ✅ 统一归一化（保持SNR不变）
            sources = torch.stack([audio1_scaled, audio2_scaled])
            mixture_norm, sources_norm = normalize_mixture(
                mixture, sources, target_level=-25.0
            )
            
            # 验证混合是否正确（mixture应该等于sources之和）
            reconstructed = sources_norm.sum(dim=0)
            reconstruction_error = torch.mean((mixture_norm - reconstructed) ** 2).item()
            
            # 计算实际SNR
            energy1 = torch.sum(sources_norm[0] ** 2).item()
            energy2 = torch.sum(sources_norm[1] ** 2).item()
            actual_snr = 10 * torch.log10(torch.tensor(energy1 / (energy2 + 1e-8))).item()
            
            # 按 wsj0-2mix 格式保存文件
            # 生成唯一ID: speaker1-speaker2-索引
            mix_id = f"{speaker1}-{speaker2}-{i:05d}"
            
            # 文件命名
            filename = f"{mix_id}.wav"
            
            # 保存混合音频到 mix_clean 目录
            mixture_path = os.path.join(mix_dir, filename)
            torchaudio.save(mixture_path, mixture_norm.unsqueeze(0), sample_rate)
            
            # 保存源音频1和2到对应的 s1, s2 目录
            s1_path = os.path.join(s1_dir, filename)
            s2_path = os.path.join(s2_dir, filename)
            torchaudio.save(s1_path, sources_norm[0].unsqueeze(0), sample_rate)
            torchaudio.save(s2_path, sources_norm[1].unsqueeze(0), sample_rate)
            
            # 记录CSV元数据 - 使用相对于 wav8k 目录的路径
            # 格式: id, mix_wav.FILE, s1_wav.FILE, s2_wav.FILE, length
            metadata.append({
                'id': mix_id,
                'mix_wav:FILE': f"{split}/mix_clean/{filename}",
                's1_wav:FILE': f"{split}/s1/{filename}",
                's2_wav:FILE': f"{split}/s2/{filename}",
                'length': target_length
            })
            
        except Exception as e:
            print(f"\n警告: 生成样本 {i} 时出错: {e}")
            continue
    
    # 保存CSV元数据（wsj0-2mix格式）
    metadata_path = os.path.join(metadata_dir, f'{split}_metadata.csv')
    with open(metadata_path, 'w', newline='', encoding='utf-8') as f:
        if metadata:
            fieldnames = ['id', 'mix_wav:FILE', 's1_wav:FILE', 's2_wav:FILE', 'length']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metadata)
    
    print(f"✓ Generated {len(metadata)} samples in {split_dir}")
    print(f"✓ Metadata saved to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate mixture dataset from AISHELL-3')
    parser.add_argument('--config', type=str, default='config/config.yml',
                       help='Path to config file')
    
    # 路径参数（可覆盖config）
    parser.add_argument('--input-dir', type=str, default=None,
                       help='AISHELL-3 input directory (override config)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (override config)')
    
    # 数据集参数（可覆盖config）
    parser.add_argument('--num-speakers', type=int, default=None,
                       help='Number of speakers to use (override config)')
    parser.add_argument('--num-utterances', type=int, default=None,
                       help='Number of utterances per speaker (override config)')
    parser.add_argument('--num-train', type=int, default=None,
                       help='Number of training samples (override config)')
    parser.add_argument('--num-test', type=int, default=None,
                       help='Number of test samples (override config)')
    
    # 音频参数（可覆盖config）
    parser.add_argument('--sample-rate', type=int, default=None,
                       help='Sample rate (override config)')
    parser.add_argument('--audio-length', type=float, default=None,
                       help='Fixed audio length in seconds (override config, will set both min and max to this value)')
    parser.add_argument('--audio-length-min', type=float, default=None,
                       help='Minimum audio length in seconds (override config)')
    parser.add_argument('--audio-length-max', type=float, default=None,
                       help='Maximum audio length in seconds (override config)')
    parser.add_argument('--snr-min', type=float, default=None,
                       help='Minimum SNR in dB (override config)')
    parser.add_argument('--snr-max', type=float, default=None,
                       help='Maximum SNR in dB (override config)')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed (override config)')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 从配置文件读取参数,命令行参数可以覆盖
    dataset_config = config['dataset']
    
    # 路径参数
    input_dir = args.input_dir if args.input_dir is not None else dataset_config['raw_data_path']
    output_dir = args.output_dir if args.output_dir is not None else dataset_config['processed_data_path']
    
    # 覆盖配置中的路径（让后续代码使用命令行参数）
    if args.input_dir:
        dataset_config['raw_data_path'] = input_dir
    if args.output_dir:
        dataset_config['processed_data_path'] = output_dir
    
    # 数据集参数
    num_speakers = args.num_speakers if args.num_speakers is not None else dataset_config['num_speakers']
    num_utterances = args.num_utterances if args.num_utterances is not None else dataset_config['samples_per_speaker']
    
    # 计算训练集、验证集和测试集样本数
    # wsj0-2mix 格式: train (训练), dev (验证), test (测试)
    total_samples = num_speakers * num_utterances
    train_ratio = dataset_config.get('train_ratio', 0.8)
    dev_ratio = dataset_config.get('dev_ratio', 0.1)  # 验证集比例，默认10%
    
    num_train = args.num_train if args.num_train is not None else int(total_samples * train_ratio)
    num_dev = int(total_samples * dev_ratio)
    num_test = args.num_test if args.num_test is not None else (total_samples - num_train - num_dev)
    
    # 音频参数
    if args.sample_rate is not None:
        dataset_config['sample_rate'] = args.sample_rate
    
    # 处理音频长度参数
    if args.audio_length is not None:
        # 如果指定固定长度，设置为相同的最小和最大值
        dataset_config['audio_length_range'] = [args.audio_length, args.audio_length]
    elif args.audio_length_min is not None or args.audio_length_max is not None:
        # 如果指定长度范围
        if 'audio_length_range' in dataset_config:
            length_min = args.audio_length_min if args.audio_length_min is not None else dataset_config['audio_length_range'][0]
            length_max = args.audio_length_max if args.audio_length_max is not None else dataset_config['audio_length_range'][1]
        else:
            # 如果配置中没有范围，使用默认值或命令行参数
            default_length = dataset_config.get('audio_length', 4.0)
            length_min = args.audio_length_min if args.audio_length_min is not None else default_length
            length_max = args.audio_length_max if args.audio_length_max is not None else default_length
        dataset_config['audio_length_range'] = [length_min, length_max]
    
    if args.snr_min is not None or args.snr_max is not None:
        snr_min = args.snr_min if args.snr_min is not None else dataset_config['snr_range'][0]
        snr_max = args.snr_max if args.snr_max is not None else dataset_config['snr_range'][1]
        dataset_config['snr_range'] = [snr_min, snr_max]
    
    # 随机种子
    seed = args.seed if args.seed is not None else config['training']['seed']
    random.seed(seed)
    torch.manual_seed(seed)
    
    print("=" * 80)
    print("从 AISHELL-3 生成混合语音数据集（wsj0-2mix 格式）")
    print("=" * 80)
    print(f"配置文件: {args.config}")
    print(f"AISHELL-3 路径: {dataset_config['raw_data_path']}")
    print(f"输出路径: {dataset_config['processed_data_path']}")
    print(f"采样率: {dataset_config['sample_rate']} Hz")
    
    # 显示音频长度（支持固定长度和长度范围）
    if 'audio_length_range' in dataset_config:
        length_range = dataset_config['audio_length_range']
        if length_range[0] == length_range[1]:
            print(f"音频长度: {length_range[0]} 秒（固定）")
        else:
            print(f"音频长度范围: {length_range[0]} - {length_range[1]} 秒（随机）")
    else:
        print(f"音频长度: {dataset_config.get('audio_length', 4.0)} 秒")
    
    print(f"SNR 范围: {dataset_config['snr_range']} dB")
    print(f"说话人数量: {num_speakers}")
    print(f"每个说话人的语音数量: {num_utterances}")
    print(f"数据集划分:")
    print(f"  训练集(train): {num_train} 样本 ({train_ratio*100:.0f}%)")
    print(f"  验证集(dev): {num_dev} 样本 ({dev_ratio*100:.0f}%)")
    print(f"  测试集(test): {num_test} 样本 ({(1-train_ratio-dev_ratio)*100:.0f}%)")
    print(f"随机种子: {seed}")
    print()
    
    # 扫描 AISHELL-3 数据集
    print("步骤 1: 扫描 AISHELL-3 数据集...")
    aishell3_path = dataset_config['raw_data_path']
    speaker_utterances = scan_aishell3_dataset(
        aishell3_path, 
        num_speakers=num_speakers,
        num_utterances_per_speaker=num_utterances
    )
    
    total_utterances = sum(len(files) for files in speaker_utterances.values())
    print(f"✓ 选择了 {len(speaker_utterances)} 个说话人，共 {total_utterances} 个语音文件")
    
    # 生成训练集 (train)
    print("\n步骤 2: 生成训练集 (train)...")
    generate_mixture_dataset(config, speaker_utterances, num_samples=num_train, split='train')
    
    # 生成验证集 (dev)
    print("\n步骤 3: 生成验证集 (dev)...")
    generate_mixture_dataset(config, speaker_utterances, num_samples=num_dev, split='dev')
    
    # 生成测试集 (test)
    print("\n步骤 4: 生成测试集 (test)...")
    generate_mixture_dataset(config, speaker_utterances, num_samples=num_test, split='test')
    
    print("\n" + "=" * 80)
    print("数据集生成完成!")
    print("=" * 80)
    sr_folder = f"wav{dataset_config['sample_rate']//1000}k"
    output_path = os.path.join(dataset_config['processed_data_path'], sr_folder)
    print(f"训练集(train): {num_train} 样本")
    print(f"验证集(dev): {num_dev} 样本")
    print(f"测试集(test): {num_test} 样本")
    print(f"数据位置: {output_path}/")
    print(f"元数据位置: {output_path}/metadata/")
    print("\n可以运行以下命令开始训练:")
    print("python scripts/3_train.py")


if __name__ == "__main__":
    main()
