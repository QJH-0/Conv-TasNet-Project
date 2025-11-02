"""
数据生成脚本
从AISHELL-3数据集生成混合语音数据
"""

import os
import sys
import argparse
import yaml
import torch
import torchaudio
import random
import json
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
    生成混合语音数据集
    
    Args:
        config: 配置字典
        speaker_utterances: {speaker_id: [audio_paths]}
        num_samples: 样本数量
        split: 'train' 或 'test'
    """
    dataset_config = config['dataset']
    sample_rate = dataset_config['sample_rate']
    audio_length = dataset_config['audio_length']
    snr_range = dataset_config['snr_range']
    
    # 计算目标样本点数
    target_length = int(sample_rate * audio_length)
    
    # 输出目录
    output_dir = os.path.join(
        dataset_config['processed_data_path'],
        f'mixed/{split}'
    )
    mixture_dir = os.path.join(output_dir, 'mixture')
    clean_dir = os.path.join(output_dir, 'clean')
    
    os.makedirs(mixture_dir, exist_ok=True)
    os.makedirs(clean_dir, exist_ok=True)
    
    # 准备说话人列表
    speakers = list(speaker_utterances.keys())
    
    print(f"\nGenerating {num_samples} {split} samples...")
    print(f"Output directory: {output_dir}")
    print(f"Available speakers: {len(speakers)}")
    
    # 保存元数据
    metadata = []
    
    for i in tqdm(range(num_samples)):
        try:
            # 随机选择两个不同的说话人
            speaker1, speaker2 = random.sample(speakers, 2)
            
            # 随机选择每个说话人的一个语音文件
            audio_path1 = random.choice(speaker_utterances[speaker1])
            audio_path2 = random.choice(speaker_utterances[speaker2])
            
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
            
            # 保存
            base_name = f"sample_{i:04d}"
            
            # 混合音频（使用归一化后的）
            mixture_path = os.path.join(mixture_dir, f"{base_name}.wav")
            torchaudio.save(mixture_path, mixture_norm.unsqueeze(0), sample_rate)
            
            # 干净音频（使用归一化后的）
            s1_path = os.path.join(clean_dir, f"{base_name}_s1.wav")
            s2_path = os.path.join(clean_dir, f"{base_name}_s2.wav")
            torchaudio.save(s1_path, sources_norm[0].unsqueeze(0), sample_rate)
            torchaudio.save(s2_path, sources_norm[1].unsqueeze(0), sample_rate)
            
            # 记录元数据（包含验证信息）
            metadata.append({
                'sample_id': base_name,
                'speaker1': speaker1,
                'speaker2': speaker2,
                'audio1_path': audio_path1,
                'audio2_path': audio_path2,
                'snr_db_target': float(snr_db),
                'snr_db_actual': float(actual_snr),
                'snr_error_db': float(abs(actual_snr - snr_db)),
                'reconstruction_error': float(reconstruction_error),
                'is_valid': bool(reconstruction_error < 1e-6 and abs(actual_snr - snr_db) < 0.5)
            })
            
        except Exception as e:
            print(f"\n警告: 生成样本 {i} 时出错: {e}")
            continue
    
    # 保存元数据
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Generated {len(metadata)} samples in {output_dir}")
    print(f"✓ Metadata saved to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate mixture dataset from AISHELL-3')
    parser.add_argument('--config', type=str, default='config/config.yml',
                       help='Path to config file')
    parser.add_argument('--num-speakers', type=int, default=None,
                       help='Number of speakers to use (override config)')
    parser.add_argument('--num-utterances', type=int, default=None,
                       help='Number of utterances per speaker (override config)')
    parser.add_argument('--num-train', type=int, default=None,
                       help='Number of training samples (override config)')
    parser.add_argument('--num-test', type=int, default=None,
                       help='Number of test samples (override config)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed (override config)')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 从配置文件读取参数,命令行参数可以覆盖
    dataset_config = config['dataset']
    num_speakers = args.num_speakers if args.num_speakers is not None else dataset_config['num_speakers']
    num_utterances = args.num_utterances if args.num_utterances is not None else dataset_config['samples_per_speaker']
    
    # 计算训练集和测试集样本数
    total_samples = num_speakers * num_utterances
    train_ratio = dataset_config.get('train_ratio', 0.8)
    num_train = args.num_train if args.num_train is not None else int(total_samples * train_ratio)
    num_test = args.num_test if args.num_test is not None else (total_samples - num_train)
    
    # 随机种子
    seed = args.seed if args.seed is not None else config['training']['seed']
    random.seed(seed)
    torch.manual_seed(seed)
    
    print("=" * 80)
    print("从 AISHELL-3 生成混合语音数据集")
    print("=" * 80)
    print(f"配置文件: {args.config}")
    print(f"AISHELL-3 路径: {dataset_config['raw_data_path']}")
    print(f"输出路径: {dataset_config['processed_data_path']}")
    print(f"采样率: {dataset_config['sample_rate']} Hz")
    print(f"音频长度: {dataset_config['audio_length']} 秒")
    print(f"SNR 范围: {dataset_config['snr_range']} dB")
    print(f"说话人数量: {num_speakers}")
    print(f"每个说话人的语音数量: {num_utterances}")
    print(f"训练集比例: {train_ratio}")
    print(f"训练样本: {num_train}")
    print(f"测试样本: {num_test}")
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
    
    # 生成训练集
    print("\n步骤 2: 生成训练集...")
    generate_mixture_dataset(config, speaker_utterances, num_samples=num_train, split='train')
    
    # 生成测试集
    print("\n步骤 3: 生成测试集...")
    generate_mixture_dataset(config, speaker_utterances, num_samples=num_test, split='test')
    
    print("\n" + "=" * 80)
    print("数据集生成完成!")
    print("=" * 80)
    print(f"训练样本: {num_train}")
    print(f"测试样本: {num_test}")
    print(f"数据位置: {dataset_config['processed_data_path']}/mixed/")
    print("\n可以运行以下命令开始训练:")
    print("python scripts/3_train.py")


if __name__ == "__main__":
    main()
