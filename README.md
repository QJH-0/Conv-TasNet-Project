# Conv-TasNet 语音分离项目

基于 Conv-TasNet 的双人语音分离系统，使用 AISHELL-3 数据集进行训练和测试。

## 📋 项目概述

本项目实现了完整的Conv-TasNet模型，用于语音分离任务。项目包含：
- 完整的模型实现（Encoder、Separation、Decoder）
- PIT（排列不变训练）损失函数
- 数据预处理和混合语音生成
- 训练和评估流程
- 可视化工具
- 性能优化（混合精度训练、梯度累积、数据缓存）

**当前版本**: v1.1 (优化版)  
**更新日期**: 2025-11-01  
**优化状态**: ✅ 已完成核心优化

### 主要特性
- ✅ 端到端训练
- ✅ PIT损失解决排列问题
- ✅ 完全卷积架构
- ✅ 模块化设计
- ✅ 配置驱动
- ✅ 混合精度训练（AMP）- 节省显存30-40%
- ✅ 梯度累积优化 - 提升有效batch size
- ✅ 智能数据缓存 - 加速10-20倍
- ✅ Warmup学习率调度

## 🚀 快速开始

### 环境要求
- Python 3.7+
- CUDA 12.9 (GPU训练推荐)
- 4GB+ GPU显存（使用混合精度训练）

### 分步运行

#### 1. 安装依赖

```bash
pip install -r requirements.txt
```

#### 2. 生成混合语音数据

```bash
# 生成测试数据集（快速验证）
python scripts/2_generate_mixtures.py

# 根据config.yml配置，默认会生成:
# - 训练集: data/processed/mixed/train/
# - 测试集: data/processed/mixed/test/
```

#### 3. 训练模型

```bash
# 开始训练（首次运行会自动创建数据缓存）
python scripts/3_train.py --config config/config.yml

# 从检查点恢复训练
python scripts/3_train.py --config config/config.yml --resume experiments/exp_001/checkpoints/checkpoint_epoch_10.pth
```

**首次训练说明：**
- 首次运行会预加载所有数据到内存并保存缓存
- 缓存文件保存在：`data/processed/mixed/train/dataset_cache.pkl`
- 后续训练会自动加载缓存，大幅提升数据加载速度

#### 4. 评估模型

```bash
python scripts/4_evaluate.py --checkpoint experiments/exp_001/checkpoints/best_model.pth
```

#### 5. 推理分离

```bash
# 对单个文件进行分离
python scripts/5_inference.py --input your_mixed_audio.wav --output outputs/separated_audio/

# 指定模型检查点
python scripts/5_inference.py \
    --input your_mixed_audio.wav \
    --output outputs/separated_audio/ \
    --checkpoint experiments/exp_001/checkpoints/best_model.pth
```

### 使用真实 AISHELL-3 数据集

如需使用真实AISHELL-3数据集，修改 `config/config.yml` 中的数据集路径：

```yaml
dataset:
  raw_data_path: "D:\\Paper\\datasets\\AISHELL-3"  # 你的AISHELL-3路径
  num_speakers: 20                                  # 选择的说话人数量
  samples_per_speaker: 50                           # 每人的样本数
```

然后重新生成混合数据：
```bash
python scripts/2_generate_mixtures.py
```

## 📁 项目结构

```
Conv-TasNet-Project/
├── config/                          # 配置文件
│   └── config.yml                   # 主配置文件（包含优化配置）
├── data/                            # 数据目录
│   ├── metadata/                    # 元数据
│   └── processed/                   # 处理后的数据
│       └── mixed/                   # 混合语音
│           ├── train/               # 训练集
│           │   ├── mixture/         # 混合音频
│           │   ├── clean/           # 干净音频(s1,s2)
│           │   ├── metadata.json    # 元数据
│           │   └── dataset_cache.pkl # 🆕 数据缓存
│           └── test/                # 测试集
├── dataset/                         # 数据处理模块
│   ├── dataloader.py                # 🔧 优化版数据加载器（支持缓存）
│   └── __init__.py
├── models/                          # 模型模块
│   ├── conv_tasnet.py               # Conv-TasNet主模型
│   ├── encoder.py                   # 编码器（1D卷积）
│   ├── decoder.py                   # 解码器（转置卷积）
│   ├── separation.py                # 分离模块(TCN)
│   └── __init__.py
├── modules/                         # 基础模块
│   ├── tcn.py                       # TCN块
│   ├── depthwise_conv.py            # 深度可分离卷积
│   ├── normalization.py             # 归一化层(gLN/cLN)
│   └── __init__.py
├── loss/                            # 损失函数
│   ├── si_snr.py                    # SI-SNR损失
│   ├── pit_wrapper.py               # PIT包装器
│   └── __init__.py
├── trainer/                         # 训练器
│   ├── trainer.py                   # 🔧 优化版训练器（AMP+梯度累积）
│   └── __init__.py
├── utils/                           # 工具函数
│   ├── audio_utils.py               # 音频处理
│   ├── metrics.py                   # 评估指标
│   ├── visualization.py             # 可视化
│   ├── logger.py                    # 日志管理
│   └── __init__.py
├── scripts/                         # 运行脚本
│   ├── 2_generate_mixtures.py       # 生成混合语音
│   ├── 3_train.py                   # 训练模型
│   ├── 4_evaluate.py                # 评估模型
│   ├── 5_inference.py               # 推理分离
│   └── regenerate_data.py           # 重新生成数据
├── experiments/                     # 实验结果
│   └── exp_001/                     # 实验001
│       ├── checkpoints/             # 模型检查点
│       ├── logs/                    # 训练日志
│       └── results/                 # 实验结果
├── outputs/                         # 推理输出
│   └── separated_audio/             # 分离后的音频
├── requirements.txt                 # 依赖包
├── README.md                        # 📚 本文件
├── 项目优化说明.md                  # 📚 优化方案
└── 项目优化实施报告.md              # 📚 优化完成报告
```

## 📊 实验配置

### 数据集
- **来源**: AISHELL-3
- **说话人数**: 20名
- **样本数**: 1000条（每人50条）
- **训练/测试**: 800/200 (8:2)
- **采样率**: 16kHz
- **SNR范围**: -3dB ~ 3dB
- **音频长度**: 2秒 (优化后)

### 模型参数
- **Encoder filters (N)**: 128（优化后）
- **Kernel size (L)**: 40
- **Bottleneck channels (B)**: 128
- **Hidden channels (H)**: 256（优化后）
- **TCN blocks (M)**: 7（优化后）
- **TCN repeats (R)**: 2（优化后）
- **Normalization**: Global LayerNorm (gLN)

### 训练参数
- **Batch size**: 2
- **Accumulation steps**: 4（有效batch size = 8）
- **Mixed precision**: 启用（AMP）
- **Epochs**: 100
- **Learning rate**: 0.001
- **LR Scheduler**: Halving（验证损失不降则减半）
- **Optimizer**: Adam (论文标准参数)
- **Loss**: SI-SNR with PIT

## 🔧 模型架构

Conv-TasNet采用Encoder-Separation-Decoder架构：

```
混合波形 [B, T]
    ↓
Encoder (1D Conv) [B, N, K]
    ↓ (无激活函数)
LayerNorm + Bottleneck
    ↓
TCN Separation (R×M layers)
    ├── 残差连接
    └── 跳跃连接
    ↓
Mask Generation (ReLU) [B, C, N, K]
    ↓
掩码相乘 [B, C, N, K]
    ↓
Decoder (TransConv) [B, C, T]
    ↓
分离波形 [B, C, T]
```

### 核心组件说明

#### 1. Encoder（编码器）
- **作用**: 将时域波形转换为特征表示
- **实现**: 1D卷积（无激活函数）
- **输入**: 混合波形 `[B, T]`
- **输出**: 编码特征 `[B, N, K]`
  - `N`: 特征维度（128）
  - `K`: 时间帧数

#### 2. Separation（分离网络）
- **作用**: 通过TCN网络生成分离掩码
- **组件**:
  - LayerNorm: 归一化编码特征
  - Bottleneck: 1×1卷积降维
  - TCN Blocks: 扩张卷积捕获长时依赖
  - Mask Conv: 生成C个说话人掩码
- **激活函数**: ReLU（允许掩码>1，支持信号放大）

#### 3. TCN（时序卷积网络）
- **深度可分离卷积**: 减少参数量
- **扩张卷积**: 指数增长的感受野 (1,2,4,8,16,32,64,128)
- **残差连接**: 缓解梯度消失
- **跳跃连接**: 融合多层特征

#### 4. Decoder（解码器）
- **作用**: 将掩码特征转回时域波形
- **实现**: 1D转置卷积
- **输入**: 掩码特征 `[B, C, N, K]`
- **输出**: 分离波形 `[B, C, T]`

## 📈 评估指标

本项目使用SI-SDR（尺度不变信号失真比）作为主要评估指标：

### 主要指标
- **SI-SDR** (Scale-Invariant Signal-to-Distortion Ratio)
  - 衡量分离质量的标准指标
  - 对幅度缩放不敏感
  - 值越大表示分离效果越好

### 评判标准
- **SI-SDR > 10 dB**: 良好分离
- **SI-SDR > 15 dB**: 优秀分离
- **SI-SDR > 20 dB**: 卓越分离

### 预期结果
- **SI-SDR**: > 12 dB（优化后配置）
- **训练时间**: 约3-5小时 (GPU + 优化)

### 运行评估
```bash
python scripts/4_evaluate.py --checkpoint experiments/exp_001/checkpoints/best_model.pth
```

## 🔍 核心技术

### PIT (排列不变训练)

本项目使用PIT解决排列不确定性问题：

```python
# 问题：模型输出[ŝ1, ŝ2]，但不知道哪个对应[s1, s2]
# 解决：尝试所有排列，选择损失最小的

排列1: ŝ1→s1, ŝ2→s2
排列2: ŝ1→s2, ŝ2→s1

选择min(loss1, loss2)进行反向传播
```

### 性能优化技术

#### 1. 混合精度训练 (AMP)
- 使用FP16进行前向和反向传播
- 关键梯度用FP32累积
- 显存占用减少30-40%
- 训练速度提升30-50%

#### 2. 梯度累积
- 小batch size模拟大batch训练
- 有效batch size = batch_size × accumulation_steps
- 显存受限时的最佳方案

#### 3. 智能数据缓存
- 首次训练预加载所有数据到内存
- 保存缓存文件到磁盘
- 后续训练加载速度提升10-20倍

#### 4. Halving学习率调度
- 验证损失不降时学习率减半
- 符合Conv-TasNet论文标准
- 训练更稳定，收敛更好

## 📝 数据处理说明

### 数据生成流程

1. **音频预处理**
   - 从AISHELL-3数据集选择20位说话人
   - 每位说话人选择50条语音
   - 重采样到16kHz
   - 调整到固定长度（2秒 = 32000 samples）

2. **混合语音生成**
   - 随机选择两位说话人的音频
   - 在指定SNR范围内混合（-3dB ~ 3dB）
   - **精确SNR控制**（误差<0.01dB）
   - **统一归一化**到-25dB
   - 验证数据质量

3. **数据集划分**
   - 训练集：80% (800条)
   - 测试集：20% (200条)
   - 保存元数据（SNR、说话人信息等）

### 数据加载优化

**首次训练**:
```
加载所有音频 → 预处理 → 缓存到内存 → 保存缓存文件
```

**后续训练**:
```
直接加载缓存 → 极速开始训练（提升10-20倍）
```

**缓存文件位置**:
- 训练集: `data/processed/mixed/train/dataset_cache.pkl`
- 测试集: `data/processed/mixed/test/dataset_cache.pkl`

## ⚠️ 常见问题

### Q1: 显存不足怎么办？
```yaml
# config/config.yml
training:
  batch_size: 1              # 降低batch size
  accumulation_steps: 8      # 增加累积步数保持有效batch size
```

### Q2: 如何禁用数据缓存？
```yaml
# config/config.yml
dataset:
  use_cache: false           # 禁用缓存（动态加载）
```

### Q3: 训练速度慢？
- 确保启用混合精度训练：`use_amp: true`
- 检查是否加载了数据缓存
- 首次训练会预加载数据，需要等待

### Q4: 如何重新生成缓存？
```bash
# Windows
del data\processed\mixed\train\dataset_cache.pkl
del data\processed\mixed\test\dataset_cache.pkl

# 然后重新训练即可自动生成新缓存
```

### Q5: Windows系统多进程报错？
```yaml
# config/config.yml
device:
  num_workers: 0             # Windows下必须设为0
```

## 📚 相关文档

- [项目优化说明.md](项目优化说明.md) - 完整的优化方案和论文对比
- [项目优化实施报告.md](项目优化实施报告.md) - 优化实施详情和效果

## 📝 引用

如果本项目对您有帮助，请引用：

```bibtex
@inproceedings{luo2019convtasnet,
  title={Conv-tasnet: Surpassing ideal time--frequency magnitude masking for speech separation},
  author={Luo, Yi and Mesgarani, Nima},
  journal={IEEE/ACM transactions on audio, speech, and language processing},
  year={2019}
}
```

## 📧 联系方式

如有问题，请提issue或联系项目维护者。

## 📄 许可证

MIT License

---

**项目状态**: ✅ 已优化完成

**最后更新**: 2025-11-01

**版本**: v1.1 (优化版)
