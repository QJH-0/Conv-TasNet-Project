"""
训练器类
完整的训练、验证、保存流程
"""

import os
import torch
import torch.nn as nn
from tqdm import tqdm
import sys
import numpy as np
from torch.cuda.amp import autocast, GradScaler

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loss.si_snr import SI_SNR_Loss
from loss.pit_wrapper import PITLossWrapper
from utils.metrics import evaluate_separation
from utils.visualization import plot_loss_curves, plot_metrics


class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Warmup + Cosine Annealing 学习率调度器
    
    前warmup_epochs个epoch线性增长到base_lr
    之后使用余弦退火降到min_lr
    """
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, 
                 min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup 阶段：线性增长
            alpha = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine Annealing 阶段
            progress = (self.last_epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            return [
                self.min_lr + (base_lr - self.min_lr) * 
                0.5 * (1 + np.cos(np.pi * progress))
                for base_lr in self.base_lrs
            ]


class Trainer:
    """
    Conv-TasNet 训练器
    
    职责:
    1. 模型训练循环
    2. 验证
    3. Checkpoint 保存
    4. 学习率调度
    5. 早停
    6. 日志记录
    """
    
    def __init__(self, model, config, logger, device='cuda'):
        """
        Args:
            model: Conv-TasNet 模型
            config: 配置字典
            logger: 日志记录器
            device: 设备
        """
        self.model = model.to(device)
        self.config = config
        self.logger = logger
        self.device = device
        
        # 训练配置
        train_config = config['training']
        self.num_epochs = train_config['num_epochs']
        self.gradient_clip = train_config['gradient_clip']
        self.early_stopping_patience = train_config['early_stopping_patience']
        
        # 梯度累积配置
        self.accumulation_steps = train_config.get('accumulation_steps', 1)
        self.effective_batch_size = config['training']['batch_size'] * self.accumulation_steps
        
        # 混合精度训练配置
        self.use_amp = train_config.get('use_amp', False)
        self.scaler = GradScaler() if self.use_amp and torch.cuda.is_available() else None
        
        # 优化器
        self.optimizer = self._create_optimizer(train_config)
        
        # 学习率调度器
        self.scheduler = self._create_scheduler(train_config)
        
        # 损失函数（带PIT）
        base_loss = SI_SNR_Loss()
        self.criterion = PITLossWrapper(
            base_loss,
            num_speakers=config['model']['separation']['num_speakers']
        )
        
        # 日志与保存配置 - 确保使用绝对路径
        log_config = config['logging']
        
        # 获取项目根目录（从当前工作目录）
        self.project_root = os.getcwd()
        
        # 将相对路径转换为绝对路径
        self.checkpoint_dir = log_config['checkpoint_dir']
        if not os.path.isabs(self.checkpoint_dir):
            self.checkpoint_dir = os.path.join(self.project_root, self.checkpoint_dir)
            
        self.result_dir = log_config['result_dir']
        if not os.path.isabs(self.result_dir):
            self.result_dir = os.path.join(self.project_root, self.result_dir)
            
        self.save_interval = log_config['save_interval']
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        
        # 训练状态
        self.start_epoch = 1
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # 历史记录
        self.train_losses = []
        self.val_losses = []
        # 从配置文件读取评估设置
        self.eval_config = config.get('evaluation', {})
        enabled_metrics = self.eval_config.get('metrics', {}).get('enabled', ['si_sdr', 'sdr', 'sir', 'sar'])
        
        # 初始化指标历史
        self.metrics_history = {metric: [] for metric in enabled_metrics}
        # SI-SDRi 总是计算（如果有 si_sdr）
        if 'si_sdr' in enabled_metrics:
            self.metrics_history['si_sdri'] = []
        # STOI 如果启用也添加
        if self.eval_config.get('include_stoi', False):
            self.metrics_history['stoi'] = []
    
    def _create_optimizer(self, config):
        """创建优化器（论文标准配置）"""
        optimizer_type = config['optimizer']
        lr = config['learning_rate']
        weight_decay = config.get('weight_decay', 0)  # 从配置读取权重衰减，默认为0
        
        if optimizer_type == 'Adam':
            # 论文标准Adam配置
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                betas=(0.9, 0.999),  # 论文标准
                eps=1e-8,            # 论文标准
                weight_decay=weight_decay  # L2正则化
            )
        elif optimizer_type == 'SGD':
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay  # L2正则化
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
        
        return optimizer
    
    def _create_scheduler(self, config):
        """创建学习率调度器"""
        scheduler_config = config['scheduler']
        scheduler_type = scheduler_config['type']
        
        if scheduler_type == 'WarmupCosine':
            scheduler = WarmupCosineLR(
                self.optimizer,
                warmup_epochs=scheduler_config.get('warmup_epochs', 5),
                total_epochs=self.num_epochs,
                min_lr=scheduler_config.get('min_lr', 1e-6)
            )
            self.scheduler_step_on_epoch = True  # 每个epoch调用一次
        elif scheduler_type in ['ReduceLROnPlateau', 'Halving']:
            # Halving策略（论文标准）：验证损失不下降时学习率减半
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=scheduler_config.get('mode', 'min'),
                patience=scheduler_config.get('patience', 3),
                factor=scheduler_config.get('factor', 0.5),  # 减半
                min_lr=scheduler_config.get('min_lr', 1e-8)  # 论文使用1e-8
            )
            self.scheduler_step_on_epoch = False  # 根据验证损失调用
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_type}")
        
        return scheduler
    
    def train_epoch(self, dataloader):
        """
        训练一个 epoch（支持梯度累积和混合精度训练）
        
        Args:
            dataloader: 训练数据加载器
        
        Returns:
            avg_loss: 平均损失
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        accumulated_loss = 0
        
        # 在累积开始前清空梯度
        self.optimizer.zero_grad()
        
        pbar = tqdm(dataloader, desc='Training')
        for batch_idx, batch in enumerate(pbar):
            try:
                # 数据格式转换（新格式 → 旧格式）
                # 新格式: batch = {'mix': [B,T], 'ref': [[B,T], [B,T]]}
                # 旧格式: mixtures=[B,T], targets=[B,C,T]
                mixtures = batch['mix'].to(self.device)  # [B, T]
                targets = torch.stack(batch['ref'], dim=1).to(self.device)  # [B, C, T]
                
                # 使用混合精度训练
                with autocast(enabled=self.use_amp):
                    # 前向传播
                    estimations = self.model(mixtures)   # [B, C, T]
                    
                    # 计算 PIT 损失
                    loss, _ = self.criterion(estimations, targets)
                    
                    # 梯度累积：损失除以累积步数
                    loss = loss / self.accumulation_steps
                
                # 反向传播
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                accumulated_loss += loss.item()
                
                # 每accumulation_steps步更新一次参数
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    # 梯度裁剪
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )
                    
                    # 更新参数
                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad()
                    
                    # 记录
                    total_loss += accumulated_loss
                    num_batches += 1
                    
                    # 更新进度条
                    pbar.set_postfix({
                        'loss': f'{accumulated_loss:.4f}',
                        'eff_bs': self.effective_batch_size
                    })
                    accumulated_loss = 0
                
                # 清理显存
                del mixtures, targets, estimations, loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e) or "CUDA" in str(e):
                    self.logger.error(f"CUDA memory error: {e}")
                    self.logger.info("Clearing cache and skipping batch...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.optimizer.zero_grad()
                    accumulated_loss = 0
                    continue
                else:
                    raise e
        
        # 处理最后不足accumulation_steps的batch
        if accumulated_loss > 0:
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip
            )
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()
            total_loss += accumulated_loss
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss
    
    def validate(self, dataloader):
        """
        验证
        
        Args:
            dataloader: 验证数据加载器
        
        Returns:
            avg_loss: 平均损失
            metrics: 评估指标
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Validation'):
                try:
                    # 数据格式转换（新格式 → 旧格式）
                    mixtures = batch['mix'].to(self.device)
                    targets = torch.stack(batch['ref'], dim=1).to(self.device)
                    
                    # 前向传播
                    estimations = self.model(mixtures)
                    
                    # 计算损失
                    loss, _ = self.criterion(estimations, targets)
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # 清理显存
                    del mixtures, targets, estimations, loss
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except RuntimeError as e:
                    if "out of memory" in str(e) or "CUDA" in str(e):
                        self.logger.error(f"CUDA memory error during validation: {e}")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
        
        avg_loss = total_loss / num_batches
        
        # 从配置读取评估设置
        enabled_metrics = self.eval_config.get('metrics', {}).get('enabled', ['si_sdr', 'sdr', 'sir', 'sar'])
        use_asteroid = self.eval_config.get('use_asteroid', True)
        include_stoi = self.eval_config.get('include_stoi', False)
        
        # 计算评估指标（使用 Asteroid 自动 PIT + 误差分解）
        metrics = evaluate_separation(
            self.model, 
            dataloader, 
            self.device,
            metrics=enabled_metrics,
            use_asteroid=use_asteroid,
            include_stoi=include_stoi
        )
        
        return avg_loss, metrics
    
    def save_checkpoint(self, epoch, is_best=False):
        """
        保存检查点（兼容单GPU和多GPU）
        
        Args:
            epoch: 当前epoch
            is_best: 是否最佳模型
        """
        # 获取实际模型（移除DataParallel包装）
        model_to_save = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),  # 保存不带module.前缀的模型
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'metrics_history': self.metrics_history
        }
        
        # 保存常规检查点
        if epoch % self.save_interval == 0:
            path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint, path)
            self.logger.info(f"Checkpoint saved: {path}")
        
        # 保存最佳模型
        if is_best:
            path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, path)
            self.logger.info(f"Best model saved: {path}")
    
    def load_checkpoint(self, checkpoint_path):
        """
        加载检查点（兼容单GPU和多GPU训练的checkpoint）
        
        Args:
            checkpoint_path: 检查点路径
        """
        from collections import OrderedDict
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # 获取state_dict
        state_dict = checkpoint['model_state_dict']
        
        # 处理DataParallel的key不匹配问题
        # 如果加载的是多GPU训练的模型，但当前是单GPU
        if not isinstance(self.model, torch.nn.DataParallel) and list(state_dict.keys())[0].startswith('module.'):
            # 移除'module.'前缀
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k  # 移除'module.'
                new_state_dict[name] = v
            state_dict = new_state_dict
            self.logger.info("Converted multi-GPU checkpoint to single-GPU format")
        # 如果当前是多GPU，但加载的是单GPU训练的模型
        elif isinstance(self.model, torch.nn.DataParallel) and not list(state_dict.keys())[0].startswith('module.'):
            # 添加'module.'前缀
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = 'module.' + k
                new_state_dict[name] = v
            state_dict = new_state_dict
            self.logger.info("Converted single-GPU checkpoint to multi-GPU format")
        
        self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        # 从checkpoint恢复metrics_history，如果不存在则初始化
        default_metrics = {metric: [] for metric in self.metrics_history.keys()}
        self.metrics_history = checkpoint.get('metrics_history', default_metrics)
        
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
        self.logger.info(f"Resuming from epoch {self.start_epoch}")
    
    def run(self, train_loader, val_loader, resume_from=None):
        """
        运行完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            resume_from: 恢复训练的检查点路径
        """
        # 恢复训练
        if resume_from and os.path.exists(resume_from):
            self.load_checkpoint(resume_from)
        
        self.logger.info("=" * 80)
        self.logger.info("Starting Training")
        self.logger.info("=" * 80)
        self.logger.info(f"Total epochs: {self.num_epochs}")
        self.logger.info(f"Start epoch: {self.start_epoch}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Training samples: {len(train_loader.dataset)}")
        self.logger.info(f"Validation samples: {len(val_loader.dataset)}")
        self.logger.info(f"\nOptimization Settings:")
        self.logger.info(f"  Optimizer: {self.config['training']['optimizer']}")
        self.logger.info(f"  Learning rate: {self.config['training']['learning_rate']}")
        self.logger.info(f"  Weight decay: {self.config['training'].get('weight_decay', 0)}")
        self.logger.info(f"  Batch size: {self.config['training']['batch_size']}")
        self.logger.info(f"  Accumulation steps: {self.accumulation_steps}")
        self.logger.info(f"  Effective batch size: {self.effective_batch_size}")
        self.logger.info(f"  Mixed precision (AMP): {self.use_amp}")
        self.logger.info(f"  Scheduler: {self.config['training']['scheduler']['type']}")
        
        try:
            for epoch in range(self.start_epoch, self.num_epochs + 1):
                self.logger.info(f"\nEpoch {epoch}/{self.num_epochs}")
                self.logger.info("-" * 80)
                
                # 训练
                train_loss = self.train_epoch(train_loader)
                self.train_losses.append(train_loss)
                self.logger.info(f"Train Loss: {train_loss:.4f}")
                
                # 验证
                val_loss, metrics = self.validate(val_loader)
                self.val_losses.append(val_loss)
                
                # 记录所有配置的指标
                for metric_name in self.metrics_history.keys():
                    self.metrics_history[metric_name].append(metrics.get(metric_name, 0))
                
                # 日志输出
                self.logger.info(f"Val Loss: {val_loss:.4f}")
                for metric_name, value in metrics.items():
                    if metric_name == 'stoi':
                        self.logger.info(f"STOI: {value:.4f} (可懂度)")
                    elif metric_name == 'sir':
                        self.logger.info(f"SIR: {value:.2f} dB (串扰抑制)")
                    elif metric_name == 'sar':
                        self.logger.info(f"SAR: {value:.2f} dB (算法失真)")
                    else:
                        self.logger.info(f"{metric_name.upper()}: {value:.2f} dB")
                
                # 学习率调度
                if self.scheduler_step_on_epoch:
                    # WarmupCosine: 每个epoch调用
                    self.scheduler.step()
                else:
                    # ReduceLROnPlateau: 根据验证损失调用
                    self.scheduler.step(val_loss)
                
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.info(f"Learning Rate: {current_lr:.6f}")
                
                # 保存检查点
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.logger.info("✓ New best model!")
                else:
                    self.patience_counter += 1
                
                self.save_checkpoint(epoch, is_best)
                
                # 早停检查
                if self.patience_counter >= self.early_stopping_patience:
                    self.logger.info(f"\nEarly stopping triggered after {epoch} epochs")
                    break
                
                # 可视化
                if self.config['logging']['visualize']:
                    plot_loss_curves(
                        self.train_losses,
                        self.val_losses,
                        os.path.join(self.result_dir, 'loss_curves.png')
                    )
                    # 从配置读取要绘制的指标
                    plot_metric_list = self.eval_config.get('plot_metrics', None)
                    plot_metrics(
                        self.metrics_history,
                        os.path.join(self.result_dir, 'metrics.png'),
                        plot_metrics=plot_metric_list
                    )
        
        except KeyboardInterrupt:
            self.logger.info("\n Training interrupted by user")
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("Training Completed")
        self.logger.info("=" * 80)
        self.logger.info(f"Best Validation Loss: {self.best_val_loss:.4f}")
        
        # 输出所有记录的指标
        for metric_name, values in self.metrics_history.items():
            if values:  # 只输出有数据的指标
                best_val = max(values) if metric_name != 'loss' else min(values)
                final_val = values[-1]
                
                if metric_name == 'stoi':
                    self.logger.info(f"Best {metric_name.upper()}: {best_val:.4f}")
                    self.logger.info(f"Final {metric_name.upper()}: {final_val:.4f}")
                else:
                    self.logger.info(f"Best {metric_name.upper()}: {best_val:.2f} dB")
                    self.logger.info(f"Final {metric_name.upper()}: {final_val:.2f} dB")
