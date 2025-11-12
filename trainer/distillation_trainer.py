"""
知识蒸馏训练器
实现教师模型(Conv-TasNet) → 学生模型(BTCN)的知识蒸馏训练
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from typing import Dict, List, Tuple

from loss.distillation_loss import DistillationLoss, BasicDistillationLoss
from loss.pit_wrapper import PITLossWrapper
from utils.logger import setup_logger
from utils.metrics import calculate_metrics
from utils.visualization import plot_distillation_metrics
from models.adapter_layers import FeatureAdapter


class DistillationTrainer:
    """知识蒸馏训练器
    
    功能:
        1. 加载并冻结教师模型
        2. 初始化学生模型和适配层
        3. 实现蒸馏训练循环
        4. 动态调整损失权重
        5. 评估和保存模型
    """
    
    def __init__(self, teacher_model, student_model, train_loader, val_loader,
                 config, device, logger=None):
        """
        Args:
            teacher_model: 教师模型(Conv-TasNet)
            student_model: 学生模型(BTCN)
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            config: 配置字典
            device: 计算设备
            logger: 日志记录器
        """
        self.teacher = teacher_model
        self.student = student_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.logger = logger
        
        # 冻结教师模型
        self._freeze_teacher()
        
        # 创建适配层
        self.adapters = self._create_adapters()
        
        # 创建损失函数
        self.criterion = self._create_criterion()
        
        # 创建优化器
        self.optimizer = self._create_optimizer()
        
        # 创建学习率调度器
        self.scheduler = self._create_scheduler()
        
        # 混合精度训练
        self.use_amp = config['training'].get('use_amp', False)
        self.scaler = GradScaler() if self.use_amp else None
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_si_sdr = float('-inf')
        self.train_losses = []
        self.val_losses = []
        
        # 梯度累积
        self.accumulation_steps = config['training'].get('accumulation_steps', 1)
        
        # 早停
        self.early_stop_patience = config['training'].get('early_stopping_patience', 30)
        self.early_stop_counter = 0
        
        # 从配置读取评估设置
        self.eval_config = config.get('evaluation', {})
        
        self.logger.info("蒸馏训练器初始化完成")
        self.logger.info(f"教师模型参数: {sum(p.numel() for p in self.teacher.parameters()):,}")
        self.logger.info(f"学生模型参数: {sum(p.numel() for p in self.student.parameters()):,}")
        self.logger.info(f"适配层参数: {sum(p.numel() for p in self.adapters.parameters()):,}")
    
    def _freeze_teacher(self):
        """冻结教师模型参数"""
        self.teacher.eval()
        
        # 冻结教师模型参数
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.logger.info("教师模型已冻结")
    
    def _create_adapters(self) -> nn.ModuleDict:
        """创建特征适配层"""
        adapters = nn.ModuleDict()
        
        distill_config = self.config['distillation']
        alignment_config = distill_config.get('feature_alignment', {})
        
        if not alignment_config.get('enabled', False):
            return adapters
        
        # 编码器适配层
        if alignment_config.get('encoder_adapter', {}).get('enabled', False):
            enc_config = alignment_config['encoder_adapter']
            student_dim = enc_config['student_dim']
            teacher_dim = enc_config['teacher_dim']
            
            if student_dim != teacher_dim:
                adapters['encoder'] = FeatureAdapter(
                    student_dim, teacher_dim,
                    use_bn=alignment_config.get('use_bn', False)
                ).to(self.device)
                self.logger.info(f"编码器适配层: {student_dim} -> {teacher_dim}")
        
        # TCN特征适配层
        if alignment_config.get('tcn_adapter', {}).get('enabled', False):
            tcn_config = alignment_config['tcn_adapter']
            student_dim = tcn_config['student_dim']
            teacher_dim = tcn_config['teacher_dim']
            
            if student_dim != teacher_dim:
                adapters['tcn'] = FeatureAdapter(
                    student_dim, teacher_dim,
                    use_bn=alignment_config.get('use_bn', False)
                ).to(self.device)
                self.logger.info(f"TCN适配层: {student_dim} -> {teacher_dim}")
        
        return adapters
    
    def _create_criterion(self):
        """创建损失函数"""
        distill_config = self.config['distillation']
        mode = distill_config.get('mode', 'full')
        
        if mode == 'basic':
            # 基础蒸馏: L_spec + L_task
            weights = distill_config['basic_loss_weights']
            criterion = BasicDistillationLoss(
                lambda_spec=weights['lambda_spec'],
                lambda_task=weights['lambda_task']
            )
            self.logger.info("使用基础蒸馏损失 (L_spec + L_task)")
        else:
            # 完整蒸馏: 5个损失
            weights = distill_config['loss_weights']
            loss_weights = [
                weights['lambda1_spec'],
                weights['lambda2_enc'],
                weights['lambda3_tcn'],
                weights['lambda4_mask'],
                weights['lambda5_task']
            ]
            use_feature = (mode == 'full')
            criterion = DistillationLoss(
                loss_weights=loss_weights,
                use_feature_distillation=use_feature
            )
            self.logger.info(f"使用完整蒸馏损失, 权重: {loss_weights}")
        
        return criterion.to(self.device)
    
    def _create_optimizer(self):
        """创建优化器"""
        train_config = self.config['training']
        
        # 收集需要训练的参数
        params = list(self.student.parameters())
        params += list(self.adapters.parameters())
        
        optimizer = torch.optim.Adam(
            params,
            lr=train_config['learning_rate'],
            weight_decay=train_config.get('weight_decay', 1e-5)
        )
        
        self.logger.info(f"优化器: Adam, lr={train_config['learning_rate']}")
        return optimizer
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        train_config = self.config['training']
        scheduler_config = train_config.get('scheduler', {})
        
        scheduler_type = scheduler_config.get('type', 'ReduceLROnPlateau')
        
        if scheduler_type == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=scheduler_config.get('mode', 'min'),
                patience=scheduler_config.get('patience', 5),
                factor=scheduler_config.get('factor', 0.5),
                min_lr=scheduler_config.get('min_lr', 1e-6)
            )
        else:
            scheduler = None
        
        return scheduler
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.student.train()
        self.teacher.eval()
        
        epoch_losses = {
            'total_loss': 0.0,
            'L_spec': 0.0,
            'L_enc': 0.0,
            'L_tcn': 0.0,
            'L_mask': 0.0,
            'L_task': 0.0
        }
        
        num_batches = len(self.train_loader)
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 数据格式转换（字典格式 → tensor格式）
            mixture = batch['mix'].to(self.device)  # [B, T]
            targets = torch.stack(batch['ref'], dim=1).to(self.device)  # [B, C, T]
            
            # 教师模型前向传播（no_grad）
            with torch.no_grad():
                if isinstance(self.criterion, BasicDistillationLoss):
                    # 基础蒸馏只需要输出
                    teacher_output = self.teacher(mixture)
                else:
                    # 完整蒸馏需要中间特征
                    teacher_outputs = self.teacher(mixture, return_intermediate=True)
            
            # 学生模型前向传播
            if self.use_amp:
                with autocast():
                    student_outputs = self._forward_student(mixture)
                    loss, loss_dict = self._compute_loss(
                        student_outputs, teacher_outputs if not isinstance(self.criterion, BasicDistillationLoss) else teacher_output,
                        targets
                    )
                    loss = loss / self.accumulation_steps
                
                # 反向传播
                self.scaler.scale(loss).backward()
                
                # 梯度累积
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    # 梯度裁剪
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.student.parameters()) + list(self.adapters.parameters()),
                        self.config['training'].get('gradient_clip', 200)
                    )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                student_outputs = self._forward_student(mixture)
                loss, loss_dict = self._compute_loss(
                    student_outputs, teacher_outputs if not isinstance(self.criterion, BasicDistillationLoss) else teacher_output,
                    targets
                )
                loss = loss / self.accumulation_steps
                
                # 反向传播
                loss.backward()
                
                # 梯度累积
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.student.parameters()) + list(self.adapters.parameters()),
                        self.config['training'].get('gradient_clip', 200)
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # 累积损失
            for key, value in loss_dict.items():
                if key in epoch_losses:
                    epoch_losses[key] += value
            
            # 打印进度
            if (batch_idx + 1) % 10 == 0:
                self.logger.info(
                    f"Epoch {self.current_epoch} [{batch_idx+1}/{num_batches}] "
                    f"Loss: {loss_dict['total_loss']:.4f}"
                )
        
        # 计算平均损失
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def _forward_student(self, mixture):
        """学生模型前向传播并应用适配层"""
        if isinstance(self.criterion, BasicDistillationLoss):
            # 基础蒸馏只返回输出
            return self.student(mixture)
        else:
            # 完整蒸馏返回中间特征
            student_outputs = self.student(mixture, return_intermediate=True)
            
            # 应用适配层
            if 'encoder' in self.adapters:
                student_outputs['encoder_output'] = self.adapters['encoder'](
                    student_outputs['encoder_output']
                )
            
            if 'tcn' in self.adapters:
                student_outputs['tcn_features'] = [
                    self.adapters['tcn'](feat) for feat in student_outputs['tcn_features']
                ]
            
            return student_outputs
    
    def _compute_loss(self, student_outputs, teacher_outputs, targets):
        """计算损失"""
        if isinstance(self.criterion, BasicDistillationLoss):
            # 基础蒸馏
            return self.criterion(student_outputs, teacher_outputs, targets)
        else:
            # 完整蒸馏
            return self.criterion(student_outputs, teacher_outputs, targets)
    
    def validate(self) -> Dict[str, float]:
        """验证模型"""
        self.student.eval()
        self.teacher.eval()
        
        val_losses = {
            'total_loss': 0.0,
            'L_spec': 0.0,
            'L_enc': 0.0,
            'L_tcn': 0.0,
            'L_mask': 0.0,
            'L_task': 0.0
        }
        
        all_metrics = []
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in self.val_loader:
                # 数据格式转换（字典格式 → tensor格式）
                mixture = batch['mix'].to(self.device)
                targets = torch.stack(batch['ref'], dim=1).to(self.device)
                
                # 教师模型
                if isinstance(self.criterion, BasicDistillationLoss):
                    teacher_output = self.teacher(mixture)
                else:
                    teacher_outputs = self.teacher(mixture, return_intermediate=True)
                
                # 学生模型
                student_outputs = self._forward_student(mixture)
                
                # 计算损失
                loss, loss_dict = self._compute_loss(
                    student_outputs, teacher_outputs if not isinstance(self.criterion, BasicDistillationLoss) else teacher_output,
                    targets
                )
                
                # 累积损失
                for key, value in loss_dict.items():
                    if key in val_losses:
                        val_losses[key] += value
                
                # 计算评估指标 (SI-SDR, SDR, SI-SDRi)
                if isinstance(student_outputs, dict):
                    estimated = student_outputs['output']
                else:
                    estimated = student_outputs
                
                # 从配置读取评估设置
                enabled_metrics = self.eval_config.get('metrics', {}).get('enabled', ['si_sdr', 'sdr', 'sir', 'sar'])
                use_asteroid = self.eval_config.get('use_asteroid', True)
                include_stoi = self.eval_config.get('include_stoi', False)
                
                # 计算指标（使用 Asteroid 自动 PIT + 误差分解）
                metrics = calculate_metrics(
                    estimated.cpu(), 
                    targets.cpu(),
                    mixture.cpu(),  # 用于计算SI-SDRi
                    metrics_list=enabled_metrics,
                    use_asteroid=use_asteroid,
                    include_stoi=include_stoi
                )
                all_metrics.append(metrics)
        
        # 计算平均损失
        for key in val_losses:
            val_losses[key] /= num_batches
        
        # 计算平均指标
        avg_metrics = {}
        if len(all_metrics) > 0:
            for key in all_metrics[0].keys():
                avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        val_losses.update(avg_metrics)
        
        return val_losses
    
    def train(self):
        """完整训练流程"""
        num_epochs = self.config['training']['num_epochs']
        
        self.logger.info(f"开始蒸馏训练，共 {num_epochs} epochs")
        
        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # 更新损失权重
            self._update_loss_weights()
            
            # 训练一个epoch
            train_losses = self.train_epoch()
            
            # 验证
            val_losses = self.validate()
            
            epoch_time = time.time() - epoch_start_time
            
            # 记录损失
            self.train_losses.append(train_losses)
            self.val_losses.append(val_losses)
            
            # 打印日志
            self._log_epoch(train_losses, val_losses, epoch_time)
            
            # 更新学习率
            if self.scheduler is not None:
                self.scheduler.step(val_losses['total_loss'])
            
            # 保存模型
            self._save_checkpoint(val_losses)
            
            # 早停检查
            if self._check_early_stopping(val_losses):
                self.logger.info(f"早停触发，训练结束于 epoch {epoch}")
                break
        
        # 保存最终的可视化图表
        self._save_final_plots()
        
        self.logger.info("蒸馏训练完成！")
    
    def _update_loss_weights(self):
        """动态调整损失权重"""
        if isinstance(self.criterion, BasicDistillationLoss):
            return
        
        distill_config = self.config['distillation']
        schedule_config = distill_config.get('weight_schedule', {})
        
        if not schedule_config.get('enabled', False):
            return
        
        start_epoch = schedule_config.get('start_epoch', 50)
        end_epoch = schedule_config.get('end_epoch', 200)
        
        if self.current_epoch < start_epoch or self.current_epoch > end_epoch:
            return
        
        # 计算进度
        progress = (self.current_epoch - start_epoch) / (end_epoch - start_epoch)
        
        # 更新权重
        self.criterion.update_weights(self.current_epoch, end_epoch)
        
        if self.current_epoch % 10 == 0:
            weights = self.criterion.get_current_weights()
            self.logger.info(f"损失权重更新: {weights}")
    
    def _log_epoch(self, train_losses, val_losses, epoch_time):
        """记录epoch信息"""
        log_msg = f"\nEpoch {self.current_epoch}/{self.config['training']['num_epochs']} "
        log_msg += f"({epoch_time:.2f}s)\n"
        log_msg += f"训练损失: {train_losses['total_loss']:.4f}\n"
        log_msg += f"验证损失: {val_losses['total_loss']:.4f}\n"
        
        # 评估指标（根据配置动态显示）
        enabled_metrics = self.eval_config.get('metrics', {}).get('enabled', ['si_sdr', 'sdr', 'sir', 'sar'])
        
        for metric_name in enabled_metrics:
            if metric_name in val_losses:
                value = val_losses[metric_name]
                if metric_name == 'stoi':
                    log_msg += f"STOI: {value:.4f} (可懂度)\n"
                elif metric_name == 'sir':
                    log_msg += f"SIR: {value:.2f} dB (串扰抑制)\n"
                elif metric_name == 'sar':
                    log_msg += f"SAR: {value:.2f} dB (算法失真)\n"
                else:
                    log_msg += f"{metric_name.upper()}: {value:.2f} dB\n"
        
        # SI-SDRi 如果有也显示
        if 'si_sdri' in val_losses:
            log_msg += f"SI-SDRi: {val_losses['si_sdri']:.2f} dB\n"
        
        current_lr = self.optimizer.param_groups[0]['lr']
        log_msg += f"学习率: {current_lr:.6f}\n"
        
        self.logger.info(log_msg)
    
    def _save_checkpoint(self, val_losses):
        """保存检查点"""
        checkpoint_dir = self.config['logging']['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 保存最新模型
        if self.current_epoch % self.config['logging'].get('save_interval', 5) == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"checkpoint_epoch_{self.current_epoch}.pth"
            )
            torch.save({
                'epoch': self.current_epoch,
                'student_state_dict': self.student.state_dict(),
                'adapter_state_dict': self.adapters.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
            }, checkpoint_path)
            self.logger.info(f"保存检查点: {checkpoint_path}")
        
        # 保存最佳模型
        if 'si_sdr' in val_losses and val_losses['si_sdr'] > self.best_si_sdr:
            self.best_si_sdr = val_losses['si_sdr']
            best_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save({
                'epoch': self.current_epoch,
                'student_state_dict': self.student.state_dict(),
                'adapter_state_dict': self.adapters.state_dict(),
                'best_si_sdr': self.best_si_sdr,
            }, best_path)
            self.logger.info(f"保存最佳模型: SI-SDR={self.best_si_sdr:.2f} dB")
    
    def _check_early_stopping(self, val_losses) -> bool:
        """检查是否早停"""
        if val_losses['total_loss'] < self.best_val_loss:
            self.best_val_loss = val_losses['total_loss']
            self.early_stop_counter = 0
            return False
        else:
            self.early_stop_counter += 1
            if self.early_stop_counter >= self.early_stop_patience:
                return True
            return False
    
    def _save_final_plots(self):
        """保存最终的可视化图表"""
        result_dir = self.config['logging'].get('result_dir', 'results')
        os.makedirs(result_dir, exist_ok=True)
        
        try:
            # 绘制蒸馏训练指标图（使用配置中的指标）
            plot_metric_list = self.eval_config.get('plot_metrics', None)
            plot_path = os.path.join(result_dir, 'distillation_metrics.png')
            plot_distillation_metrics(self.train_losses, self.val_losses, plot_path, plot_metrics=plot_metric_list)
            self.logger.info(f"可视化图表已保存至: {plot_path}")
        except Exception as e:
            self.logger.warning(f"保存可视化图表失败: {e}")

