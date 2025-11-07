"""
知识蒸馏损失函数模块
实现论文中的5个损失组件：
- L_spec: 输出波形蒸馏损失 (MSE)
- L_enc: 编码器输出蒸馏损失 (MSE)
- L_tcn: TCN特征图蒸馏损失 (MSE)
- L_mask: 掩码蒸馏损失 (KL散度)
- L_task: 任务损失 (SI-SNR)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List


class DistillationLoss(nn.Module):
    """知识蒸馏损失函数
    
    公式:
        L_total = λ1*L_spec + λ2*L_enc + λ3*L_tcn + λ4*L_mask + λ5*L_task
    
    Args:
        loss_weights: 损失权重 [λ1, λ2, λ3, λ4, λ5]
        use_feature_distillation: 是否使用特征蒸馏
    """
    
    def __init__(self, 
                 loss_weights: List[float] = [0.2, 0.1, 0.1, 0.2, 0.4],
                 use_feature_distillation: bool = True):
        super(DistillationLoss, self).__init__()
        
        assert len(loss_weights) == 5, "需要5个损失权重"
        assert abs(sum(loss_weights) - 1.0) < 1e-5, "损失权重之和应该为1.0"
        
        self.lambda1 = loss_weights[0]  # L_spec
        self.lambda2 = loss_weights[1]  # L_enc
        self.lambda3 = loss_weights[2]  # L_tcn
        self.lambda4 = loss_weights[3]  # L_mask
        self.lambda5 = loss_weights[4]  # L_task
        
        self.use_feature_distillation = use_feature_distillation
        
        # MSE损失
        self.mse_loss = nn.MSELoss()
        
        # KL散度损失
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, 
                student_outputs: Dict[str, torch.Tensor],
                teacher_outputs: Dict[str, torch.Tensor],
                targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            student_outputs: 学生模型输出字典
                - 'output': [B, C, T] 分离的语音波形
                - 'encoder_output': [B, N, K] 编码器输出
                - 'tcn_features': List[[B, H, K]] TCN特征列表
                - 'masks': [B, C, N, K] 掩码
            teacher_outputs: 教师模型输出字典 (同上)
            targets: [B, C, T] 真实目标语音
            
        Returns:
            total_loss: 总损失
            loss_dict: 各分量损失字典
        """
        
        loss_dict = {}
        total_loss = 0.0
        
        # ============ L_spec: 输出波形蒸馏损失 ============
        # 公式: L_spec = MSE(teacher_output, student_output)
        student_output = student_outputs['output']  # [B, C, T]
        teacher_output = teacher_outputs['output']  # [B, C, T]
        
        l_spec = self.mse_loss(student_output, teacher_output)
        loss_dict['L_spec'] = l_spec.item()
        total_loss += self.lambda1 * l_spec
        
        # ============ L_task: 任务损失 (SI-SNR) ============
        # 公式: L_task = -SI-SNR(student_output, targets)
        l_task = -self.si_snr(student_output, targets)
        loss_dict['L_task'] = l_task.item()
        total_loss += self.lambda5 * l_task
        
        # ============ 特征蒸馏损失 (如果启用) ============
        if self.use_feature_distillation:
            
            # L_enc: 编码器输出蒸馏损失
            # 公式: L_enc = MSE(||teacher_enc||_2, ||student_enc||_2)
            if ('encoder_output' in student_outputs and 'encoder_output' in teacher_outputs 
                and teacher_outputs['encoder_output'] is not None):
                student_enc = student_outputs['encoder_output']  # [B, N, K]
                teacher_enc = teacher_outputs['encoder_output']  # [B, N, K]
                
                # L2归一化
                student_enc_norm = F.normalize(student_enc, p=2, dim=1)
                teacher_enc_norm = F.normalize(teacher_enc, p=2, dim=1)
                
                l_enc = self.mse_loss(student_enc_norm, teacher_enc_norm)
                loss_dict['L_enc'] = l_enc.item()
                total_loss += self.lambda2 * l_enc
            else:
                loss_dict['L_enc'] = 0.0
            
            # L_tcn: TCN特征图蒸馏损失
            # 公式: L_tcn = (1/n) * Σ MSE(||teacher_tcn_i||_2, ||student_tcn_i||_2)
            if ('tcn_features' in student_outputs and 'tcn_features' in teacher_outputs 
                and len(teacher_outputs['tcn_features']) > 0):
                student_tcn_list = student_outputs['tcn_features']
                teacher_tcn_list = teacher_outputs['tcn_features']
                
                l_tcn = 0.0
                num_layers = min(len(student_tcn_list), len(teacher_tcn_list))
                
                for i in range(num_layers):
                    student_tcn = student_tcn_list[i]  # [B, H, K]
                    teacher_tcn = teacher_tcn_list[i]  # [B, H, K]
                    
                    # L2归一化
                    student_tcn_norm = F.normalize(student_tcn, p=2, dim=1)
                    teacher_tcn_norm = F.normalize(teacher_tcn, p=2, dim=1)
                    
                    l_tcn += self.mse_loss(student_tcn_norm, teacher_tcn_norm)
                
                l_tcn = l_tcn / num_layers if num_layers > 0 else 0.0
                loss_dict['L_tcn'] = l_tcn.item() if isinstance(l_tcn, torch.Tensor) else l_tcn
                total_loss += self.lambda3 * l_tcn
            else:
                loss_dict['L_tcn'] = 0.0
            
            # L_mask: 掩码蒸馏损失
            # 公式: L_mask = (1/C) * Σ KL(teacher_mask_i || student_mask_i)
            if ('masks' in student_outputs and 'masks' in teacher_outputs 
                and teacher_outputs['masks'] is not None):
                student_masks = student_outputs['masks']  # [B, C, N, K]
                teacher_masks = teacher_outputs['masks']  # [B, C, N, K]
                
                # 将掩码reshape为 [B*C, N*K]
                B, C, N, K = student_masks.shape
                student_masks_flat = student_masks.view(B * C, -1)
                teacher_masks_flat = teacher_masks.view(B * C, -1)
                
                # 添加数值稳定性
                eps = 1e-8
                teacher_masks_flat = torch.clamp(teacher_masks_flat, eps, 1.0 - eps)
                student_masks_flat = torch.clamp(student_masks_flat, eps, 1.0 - eps)
                
                # KL散度: KL(P||Q) = Σ P * log(P/Q)
                # PyTorch的KLDivLoss要求输入为log_prob和prob
                student_log_prob = torch.log(student_masks_flat)
                teacher_prob = teacher_masks_flat
                
                l_mask = self.kl_loss(student_log_prob, teacher_prob)
                loss_dict['L_mask'] = l_mask.item()
                total_loss += self.lambda4 * l_mask
            else:
                loss_dict['L_mask'] = 0.0
        
        # 总损失
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def si_snr(self, estimated: torch.Tensor, target: torch.Tensor, 
               eps: float = 1e-8) -> torch.Tensor:
        """计算SI-SNR (Scale-Invariant Signal-to-Noise Ratio)
        
        公式:
            s_target = <s', s>s / ||s||^2
            e_noise = s' - s_target
            SI-SNR = 10 * log10(||s_target||^2 / ||e_noise||^2)
        
        Args:
            estimated: [B, C, T] 估计的语音
            target: [B, C, T] 目标语音
            eps: 数值稳定性参数
            
        Returns:
            SI-SNR值 (标量)
        """
        # 确保维度一致
        assert estimated.shape == target.shape, "estimated和target的shape必须相同"
        
        # Zero-mean normalization
        estimated = estimated - torch.mean(estimated, dim=-1, keepdim=True)
        target = target - torch.mean(target, dim=-1, keepdim=True)
        
        # <s', s>
        dot_product = torch.sum(estimated * target, dim=-1, keepdim=True)
        
        # ||s||^2
        target_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + eps
        
        # s_target = <s', s>s / ||s||^2
        projection = dot_product * target / target_energy
        
        # e_noise = s' - s_target
        noise = estimated - projection
        
        # SI-SNR = 10 * log10(||s_target||^2 / ||e_noise||^2)
        signal_power = torch.sum(projection ** 2, dim=-1) + eps
        noise_power = torch.sum(noise ** 2, dim=-1) + eps
        
        si_snr = 10 * torch.log10(signal_power / noise_power)
        
        # 返回批次平均
        return torch.mean(si_snr)
    
    def update_weights(self, epoch: int, total_epochs: int):
        """动态调整损失权重
        
        策略: 从学习教师知识 → 学习真实目标
            - lambda5 (L_task): 0.4 → 0.8
            - lambda1 (L_spec): 0.2 → 0.1
        
        Args:
            epoch: 当前epoch
            total_epochs: 总epoch数
        """
        progress = epoch / total_epochs
        
        # 线性调整
        self.lambda5 = 0.4 + 0.4 * progress  # 0.4 → 0.8
        self.lambda1 = 0.2 - 0.1 * progress  # 0.2 → 0.1
        
        # 重新归一化
        total_weight = self.lambda1 + self.lambda2 + self.lambda3 + self.lambda4 + self.lambda5
        self.lambda1 /= total_weight
        self.lambda2 /= total_weight
        self.lambda3 /= total_weight
        self.lambda4 /= total_weight
        self.lambda5 /= total_weight
    
    def get_current_weights(self) -> Dict[str, float]:
        """获取当前损失权重"""
        return {
            'lambda1_spec': self.lambda1,
            'lambda2_enc': self.lambda2,
            'lambda3_tcn': self.lambda3,
            'lambda4_mask': self.lambda4,
            'lambda5_task': self.lambda5
        }


class BasicDistillationLoss(nn.Module):
    """基础知识蒸馏损失 (仅 L_spec + L_task)
    
    用于高优先级实现，只包含输出波形蒸馏和任务损失
    """
    
    def __init__(self, lambda_spec: float = 0.3, lambda_task: float = 0.7):
        super(BasicDistillationLoss, self).__init__()
        
        assert abs(lambda_spec + lambda_task - 1.0) < 1e-5, "权重之和应该为1.0"
        
        self.lambda_spec = lambda_spec
        self.lambda_task = lambda_task
        
        self.mse_loss = nn.MSELoss()
    
    def forward(self, 
                student_output: torch.Tensor,
                teacher_output: torch.Tensor,
                targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            student_output: [B, C, T] 学生模型输出
            teacher_output: [B, C, T] 教师模型输出
            targets: [B, C, T] 真实目标
            
        Returns:
            total_loss: 总损失
            loss_dict: 损失字典
        """
        
        loss_dict = {}
        
        # L_spec: 输出波形蒸馏
        l_spec = self.mse_loss(student_output, teacher_output)
        loss_dict['L_spec'] = l_spec.item()
        
        # L_task: SI-SNR任务损失
        l_task = -self.si_snr(student_output, targets)
        loss_dict['L_task'] = l_task.item()
        
        # 总损失
        total_loss = self.lambda_spec * l_spec + self.lambda_task * l_task
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def si_snr(self, estimated: torch.Tensor, target: torch.Tensor, 
               eps: float = 1e-8) -> torch.Tensor:
        """SI-SNR计算 (同上)"""
        # Zero-mean normalization
        estimated = estimated - torch.mean(estimated, dim=-1, keepdim=True)
        target = target - torch.mean(target, dim=-1, keepdim=True)
        
        # <s', s>
        dot_product = torch.sum(estimated * target, dim=-1, keepdim=True)
        
        # ||s||^2
        target_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + eps
        
        # s_target = <s', s>s / ||s||^2
        projection = dot_product * target / target_energy
        
        # e_noise = s' - s_target
        noise = estimated - projection
        
        # SI-SNR
        signal_power = torch.sum(projection ** 2, dim=-1) + eps
        noise_power = torch.sum(noise ** 2, dim=-1) + eps
        
        si_snr = 10 * torch.log10(signal_power / noise_power)
        
        return torch.mean(si_snr)
    
    def update_weights(self, epoch: int, total_epochs: int):
        """动态调整权重"""
        progress = epoch / total_epochs
        
        # lambda_task: 0.7 → 0.9
        self.lambda_task = 0.7 + 0.2 * progress
        self.lambda_spec = 1.0 - self.lambda_task
    
    def get_current_weights(self) -> Dict[str, float]:
        """获取当前权重"""
        return {
            'lambda_spec': self.lambda_spec,
            'lambda_task': self.lambda_task
        }

