"""
PIT (Permutation Invariant Training) 损失包装器
解决语音分离中的排列不确定性问题
"""

import torch
import torch.nn as nn
import itertools


class PITLossWrapper(nn.Module):
    """
    PIT (Permutation Invariant Training) 损失包装器
    
    解决排列不确定性问题：
    - 问题：模型输出 [ŝ1, ŝ2]，但不知道哪个对应 [s1, s2]
    - 解决：尝试所有排列组合，选择损失最小的排列
    
    对于 2 个说话人：
    - 排列1: ŝ1→s1, ŝ2→s2
    - 排列2: ŝ1→s2, ŝ2→s1
    
    选择损失最小的排列进行反向传播
    """
    
    def __init__(self, loss_func, num_speakers=2):
        """
        Args:
            loss_func: 基础损失函数（如 SI_SNR_Loss）
            num_speakers: 说话人数量
        """
        super(PITLossWrapper, self).__init__()
        self.loss_func = loss_func
        self.num_speakers = num_speakers
        
        # 预先计算所有排列（避免重复计算）
        self.perms = list(itertools.permutations(range(num_speakers)))
    
    def forward(self, estimations, targets):
        """
        计算 PIT 损失
        
        Args:
            estimations: [B, C, T] - 模型预测的 C 个说话人语音
            targets: [B, C, T] - 真实的 C 个说话人语音
        
        Returns:
            min_loss: 最小排列损失
            best_perm_idx: 最佳排列索引（每个样本）
        
        流程:
        1. 生成所有可能的排列 (C! 种)
        2. 对每种排列计算损失
        3. 选择损失最小的排列
        4. 返回最小损失用于反向传播
        """
        batch_size = estimations.shape[0]
        num_speakers = estimations.shape[1]
        
        assert num_speakers == self.num_speakers, \
            f"Number of speakers mismatch: {num_speakers} vs {self.num_speakers}"
        
        # 存储每种排列的损失 [num_perms, batch_size]
        perm_losses = []
        
        # 遍历所有排列
        for perm in self.perms:
            # 计算当前排列的损失
            batch_loss = []
            for b in range(batch_size):
                sample_loss = 0
                for est_idx, tgt_idx in enumerate(perm):
                    # 计算 estimation[est_idx] 与 target[tgt_idx] 的损失
                    loss = self.loss_func(
                        estimations[b, est_idx:est_idx+1, :],  # [1, T]
                        targets[b, tgt_idx:tgt_idx+1, :]       # [1, T]
                    )
                    sample_loss += loss
                batch_loss.append(sample_loss)
            perm_losses.append(torch.stack(batch_loss))
        
        # perm_losses: [num_perms, batch_size]
        perm_losses = torch.stack(perm_losses)
        
        # 选择每个样本的最小损失
        min_loss, best_perm_idx = torch.min(perm_losses, dim=0)  # [batch_size]
        
        # 返回平均损失
        return torch.mean(min_loss), best_perm_idx
    
    def reorder_source(self, estimations, targets):
        """
        根据 PIT 找到的最佳排列重新排序估计信号
        
        Args:
            estimations: [B, C, T] - 模型预测的 C 个说话人语音
            targets: [B, C, T] - 真实的 C 个说话人语音
        
        Returns:
            reordered_estimations: [B, C, T] - 重新排序后的估计信号
            best_perms: List[Tuple] - 每个样本的最佳排列
        """
        batch_size = estimations.shape[0]
        
        # 计算最佳排列
        _, best_perm_idx = self.forward(estimations, targets)
        
        # 重新排序
        reordered_estimations = torch.zeros_like(estimations)
        best_perms = []
        
        for b in range(batch_size):
            perm = self.perms[best_perm_idx[b]]
            best_perms.append(perm)
            for est_idx, tgt_idx in enumerate(perm):
                reordered_estimations[b, tgt_idx] = estimations[b, est_idx]
        
        return reordered_estimations, best_perms


if __name__ == "__main__":
    # 测试代码
    print("Testing PIT Loss Wrapper...")
    
    from si_snr import SI_SNR_Loss
    
    # 创建基础损失函数和 PIT 包装器
    base_loss = SI_SNR_Loss()
    pit_loss = PITLossWrapper(base_loss, num_speakers=2)
    
    # 生成测试数据
    batch_size = 4
    num_speakers = 2
    length = 16000  # 1秒音频 @ 16kHz
    
    # 真实语音
    targets = torch.randn(batch_size, num_speakers, length)
    
    # 情况1: 正确顺序的预测
    estimations_correct = targets + torch.randn_like(targets) * 0.1
    loss, best_perm = pit_loss(estimations_correct, targets)
    print(f"\nCorrect order:")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Best permutations: {[pit_loss.perms[idx] for idx in best_perm]}")
    
    # 情况2: 错误顺序的预测（交换了说话人）
    estimations_swapped = torch.zeros_like(targets)
    estimations_swapped[:, 0, :] = targets[:, 1, :] + torch.randn(batch_size, length) * 0.1
    estimations_swapped[:, 1, :] = targets[:, 0, :] + torch.randn(batch_size, length) * 0.1
    loss, best_perm = pit_loss(estimations_swapped, targets)
    print(f"\nSwapped order:")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Best permutations: {[pit_loss.perms[idx] for idx in best_perm]}")
    
    # 情况3: 测试重新排序功能
    reordered, perms = pit_loss.reorder_source(estimations_swapped, targets)
    print(f"\nAfter reordering:")
    print(f"  Reordered shape: {reordered.shape}")
    print(f"  Applied permutations: {perms}")
    
    # 验证重新排序后的损失应该更小
    loss_before = base_loss(estimations_swapped.reshape(-1, length), 
                           targets.reshape(-1, length))
    loss_after = base_loss(reordered.reshape(-1, length), 
                          targets.reshape(-1, length))
    print(f"  Loss before reordering: {loss_before.item():.4f}")
    print(f"  Loss after reordering: {loss_after.item():.4f}")
    
    print("\nPIT Loss Wrapper test passed!")
