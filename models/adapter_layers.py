"""
特征适配层模块
用于对齐教师模型和学生模型的中间特征维度
"""

import torch
import torch.nn as nn


class FeatureAdapter(nn.Module):
    """特征适配层
    
    将学生模型的特征维度转换为教师模型的维度，用于特征蒸馏
    
    Args:
        student_dim: 学生模型特征维度
        teacher_dim: 教师模型特征维度
        use_bn: 是否使用批归一化
    """
    
    def __init__(self, student_dim: int, teacher_dim: int, use_bn: bool = False):
        super(FeatureAdapter, self).__init__()
        
        self.student_dim = student_dim
        self.teacher_dim = teacher_dim
        
        # 使用1x1卷积进行维度转换
        self.adapter = nn.Conv1d(student_dim, teacher_dim, kernel_size=1, bias=True)
        
        # 可选的批归一化
        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm1d(teacher_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        # Xavier初始化
        nn.init.xavier_uniform_(self.adapter.weight)
        if self.adapter.bias is not None:
            nn.init.zeros_(self.adapter.bias)
    
    def forward(self, student_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            student_features: [B, student_dim, T] 学生模型特征
            
        Returns:
            adapted_features: [B, teacher_dim, T] 适配后的特征
        """
        # 1x1卷积转换维度
        adapted = self.adapter(student_features)
        
        # 可选的批归一化
        if self.use_bn:
            adapted = self.bn(adapted)
        
        return adapted


class MultiLayerAdapter(nn.Module):
    """多层特征适配器
    
    为多个TCN层创建适配层
    
    Args:
        student_dim: 学生模型特征维度
        teacher_dim: 教师模型特征维度
        num_layers: TCN层数
        use_bn: 是否使用批归一化
    """
    
    def __init__(self, student_dim: int, teacher_dim: int, 
                 num_layers: int, use_bn: bool = False):
        super(MultiLayerAdapter, self).__init__()
        
        self.num_layers = num_layers
        
        # 为每一层创建适配器
        self.adapters = nn.ModuleList([
            FeatureAdapter(student_dim, teacher_dim, use_bn)
            for _ in range(num_layers)
        ])
    
    def forward(self, student_features_list: list) -> list:
        """
        Args:
            student_features_list: List[[B, student_dim, T]] 学生模型特征列表
            
        Returns:
            adapted_features_list: List[[B, teacher_dim, T]] 适配后的特征列表
        """
        adapted_list = []
        
        for i, student_feat in enumerate(student_features_list):
            if i < len(self.adapters):
                adapted_feat = self.adapters[i](student_feat)
                adapted_list.append(adapted_feat)
            else:
                # 如果层数超过适配器数量，直接使用原特征
                adapted_list.append(student_feat)
        
        return adapted_list


class AdapterWithResidual(nn.Module):
    """带残差连接的适配层
    
    当student_dim和teacher_dim相同时，添加残差连接可以保留更多原始信息
    
    Args:
        student_dim: 学生模型特征维度
        teacher_dim: 教师模型特征维度
        use_residual: 是否使用残差连接（仅当维度相同时有效）
    """
    
    def __init__(self, student_dim: int, teacher_dim: int, 
                 use_residual: bool = True):
        super(AdapterWithResidual, self).__init__()
        
        self.student_dim = student_dim
        self.teacher_dim = teacher_dim
        self.use_residual = use_residual and (student_dim == teacher_dim)
        
        # 主适配器
        self.adapter = nn.Sequential(
            nn.Conv1d(student_dim, teacher_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(teacher_dim, teacher_dim, kernel_size=1)
        )
        
        # 残差连接的投影层（如果维度不同）
        if not self.use_residual and student_dim != teacher_dim:
            self.residual_proj = nn.Conv1d(student_dim, teacher_dim, kernel_size=1)
        else:
            self.residual_proj = None
    
    def forward(self, student_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            student_features: [B, student_dim, T]
            
        Returns:
            adapted_features: [B, teacher_dim, T]
        """
        # 主适配路径
        adapted = self.adapter(student_features)
        
        # 残差连接
        if self.use_residual:
            adapted = adapted + student_features
        elif self.residual_proj is not None:
            adapted = adapted + self.residual_proj(student_features)
        
        return adapted


class AttentionAdapter(nn.Module):
    """基于注意力机制的适配层
    
    使用注意力机制自适应地调整特征对齐
    
    Args:
        student_dim: 学生模型特征维度
        teacher_dim: 教师模型特征维度
        reduction: 注意力机制的降维比率
    """
    
    def __init__(self, student_dim: int, teacher_dim: int, reduction: int = 16):
        super(AttentionAdapter, self).__init__()
        
        # 主适配器
        self.adapter = nn.Conv1d(student_dim, teacher_dim, kernel_size=1)
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(teacher_dim, teacher_dim // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(teacher_dim // reduction, teacher_dim, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, student_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            student_features: [B, student_dim, T]
            
        Returns:
            adapted_features: [B, teacher_dim, T]
        """
        # 维度转换
        adapted = self.adapter(student_features)
        
        # 通道注意力
        attention = self.channel_attention(adapted)
        
        # 应用注意力
        adapted = adapted * attention
        
        return adapted


def create_adapter(student_dim: int, teacher_dim: int, 
                   adapter_type: str = 'simple',
                   **kwargs) -> nn.Module:
    """创建适配层的工厂函数
    
    Args:
        student_dim: 学生模型维度
        teacher_dim: 教师模型维度
        adapter_type: 适配器类型 ('simple', 'residual', 'attention')
        **kwargs: 其他参数
        
    Returns:
        adapter: 适配层模块
    """
    
    if adapter_type == 'simple':
        return FeatureAdapter(student_dim, teacher_dim, 
                            use_bn=kwargs.get('use_bn', False))
    
    elif adapter_type == 'residual':
        return AdapterWithResidual(student_dim, teacher_dim,
                                 use_residual=kwargs.get('use_residual', True))
    
    elif adapter_type == 'attention':
        return AttentionAdapter(student_dim, teacher_dim,
                              reduction=kwargs.get('reduction', 16))
    
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")


# 示例使用
if __name__ == "__main__":
    # 测试简单适配器
    adapter = FeatureAdapter(student_dim=128, teacher_dim=256)
    x = torch.randn(4, 128, 1000)  # [B, student_dim, T]
    y = adapter(x)
    print(f"Simple Adapter: {x.shape} -> {y.shape}")
    
    # 测试多层适配器
    multi_adapter = MultiLayerAdapter(student_dim=128, teacher_dim=256, num_layers=7)
    x_list = [torch.randn(4, 128, 1000) for _ in range(7)]
    y_list = multi_adapter(x_list)
    print(f"Multi-Layer Adapter: {len(x_list)} layers -> {len(y_list)} layers")
    
    # 测试残差适配器
    residual_adapter = AdapterWithResidual(student_dim=128, teacher_dim=256)
    x = torch.randn(4, 128, 1000)
    y = residual_adapter(x)
    print(f"Residual Adapter: {x.shape} -> {y.shape}")
    
    # 测试注意力适配器
    attention_adapter = AttentionAdapter(student_dim=128, teacher_dim=256)
    x = torch.randn(4, 128, 1000)
    y = attention_adapter(x)
    print(f"Attention Adapter: {x.shape} -> {y.shape}")

