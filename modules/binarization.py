"""
二值化相关模块
实现二值神经网络的核心组件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryActivation(torch.autograd.Function):
    """
    二值化激活函数
    前向传播: sign(x) -> {-1, +1}
    反向传播: 使用直通估计器 (STE, Straight-Through Estimator)
    """
    
    @staticmethod
    def forward(ctx, input):
        """
        前向传播: 二值化
        Args:
            input: 输入张量
        Returns:
            output: 二值化后的张量 {-1, +1}
        """
        ctx.save_for_backward(input)
        # sign函数: x >= 0 -> +1, x < 0 -> -1
        output = torch.sign(input)
        # 处理0值: 将0映射到+1
        output[output == 0] = 1
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播: 使用直通估计器 (STE)
        梯度近似: hardtanh的导数
        Args:
            grad_output: 上游梯度
        Returns:
            grad_input: 近似梯度
        """
        input, = ctx.saved_tensors
        # hardtanh: 当 |x| <= 1 时梯度为1，否则为0
        grad_input = grad_output.clone()
        grad_input[torch.abs(input) > 1] = 0
        return grad_input


class BinaryConv1d(nn.Module):
    """
    二值化1D卷积层
    实现XNOR-Net风格的二值卷积
    
    前向传播:
        1. 二值化权重: W^b = sign(W)
        2. 二值化激活: A^b = sign(A)
        3. 计算缩放因子: α = ||W||_1 / n, β = ||A||_1 / m
        4. 二值卷积: Y = Conv(A^b, W^b) ⊙ α ⊙ β
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, dilation=1, groups=1, bias=False):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            stride: 步长
            padding: 填充
            dilation: 膨胀率
            groups: 分组卷积
            bias: 是否使用偏置
        """
        super(BinaryConv1d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # 全精度权重 (用于训练)
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, self.kernel_size[0])
        )
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # 初始化权重
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, input):
        """
        前向传播
        Args:
            input: [B, C_in, T] - 输入特征 (全精度)
        Returns:
            output: [B, C_out, T] - 输出特征 (全精度，但经过二值化计算)
        """
        # 1. 计算激活值缩放因子 β
        # β = ||A||_1 / m (每个样本的平均绝对值)
        beta = torch.mean(torch.abs(input), dim=(1, 2), keepdim=True)
        
        # 2. 二值化激活值
        binary_input = BinaryActivation.apply(input)
        
        # 3. 计算权重缩放因子 α
        # α = ||W||_1 / n (每个卷积核的平均绝对值)
        alpha = torch.mean(torch.abs(self.weight), dim=(1, 2), keepdim=True)
        
        # 4. 二值化权重
        binary_weight = BinaryActivation.apply(self.weight)
        
        # 5. 二值卷积
        output = F.conv1d(
            binary_input, 
            binary_weight, 
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )
        
        # 6. 应用缩放因子: Y = Conv(A^b, W^b) ⊙ α ⊙ β
        # alpha: [C_out, 1, 1] -> [1, C_out, 1]
        # beta: [B, 1, 1] -> [B, 1, 1]
        alpha = alpha.view(1, self.out_channels, 1)
        output = output * alpha * beta
        
        # 7. 添加偏置
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1)
        
        return output
    
    def extra_repr(self):
        """打印层信息"""
        return (f'in_channels={self.in_channels}, '
                f'out_channels={self.out_channels}, '
                f'kernel_size={self.kernel_size[0]}, '
                f'stride={self.stride}, '
                f'padding={self.padding}, '
                f'dilation={self.dilation}, '
                f'groups={self.groups}, '
                f'bias={self.bias is not None}')


class BinaryDepthwiseConv1d(nn.Module):
    """
    二值化深度卷积 (Depthwise Convolution)
    用于TCN中的深度可分离卷积
    """
    
    def __init__(self, channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        """
        Args:
            channels: 通道数 (输入=输出)
            kernel_size: 卷积核大小
            stride: 步长
            padding: 填充
            dilation: 膨胀率
            bias: 是否使用偏置
        """
        super(BinaryDepthwiseConv1d, self).__init__()
        
        # 使用groups=channels实现深度卷积
        self.conv = BinaryConv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=channels,  # 关键：每个通道独立卷积
            bias=bias
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, C, T]
        Returns:
            output: [B, C, T]
        """
        return self.conv(x)


def test_binarization():
    """测试二值化模块"""
    print("Testing Binarization Modules...")
    print("=" * 80)
    
    # 测试参数
    batch_size = 4
    in_channels = 32
    out_channels = 64
    seq_len = 100
    kernel_size = 3
    
    # 1. 测试BinaryActivation
    print("\n1. Testing BinaryActivation...")
    x = torch.randn(batch_size, in_channels, seq_len)
    x_binary = BinaryActivation.apply(x)
    
    print(f"  Input range: [{x.min():.2f}, {x.max():.2f}]")
    print(f"  Binary output: unique values = {x_binary.unique().tolist()}")
    assert set(x_binary.unique().tolist()) == {-1.0, 1.0}, "Binary values should be {-1, 1}"
    print("  ✓ BinaryActivation passed!")
    
    # 2. 测试BinaryConv1d
    print("\n2. Testing BinaryConv1d...")
    x = torch.randn(batch_size, in_channels, seq_len)
    binary_conv = BinaryConv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding=(kernel_size - 1) // 2
    )
    
    # 前向传播
    output = binary_conv(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected shape: [{batch_size}, {out_channels}, {seq_len}]")
    assert output.shape == (batch_size, out_channels, seq_len), "Shape mismatch!"
    print("  ✓ BinaryConv1d passed!")
    
    # 3. 测试反向传播
    print("\n3. Testing backward pass...")
    x = torch.randn(batch_size, in_channels, seq_len, requires_grad=True)
    output = binary_conv(x)
    loss = output.sum()
    loss.backward()
    
    print(f"  Gradient exists: {x.grad is not None}")
    print(f"  Gradient shape: {x.grad.shape}")
    print(f"  Gradient range: [{x.grad.min():.4f}, {x.grad.max():.4f}]")
    assert x.grad is not None, "Gradient should exist!"
    print("  ✓ Backward pass passed!")
    
    # 4. 测试BinaryDepthwiseConv1d
    print("\n4. Testing BinaryDepthwiseConv1d...")
    x = torch.randn(batch_size, in_channels, seq_len)
    binary_dw_conv = BinaryDepthwiseConv1d(
        channels=in_channels,
        kernel_size=kernel_size,
        padding=(kernel_size - 1) // 2
    )
    
    output = binary_dw_conv(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    assert output.shape == x.shape, "Depthwise conv should preserve shape!"
    print("  ✓ BinaryDepthwiseConv1d passed!")
    
    # 5. 参数量对比
    print("\n5. Comparing parameters...")
    
    # 标准卷积
    std_conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size-1)//2)
    std_params = sum(p.numel() for p in std_conv.parameters())
    
    # 二值卷积（参数数量相同，但存储只需1bit）
    binary_params = sum(p.numel() for p in binary_conv.parameters())
    
    print(f"  Standard Conv1d parameters: {std_params:,}")
    print(f"  Binary Conv1d parameters: {binary_params:,}")
    print(f"  Memory reduction (1-bit vs 32-bit): {32:.1f}x")
    
    print("\n" + "=" * 80)
    print("All tests passed! ✓")


if __name__ == "__main__":
    test_binarization()
