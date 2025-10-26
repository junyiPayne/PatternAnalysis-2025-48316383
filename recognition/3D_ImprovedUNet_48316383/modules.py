# modules.py

# 导入您的深度学习框架
# 如果使用 PyTorch:
import torch
import torch.nn as nn
# 如果使用 TensorFlow:
# import tensorflow as tf
# from tensorflow.keras.layers import *
# from tensorflow.keras.models import Model

# --- 基础卷积块 ---
class ConvBlock3D(nn.Module):
    """
    带残差连接的 3D 卷积块
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

        # 如果输入和输出通道数不同，使用 1x1 卷积调整通道数
        self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)  # 残差分支
        out = self.double_conv(x)   # 主分支
        out += identity             # 残差连接
        return self.relu(out)       # 激活函数

# --- 标准 3D U-Net ---
class UNet3D(nn.Module):
    """
    改进的 3D U-Net 模型（带残差连接）
    """
    def __init__(self, in_channels=1, num_classes=6, base_filters=16):
        super().__init__()
        f = base_filters  # 基础通道数

        # 编码器
        self.enc1 = ConvBlock3D(in_channels, f)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = ConvBlock3D(f, f * 2)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = ConvBlock3D(f * 2, f * 4)
        self.pool3 = nn.MaxPool3d(2)

        # 瓶颈层
        self.bottleneck = ConvBlock3D(f * 4, f * 8)

        # 解码器
        self.up3 = nn.ConvTranspose3d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock3D(f * 8, f * 4)

        self.up2 = nn.ConvTranspose3d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(f * 4, f * 2)

        self.up1 = nn.ConvTranspose3d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(f * 2, f)

        # 输出层
        self.out_conv = nn.Conv3d(f, num_classes, kernel_size=1)

    def forward(self, x):
        # 编码器
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        # 瓶颈层
        b = self.bottleneck(p3)

        # 解码器
        d3 = self.up3(b)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.dec1(d1)

        # 输出
        out = self.out_conv(d1)
        return out