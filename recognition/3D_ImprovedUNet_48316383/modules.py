# modules.py

# 导入您的深度学习框架
# 如果使用 PyTorch:
import torch
import torch.nn as nn
# 如果使用 TensorFlow:
# import tensorflow as tf
# from tensorflow.keras.layers import *
# from tensorflow.keras.models import Model

# --- 辅助模块：3D 卷积块 ---
class ConvBlock3D(nn.Module):
    """
    标准的 3D 卷积块：Conv3D -> BatchNorm3D -> ReLU
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

# --- 核心模块：3D UNet 模型 ---
class UNet3D(nn.Module):
    """
    基于 3D 卷积的 U-Net 模型
    """
    def __init__(self, in_channels=1, num_classes=6, base_filters=16):  # 改为6个类别
        super().__init__()
        f = base_filters

        # 编码器部分 (下采样路径)
        self.enc1 = ConvBlock3D(in_channels, f)
        self.pool1 = nn.MaxPool3d(2)
        
        self.enc2 = ConvBlock3D(f, f * 2)
        self.pool2 = nn.MaxPool3d(2)
        
        self.enc3 = ConvBlock3D(f * 2, f * 4)
        self.pool3 = nn.MaxPool3d(2)
        
        # 瓶颈层 (Bottleneck)
        self.bottleneck = ConvBlock3D(f * 4, f * 8)
        
        # 解码器部分 (上采样路径)
        # 上采样 + 3D 卷积，然后与跳跃连接合并
        self.up3 = nn.ConvTranspose3d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock3D(f * 8, f * 4) # 注意：输入通道是跳跃连接和上采样的总和
        
        self.up2 = nn.ConvTranspose3d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(f * 4, f * 2)
        
        self.up1 = nn.ConvTranspose3d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(f * 2, f)

        # 最终输出层：将通道数映射到类别数
        self.out_conv = nn.Conv3d(f, num_classes, kernel_size=1)

    def forward(self, x):
        # 编码器 (下采样)
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        # 瓶颈层
        b = self.bottleneck(p3)

        # 解码器 (上采样)
        d3 = self.up3(b)
        # 裁剪/填充并拼接跳跃连接 (e3)
        d3 = torch.cat((e3, d3), dim=1) # 拼接通道维度
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.dec1(d1)
        
        # 最终输出
        out = self.out_conv(d1)
        return out