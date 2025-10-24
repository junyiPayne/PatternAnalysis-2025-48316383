# modules.py

# 导入您的深度学习框架
# 如果使用 PyTorch:
import torch
import torch.nn as nn
import torch.nn.functional as F
# 如果使用 TensorFlow:
# import tensorflow as tf
# from tensorflow.keras.layers import *
# from tensorflow.keras.models import Model

# --- 辅助模块：3D 卷积块 ---
class ConvBlock3D(nn.Module):
    """
    标准的 3D 卷积块：Conv3D -> BatchNorm3D -> ReLU
    """
    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super().__init__()
        # Project 7 难度较高的任务（Hard/Improved UNet）可能需要更复杂的块结构
        # 或考虑使用 Dilated Convolutions (CAN3D) 或残差连接。
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            
            # UNet 通常有第二个卷积层
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            
            # 添加 Dropout (可选，但推荐)
            # nn.Dropout3d(dropout_rate) 
        )

    def forward(self, x):
        return self.double_conv(x)

# --- 核心模块：3D UNet 模型 ---
class UNet3D(nn.Module):
    """
    基于 3D 卷积的 U-Net 模型
    """
    def __init__(self, in_channels=1, num_classes=6):  # 改为6个类别
        super().__init__()
        
        # 编码器部分 (下采样路径)
        self.enc1 = ConvBlock3D(in_channels, 32)
        self.pool1 = nn.MaxPool3d(2)
        
        self.enc2 = ConvBlock3D(32, 64)
        self.pool2 = nn.MaxPool3d(2)
        
        self.enc3 = ConvBlock3D(64, 128)
        self.pool3 = nn.MaxPool3d(2)
        
        # 瓶颈层 (Bottleneck)
        self.bottleneck = ConvBlock3D(128, 256)
        
        # 解码器部分 (上采样路径)
        # 上采样 + 3D 卷积，然后与跳跃连接合并
        self.upconv3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock3D(128 + 128, 128) # 注意：输入通道是跳跃连接和上采样的总和
        
        self.upconv2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(64 + 64, 64)
        
        self.upconv1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(32 + 32, 32)

        # 最终输出层：将通道数映射到类别数
        self.out_conv = nn.Conv3d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # 编码器 (下采样)
        e1 = self.enc1(x)       # (N, 32, D, H, W)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)      # (N, 64, D/2, H/2, W/2)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)      # (N, 128, D/4, H/4, W/4)
        p3 = self.pool3(e3)

        # 瓶颈层
        b = self.bottleneck(p3) # (N, 256, D/8, H/8, W/8)

        # 解码器 (上采样)
        d3 = self.upconv3(b)
        # 裁剪/填充并拼接跳跃连接 (e3)
        d3 = torch.cat((e3, d3), dim=1) # 拼接通道维度
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.dec1(d1)
        
        # 最终输出
        output = self.out_conv(d1)
        
        return output