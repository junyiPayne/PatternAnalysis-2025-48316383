import torch
import torch.nn as nn
# --- basic 3D convolution block ---
class ConvBlock3D(nn.Module):
    """
    Post-Activation Residual Block for 3D U-Net
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.01, inplace=False),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.01, inplace=False)
        )
        self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.final_activation = nn.LeakyReLU(0.01, inplace=False)  

    def forward(self, x):
        identity = self.shortcut(x)  
        out = self.double_conv(x)   
        out += identity             
        return self.final_activation(out)  

# --- Standard 3D U-Net ---
class UNet3D(nn.Module):
    """
    Improved 3D U-Net model with residual connections
    """
    def __init__(self, in_channels=1, num_classes=6, base_filters=16):
        super().__init__()
        f = base_filters  # base number of filters

        # Encoder
        self.enc1 = ConvBlock3D(in_channels, f)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = ConvBlock3D(f, f * 2)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = ConvBlock3D(f * 2, f * 4)
        self.pool3 = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = ConvBlock3D(f * 4, f * 8)

        # Decoder
        self.up3 = nn.ConvTranspose3d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock3D(f * 8, f * 4)

        self.up2 = nn.ConvTranspose3d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(f * 4, f * 2)

        self.up1 = nn.ConvTranspose3d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(f * 2, f)

        # Output layer
        self.out_conv = nn.Conv3d(f, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder
        d3 = self.up3(b)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.dec1(d1)

        # Output layer
        out = self.out_conv(d1)
        return out