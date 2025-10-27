import torch
import torch.nn as nn


# BUILDING BLOCKS
class ConvBlock3D(nn.Module):
    """
    Post-Activation Residual Convolution Block for 3D U-Net
    
    Architecture:
        Input --> Conv3D --> InstanceNorm3D --> LeakyReLU --> 
        Conv3D --> InstanceNorm3D --> LeakyReLU --> (+) Residual --> LeakyReLU --> Output
                                                      ↑
                                                   Shortcut (1x1 Conv if needed)
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    
    Returns:
        Tensor with shape (B, out_channels, D, H, W)
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock3D, self).__init__()
        
        # Main convolution path (two 3x3x3 convolutions)
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.01, inplace=False),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.01, inplace=False)
        )
        
        # Shortcut connection (1x1 conv if channel dimensions don't match)
        self.shortcut = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        
        # Final activation after residual addition
        self.final_activation = nn.LeakyReLU(0.01, inplace=False)

    def forward(self, x):
        """
        Forward pass with residual connection
        
        Args:
            x: Input tensor (B, in_channels, D, H, W)
        
        Returns:
            Output tensor (B, out_channels, D, H, W)
        """
        identity = self.shortcut(x)      # Residual shortcut
        out = self.double_conv(x)        # Main convolution path
        out = out + identity             # Residual addition
        return self.final_activation(out)  # Post-activation


# 3D U-NET MODEL

class UNet3D(nn.Module):
    """
    Improved 3D U-Net with Deeper Architecture
    
    Architecture Overview:
        - 4 Encoder Levels (16 -> 32 -> 64 -> 128 filters)
        - Bottleneck (256 filters)
        - 4 Decoder Levels (128 -> 64 -> 32 -> 16 filters)
        - Skip Connections between encoder and decoder
        - Post-activation residual blocks
        
    Args:
        in_channels (int): Number of input channels (default: 1 for grayscale)
        num_classes (int): Number of output classes
        base_filters (int): Base number of filters (default: 16)
        
    Input Shape:
        (B, in_channels, D, H, W)
        
    Output Shape:
        (B, num_classes, D, H, W)
    """
    def __init__(self, in_channels=1, num_classes=6, base_filters=16):
        super(UNet3D, self).__init__()
        
        f = base_filters  # Base filter count
        
        # ENCODER PATH (Contracting Path)
        
        # Encoder Level 1: (B, 1, D, H, W) -> (B, f, D, H, W)
        self.enc1 = ConvBlock3D(in_channels, f)  # Residual Block
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)  # Downsampling

        # Encoder Level 2: (B, f, D/2, H/2, W/2) -> (B, 2f, D/2, H/2, W/2)
        self.enc2 = ConvBlock3D(f, f * 2)  # Residual Block
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)  # Downsampling

        # Encoder Level 3: (B, 2f, D/4, H/4, W/4) -> (B, 4f, D/4, H/4, W/4)
        self.enc3 = ConvBlock3D(f * 2, f * 4)  # Residual Block
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)  # Downsampling

        # Encoder Level 4: (B, 4f, D/8, H/8, W/8) -> (B, 8f, D/8, H/8, W/8)
        self.enc4 = ConvBlock3D(f * 4, f * 8)  # Residual Block
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)  # Downsampling

        # BOTTLENECK (Deepest Level)
        
        # Bottleneck: (B, 8f, D/16, H/16, W/16) -> (B, 16f, D/16, H/16, W/16)
        self.bottleneck = ConvBlock3D(f * 8, f * 16)  # Residual Block

        # DECODER PATH (Expanding Path)
        
        # Decoder Level 4: (B, 16f, D/16, H/16, W/16) -> (B, 8f, D/8, H/8, W/8)
        self.up4 = nn.ConvTranspose3d(f * 16, f * 8, kernel_size=2, stride=2)  # Upsampling
        self.dec4 = ConvBlock3D(f * 16, f * 8)  # Residual Block (16f from concat)

        # Decoder Level 3: (B, 8f, D/8, H/8, W/8) -> (B, 4f, D/4, H/4, W/4)
        self.up3 = nn.ConvTranspose3d(f * 8, f * 4, kernel_size=2, stride=2)  # Upsampling
        self.dec3 = ConvBlock3D(f * 8, f * 4)  # Residual Block (8f from concat)

        # Decoder Level 2: (B, 4f, D/4, H/4, W/4) -> (B, 2f, D/2, H/2, W/2)
        self.up2 = nn.ConvTranspose3d(f * 4, f * 2, kernel_size=2, stride=2)  # Upsampling
        self.dec2 = ConvBlock3D(f * 4, f * 2)  # Residual Block (4f from concat)

        # Decoder Level 1: (B, 2f, D/2, H/2, W/2) -> (B, f, D, H, W)
        self.up1 = nn.ConvTranspose3d(f * 2, f, kernel_size=2, stride=2)  # Upsampling
        self.dec1 = ConvBlock3D(f * 2, f)  # Residual Block (2f from concat)

        # OUTPUT LAYER
        
        # Final 1x1x1 convolution: (B, f, D, H, W) -> (B, num_classes, D, H, W)
        self.out_conv = nn.Conv3d(f, num_classes, kernel_size=1)

    def forward(self, x):
        """
        Forward pass through the U-Net
        
        Args:
            x: Input tensor (B, in_channels, D, H, W)
            
        Returns:
            Output logits (B, num_classes, D, H, W)
        """
        # ENCODER PATH
        
        # Level 1
        e1 = self.enc1(x)      # Residual Block
        p1 = self.pool1(e1)    # Downsampling

        # Level 2
        e2 = self.enc2(p1)     # Residual Block
        p2 = self.pool2(e2)    # Downsampling

        # Level 3
        e3 = self.enc3(p2)     # Residual Block
        p3 = self.pool3(e3)    # Downsampling

        # Level 4
        e4 = self.enc4(p3)     # Residual Block
        p4 = self.pool4(e4)    # Downsampling

        # BOTTLENECK
        
        b = self.bottleneck(p4)  # Deepest level

        # DECODER PATH        
        # Level 4: Upsample + Skip Connection + Residual Block
        d4 = self.up4(b)                    # Upsampling
        d4 = torch.cat([e4, d4], dim=1)     # Skip connection (concatenate)
        d4 = self.dec4(d4)                  # Residual Block

        # Level 3: Upsample + Skip Connection + Residual Block
        d3 = self.up3(d4)                   # Upsampling
        d3 = torch.cat([e3, d3], dim=1)     # Skip connection (concatenate)
        d3 = self.dec3(d3)                  # Residual Block

        # Level 2: Upsample + Skip Connection + Residual Block
        d2 = self.up2(d3)                   # Upsampling
        d2 = torch.cat([e2, d2], dim=1)     # Skip connection (concatenate)
        d2 = self.dec2(d2)                  # Residual Block

        # Level 1: Upsample + Skip Connection + Residual Block
        d1 = self.up1(d2)                   # Upsampling
        d1 = torch.cat([e1, d1], dim=1)     # Skip connection (concatenate)
        d1 = self.dec1(d1)                  # Residual Block

        # OUTPUT        
        out = self.out_conv(d1)  # Final 1x1x1 convolution
        return out


# MODEL SUMMARY
def print_model_summary(model, input_shape=(1, 1, 64, 128, 128)):
    """
    Print a summary of the model architecture
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (B, C, D, H, W)
    """
    print("="*80)
    print("3D U-Net Model Architecture Summary")
    print("="*80)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Input Shape: {input_shape}")
    
    # Test forward pass
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_shape).to(device)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output Shape: {tuple(output.shape)}")
    print("="*80)


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet3D(in_channels=1, num_classes=6, base_filters=16).to(device)
    
    print_model_summary(model, input_shape=(1, 1, 64, 128, 128))
    
    print("\nModel Architecture:")
    print(model)