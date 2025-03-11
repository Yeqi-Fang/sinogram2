import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Double convolution block with batch normalization"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class DownBlock(nn.Module):
    """Downsampling block with maxpool followed by double convolution"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)

class UpBlock(nn.Module):
    """Upsampling block with transpose convolution and double convolution"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        # Use different upsample methods based on preference
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = ConvBlock(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ConvBlock(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Calculate padding for input size mismatch
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, 
                         diffY // 2, diffY - diffY // 2])
        
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """Final 1x1 convolution layer"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """U-Net architecture for sinogram reconstruction"""
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512], bilinear=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        
        # Initial convolution block
        self.inc = ConvBlock(in_channels, features[0])
        
        # Downsampling path
        self.down_blocks = nn.ModuleList()
        in_features = features[0]
        for feature in features[1:]:
            self.down_blocks.append(DownBlock(in_features, feature))
            in_features = feature
        
        # Bottleneck factor is 2 for bilinear interpolation, 1 for transpose conv
        factor = 2 if bilinear else 1
        self.down_blocks.append(DownBlock(features[-1], features[-1] * 2 // factor))
        
        # Upsampling path
        self.up_blocks = nn.ModuleList()
        for feature in reversed(features):
            self.up_blocks.append(UpBlock(feature * 2, feature, bilinear))
        
        # Final convolution
        self.outc = OutConv(features[0], out_channels)
    
    def forward(self, x):
        # Initial feature extraction
        features = [self.inc(x)]
        
        # Downsampling and feature extraction
        for down in self.down_blocks:
            features.append(down(features[-1]))
        
        # Start with the bottleneck features
        x = features[-1]
        
        # Upsampling and concatenation with skip connections
        for i, up in enumerate(self.up_blocks):
            x = up(x, features[-i-2])
        
        # Final projection to output channels
        return self.outc(x)

class ResidualUNet(nn.Module):
    """U-Net with residual connection for incomplete sinogram reconstruction"""
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512], bilinear=True):
        super().__init__()
        self.unet = UNet(in_channels, out_channels, features, bilinear)
    
    def forward(self, x):
        # U-Net predicts the residual (difference between complete and incomplete)
        residual = self.unet(x)
        
        # Add the input to get the complete sinogram
        return x + residual

# Smaller UNet variant for memory efficiency
class SmallUNet(nn.Module):
    """Smaller U-Net architecture for memory efficiency"""
    def __init__(self, in_channels=1, out_channels=1, bilinear=True):
        super().__init__()
        # Use smaller feature maps
        self.model = UNet(in_channels, out_channels, features=[32, 64, 128, 256], bilinear=bilinear)
    
    def forward(self, x):
        return self.model(x)

# Simplified U-Net for faster training
class SimpleUNet(nn.Module):
    """Simplified U-Net with fewer layers for faster training"""
    def __init__(self, in_channels=1, out_channels=1, bilinear=True):
        super().__init__()
        # Use only 3 levels instead of 4
        self.inc = ConvBlock(in_channels, 64)
        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)
        
        self.up1 = UpBlock(256, 128, bilinear)
        self.up2 = UpBlock(128, 64, bilinear)
        self.outc = OutConv(64, out_channels)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        return self.outc(x)

# Factory function to create different U-Net variants
def create_unet_model(model_type="standard", in_channels=1, out_channels=1, bilinear=True):
    """
    Create a specific U-Net model variant.
    
    Args:
        model_type (str): Type of U-Net model to create
                         "standard" - Standard U-Net 
                         "residual" - U-Net with residual connection
                         "small" - UNet with smaller feature maps
                         "simple" - UNet with fewer layers
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        bilinear (bool): Whether to use bilinear upsampling
        
    Returns:
        nn.Module: The requested UNet model
    """
    if model_type == "standard":
        return UNet(in_channels, out_channels, bilinear=bilinear)
    elif model_type == "residual":
        return ResidualUNet(in_channels, out_channels, bilinear=bilinear)
    elif model_type == "small":
        return SmallUNet(in_channels, out_channels, bilinear=bilinear)
    elif model_type == "simple":
        return SimpleUNet(in_channels, out_channels, bilinear=bilinear)
    else:
        raise ValueError(f"Unknown model type: {model_type}")