import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights


class ConvBlock(nn.Module):
    """Convolution + BatchNorm + ReLU block."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class CustomDecoderBlock(nn.Module):
    """
    Decoder block that combines upsampled high-level features with 
    skip-connected low-level encoder features.
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv_skip = nn.Conv2d(skip_channels, out_channels, kernel_size=1) if skip_channels > 0 else None
        self.conv_fuse = nn.Sequential(
            ConvBlock(in_channels + out_channels if skip_channels > 0 else in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        if skip is not None and self.conv_skip is not None:
            skip = self.conv_skip(skip)
            # Ensure spatial dimension match
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv_fuse(x)


class ProposalFreeInstanceSegNet(nn.Module):
    """
    Dense Proposal-Free Instance Segmentation Network for Person + Car on Cityscapes.
    
    Architecture:
      - Backbone: Pretrained torchvision ResNet (e.g. ResNet18/34/50)
      - Decoder: Custom Feature Fusion Decoder with skip connections
      - Semantic Head: Predicts 3 class logits (BG, Person, Car)
      - Center Heatmap Head: Predicts 2 class Gaussian heatmaps (Person, Car)
      - Center Offset Head: Predicts (dx, dy) 2D spatial vectors to instance centroids
    """
    def __init__(self, backbone_name='resnet18', pretrained=True, num_semantic_classes=3, num_center_classes=2):
        super().__init__()
        self.backbone_name = backbone_name

        # Load torchvision pretrained encoder backbone
        if backbone_name == 'resnet18':
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
            channels = [64, 64, 128, 256, 512]
        elif backbone_name == 'resnet34':
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            backbone = models.resnet34(weights=weights)
            channels = [64, 64, 128, 256, 512]
        elif backbone_name == 'resnet50':
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            backbone = models.resnet50(weights=weights)
            channels = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Encoder stages
        self.initial = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu
        )
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1  # C2
        self.layer2 = backbone.layer2  # C3
        self.layer3 = backbone.layer3  # C4
        self.layer4 = backbone.layer4  # C5

        # Custom Decoder Stage
        dec_channels = 128
        self.dec4 = CustomDecoderBlock(channels[4], channels[3], dec_channels)  # C5 + C4 -> (H/16, W/16)
        self.dec3 = CustomDecoderBlock(dec_channels, channels[2], dec_channels) # -> (H/8, W/8)
        self.dec2 = CustomDecoderBlock(dec_channels, channels[1], dec_channels) # -> (H/4, W/4)
        self.dec1 = CustomDecoderBlock(dec_channels, channels[0], dec_channels) # -> (H/2, W/2)

        # Final upsample to full resolution (H, W)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(dec_channels, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ConvBlock(64, 64)
        )

        # Prediction Heads
        # 1. Semantic Head: 3 classes (0: Background, 1: Person, 2: Car)
        self.semantic_head = nn.Sequential(
            ConvBlock(64, 32),
            nn.Conv2d(32, num_semantic_classes, kernel_size=1)
        )

        # 2. Center Heatmap Head: 2 channels (0: Person center, 1: Car center)
        self.center_head = nn.Sequential(
            ConvBlock(64, 32),
            nn.Conv2d(32, num_center_classes, kernel_size=1),
            nn.Sigmoid()
        )

        # 3. Spatial Offset Regression Head: 2 channels (dx, dy)
        self.offset_head = nn.Sequential(
            ConvBlock(64, 32),
            nn.Conv2d(32, 2, kernel_size=1)
        )

    def forward(self, x):
        input_size = x.shape[2:]

        # Encoder forward
        c1 = self.initial(x)                  # H/2, W/2
        x_mp = self.maxpool(c1)               # H/4, W/4
        c2 = self.layer1(x_mp)                # H/4, W/4
        c3 = self.layer2(c2)                  # H/8, W/8
        c4 = self.layer3(c3)                  # H/16, W/16
        c5 = self.layer4(c4)                  # H/32, W/32

        # Decoder forward with skip connections
        d4 = self.dec4(c5, c4)                # H/16, W/16
        d3 = self.dec3(d4, c3)                # H/8, W/8
        d2 = self.dec2(d3, c2)                # H/4, W/4
        d1 = self.dec1(d2, c1)                # H/2, W/2

        feat = self.final_up(d1)              # H, W
        if feat.shape[2:] != input_size:
            feat = F.interpolate(feat, size=input_size, mode='bilinear', align_corners=False)

        # Output predictions
        semantic_logits = self.semantic_head(feat)
        center_heatmap = self.center_head(feat)
        offset_map = self.offset_head(feat)

        return {
            'semantic_logits': semantic_logits, # (B, 3, H, W)
            'center_heatmap': center_heatmap,   # (B, 2, H, W)
            'offset_map': offset_map            # (B, 2, H, W)
        }
