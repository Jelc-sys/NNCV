import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet34_Weights


class Model(nn.Module):
    """ 
    A U-net architecture with residual connections and a pretrained ResNet encoder for image sgmentation.
    """
    def __init__(self, in_channels=3, n_classes=19, dropout=0.0):
        
        super(Model, self).__init__()

        # Import ResNet backbone
        weights = ResNet34_Weights.IMAGENET1K_V1
        self.backbone = models.resnet34(weights=weights)
        encoder_out_channels = [64, 64, 128, 256, 512]
        decoder_out_channels = [512, 256, 128, 64]
        upsampling_out_channels = [128, 256]

        # Extract ResNet layers
        self.conv1 = nn.Sequential(self.backbone.conv1, self.backbone.bn1, self.backbone.relu, self.backbone.maxpool)
        self.layer1 = self.backbone.layer1
        self.layer2 = self.backbone.layer2
        self.layer3 = self.backbone.layer3
        self.layer4 = self.backbone.layer4

        # Define decoder
        self.up1 = Up(encoder_out_channels[4] + encoder_out_channels[3], decoder_out_channels[0], dropout=dropout) # Input from layer4, skip from layer3
        self.up2 = Up(512 + encoder_out_channels[2], decoder_out_channels[1], dropout=dropout) # Input from Up1, skip from layer2
        self.up3 = Up(256 + encoder_out_channels[1], decoder_out_channels[2], dropout=dropout) # Input from Up2, skip from layer1
        self.up4 = Up(128 + encoder_out_channels[0], decoder_out_channels[3], dropout=dropout) # Input from Up3, skip from conv1+maxpool

        # 2x (upsampling + conv) to match 256x256 output
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DoubleConv(decoder_out_channels[3], 64, dropout=dropout)
        )

        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DoubleConv(64, 64, dropout=dropout)
        )


        # Final output conv layer
        self.outc = (OutConv(decoder_out_channels[3], n_classes))
        

    def forward(self, x):
        x_in = x
        enc1 = self.conv1(x)
        enc2 = self.layer1(enc1)
        enc3 = self.layer2(enc2)
        enc4 = self.layer3(enc3)

        bottleneck = self.layer4(enc4)

        x = self.up1(bottleneck, enc4)
        x = self.up2(x, enc3)
        x = self.up3(x, enc2)
        x = self.up4(x, enc1)
        
        #x = self.upsample1(x)
        #x = self.upsample2(x)


        logits = self.outc(x)

        output_size = (x_in.size(2), x_in.size(3)) # Match input spatial dimensions (assumed 256x256)
        logits_upsampled = F.interpolate(logits, size=output_size, mode='bilinear', align_corners=False)

        return logits_upsampled
        


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, dropout=0.0, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.double_conv(x)
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv_block(x)
        residual = self.shortcut(x)
        return self.relu(out + residual)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ResidualBlock(in_channels, out_channels, dropout)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then residual block"""

    def __init__(self, in_channels, out_channels, dropout=0.0, bilinear=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ResidualBlock(in_channels, out_channels, dropout)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)

        # If x1 and x2 do not match size
        delta_y = x2.size()[2] - x1.size()[2]
        delta_x = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [delta_x // 2, delta_x - delta_x // 2,
                                    delta_y // 2, delta_y - delta_y // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)