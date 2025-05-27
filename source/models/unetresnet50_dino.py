import torch.nn as nn
import torch.nn.functional as F
from models import remove_module_prefix
import torch 


class Conv2dReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(Conv2dReLU, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
     
    def forward(self, x):
        return self.block(x)
 
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv1 = Conv2dReLU(in_channels, out_channels)
        self.conv2 = Conv2dReLU(out_channels, out_channels)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
     
    def forward(self, x, skip):
        # Upsample
        x = self.up(x)
         
        # Resize skip connection to match the size of x
        if x.size() != skip.size():
            skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=True)
         
        # Concatenate the skip connection (encoder output)
        x = torch.cat([x, skip], dim=1)
         
        # Apply convolutions
        x = self.conv1(x)
        x = self.conv2(x)
        return x
 
class UNetResNet50(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(UNetResNet50, self).__init__()
 
        # Load ResNet50 pre-trained on ImageNet
#         self.encoder = models.resnet50(pretrained=pretrained)
        self.encoder = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
         
 
        # Encoder layers from ResNet-50 (for skip connections)
        self.encoder_layers = [
            nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu, self.encoder.maxpool),  # (64, H/4, W/4)
            self.encoder.layer1,  # (256, H/4, W/4)
            self.encoder.layer2,  # (512, H/8, W/8)
            self.encoder.layer3,  # (1024, H/16, W/16)
            self.encoder.layer4   # (2048, H/32, W/32)
        ]
 
        # Decoder (Upsampling blocks)
        self.decoder4 = DecoderBlock(2048 + 1024, 512)  # Block for layer4 + layer3
        self.decoder3 = DecoderBlock(512 + 512, 256)    # Block for layer3 + layer2
        self.decoder2 = DecoderBlock(256 + 256, 128)    # Block for layer2 + layer1
        self.decoder1 = DecoderBlock(128 + 64, 64)      # Block for layer1 + conv1
 
        # Final segmentation head
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(64, num_classes, kernel_size=3, padding=1),
            # nn.Sigmoid()  # Use sigmoid for binary segmentation
        )
 
    def forward(self, x):
        # Save original input size for final upsampling
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)  # (B, 1, H, W) → (B, 3, H, W)

        original_size = x.shape[2:]  # (H, W)
 
        # Encoder forward pass
        x0 = self.encoder_layers[0](x)  # Initial convolution block (conv1)
        x1 = self.encoder_layers[1](x0)  # Skip connection 1 (layer1)
        x2 = self.encoder_layers[2](x1)  # Skip connection 2 (layer2)
        x3 = self.encoder_layers[3](x2)  # Skip connection 3 (layer3)
        x4 = self.encoder_layers[4](x3)  # Skip connection 4 (layer4)
 
        # Decoder forward pass
        x = self.decoder4(x4, x3)  # Decoder for layer4 + skip3
        x = self.decoder3(x, x2)   # Decoder for layer3 + skip2
        x = self.decoder2(x, x1)   # Decoder for layer2 + skip1
        x = self.decoder1(x, x0)    # Decoder for layer1 + initial conv1 output
 
        # Upsample the final output to match the input size dynamically
        x = F.interpolate(x, size=original_size, mode='bilinear', align_corners=True)
 
        # Final segmentation output
        x = self.segmentation_head(x)
 
        return x
    
    
def build_model(ckpt_path:str = None, device:str='cpu'):
    pretrained_tag = True if ckpt_path is None else False
    model = UNetResNet50(num_classes=1, pretrained=pretrained_tag).to(device)
    if ckpt_path is not None:
        checkpoint = torch.load(ckpt_path, map_location = 'cuda')
        try:
            model.load_state_dict(checkpoint)
        except:
            model.load_state_dict(remove_module_prefix(checkpoint))
    return model
    