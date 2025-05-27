import torch.nn as nn 
import torch 
import torch.nn.functional as F
from models import remove_module_prefix

class ConvBlock(nn.Module) :
    def __init__(self, in_channels, out_channels) :
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
        )

    def forward(self, x) :
        return self.block(x)

class UNetPP(nn.Module) :
    def __init__(self, in_channels, out_channels, base_channels = 64, deep_supervision = False):
        super().__init__()
        self.deep_supervision = deep_supervision

        # encoder
        self.conv00 = ConvBlock(in_channels, base_channels)
        self.pool0 = nn.MaxPool2d(2)
        self.conv10 = ConvBlock(base_channels, base_channels * 2)
        self.pool1 = nn.MaxPool2d(2)
        self.conv20 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool2 = nn.MaxPool2d(2)
        self.conv30 = ConvBlock(base_channels * 4, base_channels * 8)
        self.pool3 = nn.MaxPool2d(2)
        self.conv40 = ConvBlock(base_channels * 8, base_channels * 16)

        # decoder - nested blocks
        self.up01 = ConvBlock(base_channels + base_channels * 2, base_channels)
        self.up11 = ConvBlock(base_channels * 2 + base_channels * 4, base_channels * 2)
        self.up21 = ConvBlock(base_channels * 4 + base_channels * 8, base_channels * 4)
        self.up31 = ConvBlock(base_channels * 8 + base_channels * 16, base_channels * 8)

        self.up02 = ConvBlock(base_channels * 2 + base_channels + base_channels, base_channels)
        self.up12 = ConvBlock(base_channels * 4 + base_channels * 2 + base_channels * 2, base_channels * 2)
        self.up22 = ConvBlock(base_channels * 8 + base_channels * 4 + base_channels * 4, base_channels * 4)

        self.up03 = ConvBlock(base_channels * 2 + base_channels + base_channels + base_channels, base_channels)
        self.up13 = ConvBlock(base_channels * 4 + base_channels * 2 + base_channels * 2 + base_channels * 2, base_channels * 2)

        self.up04 = ConvBlock(base_channels * 2 + base_channels + base_channels + base_channels + base_channels, base_channels)

        # output layer(s)
        if self.deep_supervision :
            self.final1 = nn.Conv2d(base_channels, out_channels, kernel_size = 1)
            self.final2 = nn.Conv2d(base_channels, out_channels, kernel_size = 1)
            self.final3 = nn.Conv2d(base_channels, out_channels, kernel_size = 1)
            self.final4 = nn.Conv2d(base_channels, out_channels, kernel_size = 1)
        else:
            self.final = nn.Conv2d(base_channels, out_channels, kernel_size = 1)

    def forward(self, x) :
        x00 = self.conv00(x)
        x10 = self.conv10(self.pool0(x00))
        x20 = self.conv20(self.pool1(x10))
        x30 = self.conv30(self.pool2(x20))
        x40 = self.conv40(self.pool3(x30))

        x01 = self.up01(torch.cat([x00, F.interpolate(x10, scale_factor = 2, mode = 'bilinear', align_corners = True)], dim = 1))
        x11 = self.up11(torch.cat([x10, F.interpolate(x20, scale_factor = 2, mode = 'bilinear', align_corners = True)], dim = 1))
        x21 = self.up21(torch.cat([x20, F.interpolate(x30, scale_factor = 2, mode = 'bilinear', align_corners = True)], dim = 1))
        x31 = self.up31(torch.cat([x30, F.interpolate(x40, scale_factor = 2, mode = 'bilinear', align_corners = True)], dim = 1))

        x02 = self.up02(torch.cat([x00, x01, F.interpolate(x11, scale_factor = 2, mode = 'bilinear', align_corners = True)], dim = 1))
        x12 = self.up12(torch.cat([x10, x11, F.interpolate(x21, scale_factor = 2, mode = 'bilinear', align_corners = True)], dim = 1))
        x22 = self.up22(torch.cat([x20, x21, F.interpolate(x31, scale_factor = 2, mode = 'bilinear', align_corners = True)], dim = 1))

        x03 = self.up03(torch.cat([x00, x01, x02, F.interpolate(x12, scale_factor = 2, mode = 'bilinear', align_corners = True)], dim = 1))
        x13 = self.up13(torch.cat([x10, x11, x12, F.interpolate(x22, scale_factor = 2, mode = 'bilinear', align_corners = True)], dim = 1))

        x04 = self.up04(torch.cat([x00, x01, x02, x03, F.interpolate(x13, scale_factor = 2, mode = 'bilinear', align_corners = True)], dim = 1))

        if self.deep_supervision:
            return [
                self.final1(x01),
                self.final2(x02),
                self.final3(x03),
                self.final4(x04),
            ]
        else:
            return self.final(x04)
        
        
def build_model(ckpt_path:str = None, device:str='cpu'):
    model = UNetPP(in_channels=1, out_channels=1, base_channels=64, deep_supervision=False).to(device)
    if ckpt_path is not None:
        checkpoint = torch.load(ckpt_path, map_location = 'cuda')
        try:
            model.load_state_dict(checkpoint)
        except:
            model.load_state_dict(remove_module_prefix(checkpoint))
    return model