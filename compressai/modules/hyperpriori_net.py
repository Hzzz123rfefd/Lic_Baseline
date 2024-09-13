import torch.nn as nn
from compressai.layers import *


def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


class HyperprioriEncoder(nn.Module):
    def __init__(self,feather_shape,out_channel_m,out_channel_n):
        super(HyperprioriEncoder, self).__init__()
        self.feather_shape = feather_shape
        self.h_a = nn.Sequential(
            conv(out_channel_m, out_channel_n, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(out_channel_n, out_channel_n),
            nn.ReLU(inplace=True),
            conv(out_channel_n, out_channel_n),
        )

    def forward(self,x):
        return self.h_a(torch.abs(x))
    
def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )

class HyperprioriDecoder(nn.Module):
    def __init__(self,feather_shape,out_channel_m,out_channel_n):
        super(HyperprioriDecoder, self).__init__()
        self.feather_shape = feather_shape
        self.h_s = nn.Sequential(
            deconv(out_channel_n, out_channel_n),
            nn.ReLU(inplace=True),
            deconv(out_channel_n, out_channel_n),
            nn.ReLU(inplace=True),
            conv(out_channel_n, out_channel_m, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        return F.interpolate(self.h_s(x), size=(self.feather_shape[1], self.feather_shape[2]), mode='bilinear', align_corners=False)
    