import torch.nn as nn
from compressai.modules import *
from compressai.entropy_models import *
from .base import *


class CDC(CompressionModel):
    def __init__(self,image_channel,image_height,image_weight,out_channel_m,out_channel_n):
        super().__init__(image_channel,image_height,image_weight,out_channel_m,out_channel_n)
        self.image_transform_encoder = nn.Sequential(
            conv(image_channel,out_channel_n),
            GDN(out_channel_n),
            conv(out_channel_n,out_channel_n),
            GDN(out_channel_n),
            conv(out_channel_n,out_channel_n),
            GDN(out_channel_n),
            conv(out_channel_n,out_channel_m)
        )
        self.image_transform_decoder = nn.Sequential(
            deconv(out_channel_m,out_channel_n),
            GDN(out_channel_n,inverse= True),
            deconv(out_channel_n,out_channel_n),
            GDN(out_channel_n,inverse= True),
            deconv(out_channel_n,out_channel_n),
            GDN(out_channel_n,inverse= True),
            deconv(out_channel_n,image_channel)  
        )     