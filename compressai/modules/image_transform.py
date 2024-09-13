
import torch.nn as nn
from compressai.layers import *
"""
    init:
        image_shape(3,H,W): shape of image
        patch_size:linear_embedding arg
        embed_dim:linear_embedding arg


    input:
        x(B,3,H,W): images

    output:
        feather(B,embed_dim*4,H/patch_size/8,W/patch_size/8):latent feather
        mid_feather[0](B,embed_dim*2,H/patch_size/2,W/patch_size/2:fetaher one
        mid_feather[0](B,embed_dim*4,H/patch_size/4,W/patch_size/4:fetaher two
"""
class Encoder(nn.Module):
    def __init__(self,image_shape,patch_size,embed_dim,out_channel_m):
        super(Encoder, self).__init__()
        self.image_shape = image_shape
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.patchEmbed = PatchEmbed(
            patch_size=patch_size, 
            in_chans=3, 
            embed_dim=embed_dim, 
            norm_layer=None
        )

        # 循环构建3个attention层  
        self.swinTransfomerBlockLayers = nn.ModuleList()
        self.patchMergerLayers = nn.ModuleList()
        channel = embed_dim
        resolution = [(int)(image_shape[1]/patch_size),(int)(image_shape[2]/patch_size)]
        for i in range(3):  
            # 创建注意力层并添加到列表中
            if(i == 2):
                swinTransfomerBlock2Layers = nn.ModuleList()
                for j in range(3):
                    swinTransfomerBlock = SwinTransformerBlock2(dim=channel, input_resolution=resolution, num_heads=1, window_size=4,shift_size=0)
                    swinTransfomerBlock2Layers.append(swinTransfomerBlock)
                self.swinTransfomerBlockLayers.append(swinTransfomerBlock2Layers)
            else:
                swinTransfomerBlock = SwinTransformerBlock2(dim=channel, input_resolution=resolution, num_heads=1, window_size=4,shift_size=0)
                self.swinTransfomerBlockLayers.append(swinTransfomerBlock)
            patchMerge = PatchMerging(input_resolution = resolution, dim = channel, norm_layer=nn.LayerNorm)
            self.patchMergerLayers.append(patchMerge)
            channel = channel*2
            resolution = [(int)((x+1) / 2) for x in resolution]
        # 最后一层attention
        self.lastSwinTransfomerBlockLayer = SwinTransformerBlock2(dim=channel, input_resolution=resolution, num_heads=1, window_size=4,shift_size=0)
        self.last_out = nn.Conv2d(embed_dim*8,out_channel_m,1,1)

    def forward(self,x):  
        b,c,h,w = x.shape
        # 1 词嵌入
        x = self.patchEmbed(x)    # (b,c,h,w)
        # 2 窗口注意力
        x = x.permute(0,2,3,1).view(b,-1,self.embed_dim)   #x:[b,w*h,c]
        for i in range(3):  
            if(i == 2):
                for j in range(3):
                    x = self.swinTransfomerBlockLayers[i][j](x) #x:[b,w*h,c]
            else:
                x = self.swinTransfomerBlockLayers[i](x) #x:[b,w*h,c]
            x = self.patchMergerLayers[i](x)
        # 3 最后一层窗口注意力
        x = self.lastSwinTransfomerBlockLayer(x)
        x = x.view(b,(int)(self.image_shape[1]/self.patch_size/8),
                     (int)(self.image_shape[2]/self.patch_size/8),self.embed_dim*8).permute(0,3,1,2) #[b,c,h,w]
        x = self.last_out(x)
        return x




"""
    init:
        image_shape(3,H,W): shape of image
        patch_size:linear_embedding arg
        embed_dim:linear_embedding arg


    input:
        feather(B,embed_dim*4,H/patch_size/8,W/patch_size/8): encoder feather of image

    output:
        x(B,3,H,W): reconstruction image
"""

class Decoder(nn.Module):
    def __init__(self,image_shape,patch_size,embed_dim,out_channel_m):
        super(Decoder, self).__init__()
        self.image_shape = image_shape
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.end_conv = nn.Sequential(nn.Conv2d(embed_dim, embed_dim * patch_size ** 2, kernel_size=5, stride=1, padding=2),
                                      nn.PixelShuffle(patch_size),
                                      nn.Conv2d(embed_dim, 3, kernel_size=3, stride=1, padding=1),
                                      )
        channel = embed_dim*8
        resolution = [(int)(image_shape[1]/patch_size/8),(int)(image_shape[2]/patch_size/8)]
        #第一个attention
        self.first_in = nn.Conv2d(out_channel_m,embed_dim*8,1,1)
        self.firstSwinTransfomerBlockLayer = SwinTransformerBlock2(dim=channel, input_resolution=resolution, num_heads=1, window_size=4,shift_size=0)
        #循环构建SwinTransformerBlock
        self.swinTransfomerBlockLayers = nn.ModuleList()
        self.patchMergerLayers = nn.ModuleList()
        for i in range(3):  
            patchMerge = PatchExpanding(input_resolution = resolution, dim = channel, norm_layer=nn.LayerNorm)
            self.patchMergerLayers.append(patchMerge)
            channel = (int)(channel/2)
            resolution = [(int)(2*x) for x in resolution]
            if(i == 0):
                swinTransfomerBlock2Layers = nn.ModuleList()
                for j in range(3):
                    swinTransfomerBlock = SwinTransformerBlock2(dim=channel, input_resolution=resolution, num_heads=1, window_size=4,shift_size=0)
                    swinTransfomerBlock2Layers.append(swinTransfomerBlock)
                self.swinTransfomerBlockLayers.append(swinTransfomerBlock2Layers)
            else:
                swinTransfomerBlock = SwinTransformerBlock2(dim=channel, input_resolution=resolution, num_heads=1, window_size=4,shift_size=0)
                self.swinTransfomerBlockLayers.append(swinTransfomerBlock)

    def forward(self,x):  # x:[B,C,H,W]
        # 1 第一层窗口注意力
        x = self.first_in(x)
        b,c,h,w = x.shape
        x = x.permute(0,2,3,1).view(b,-1,c)   #x:[b,w*h,c]
        x = self.firstSwinTransfomerBlockLayer(x)
        # 2 窗口注意力
        for i in range(3):  
            x = self.patchMergerLayers[i](x)
            if(i == 0):
                for j in range(3):
                    x = self.swinTransfomerBlockLayers[i][j](x) #x:[b,w*h,c]
            else:
                x = self.swinTransfomerBlockLayers[i](x) #x:[b,w*h,c]
        # 3 反向词嵌入
        x = x.view(b,(int)(self.image_shape[1]/self.patch_size),(int)(self.image_shape[2]/self.patch_size),self.embed_dim).permute(0,3,1,2) #[b,c,h,w]
        x = self.end_conv(x)
        return x
