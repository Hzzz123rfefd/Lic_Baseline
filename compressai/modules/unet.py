import sys
import os
sys.path.append(os.getcwd())

import torch
from torch import nn
from einops import rearrange 
# 两次3x3卷积，不改变feature大小，conv+bn+relu
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            torch.nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=(3, 3), stride=(1, 1), padding=1),
            torch.nn.BatchNorm2d(out_c),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_c, out_c, (3, 3), (1, 1), 1),
            torch.nn.BatchNorm2d(out_c),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DownSample(nn.Module):
    def __init__(self, in_c, out_c):
        super(DownSample, self).__init__()
        self.down_sample = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=(1, 1), stride=(2, 2)),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.down_sample(x)

class TimeEmbed(nn.Module):
    def __init__(self, time_dim, out_dim):
        super(TimeEmbed, self).__init__()
        self.time_embed_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim,out_dim)
        )
    def forward(self,x):
        x = self.time_embed_layer(x)
        return rearrange(x,"b c -> b c 1 1")

class SelfAttation3D(nn.Module):
    def __init__(self, channels, h,w):
        super(SelfAttation3D, self).__init__()
        self.channels = channels
        self.h = h
        self.w = w
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.h * self.w).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.h, self.w)
# 输入和输出分别为通道为in_c = 3,out_c = 3的图片

class Unet(nn.Module):
    def __init__(self, 
                 img_shape ,
                 in_c = 3,
                 out_c = 3,
                 time_dim=32,
                 out_channel_n = 128,
                 out_channel_m = 192
    ):
        super(Unet, self).__init__()
        self.h = img_shape[0]
        self.w = img_shape[1]
        self.time_dim = time_dim
        # 4次下采样: double_conv + max_pool
        self.conv1 = DoubleConv(in_c, out_channel_n)
        self.att1 = SelfAttation3D(out_channel_n,self.h,self.w)
        self.e_emd_1 = TimeEmbed(time_dim,out_channel_n)
        self.pool1 = DownSample(out_channel_n, out_channel_n)  # 1/2
        self.conv2 = DoubleConv(out_channel_n, out_channel_n)
        self.att2 = SelfAttation3D(out_channel_n,(int)(self.h/2),(int)(self.w/2))
        self.e_emd_2 = TimeEmbed(time_dim,out_channel_n)
        self.pool2 = DownSample(out_channel_n, out_channel_n)  # 1/4
        self.conv3 = DoubleConv(out_channel_n, out_channel_n)
        self.att3 = SelfAttation3D(out_channel_n,(int)(self.h/4),(int)(self.w/4))
        self.e_emd_3 = TimeEmbed(time_dim,out_channel_n)
        self.pool3 = DownSample(out_channel_n, out_channel_n)  # 1/8
        self.conv4 = DoubleConv(out_channel_n, out_channel_n)
        self.att4 = SelfAttation3D(out_channel_n,(int)(self.h/8),(int)(self.w/8))
        self.e_emd_4 = TimeEmbed(time_dim,out_channel_n)
        self.pool4 = DownSample(out_channel_n, out_channel_n)  # 1/16

        self.conv5 = DoubleConv(out_channel_n, out_channel_m)

        # 4次上采样: up+cat+conv
        self.up6 = torch.nn.ConvTranspose2d(out_channel_m, out_channel_n, kernel_size=(2, 2), stride=(2, 2))  # 卷积步长，即要将输入扩大的倍数。
        self.conv6 = DoubleConv(out_channel_n * 2, out_channel_n)
        self.att6 = SelfAttation3D(out_channel_n,(int)(self.h/8),(int)(self.w/8))
        self.d_emd_1 = TimeEmbed(time_dim,out_channel_n)

        self.up7 = torch.nn.ConvTranspose2d(out_channel_n, out_channel_n, (2, 2), (2, 2))
        self.conv7 = DoubleConv(out_channel_n * 2 , out_channel_n)
        self.att7 = SelfAttation3D(out_channel_n,(int)(self.h/4),(int)(self.w/4))
        self.d_emd_2 = TimeEmbed(time_dim,out_channel_n)

        self.up8 = torch.nn.ConvTranspose2d(out_channel_n, out_channel_n, (2, 2), (2, 2))
        self.conv8 = DoubleConv(out_channel_n * 2, out_channel_n)
        self.att8 = SelfAttation3D(out_channel_n,(int)(self.h/2),(int)(self.w/2))
        self.d_emd_3 = TimeEmbed(time_dim,out_channel_n)

        self.up9 = torch.nn.ConvTranspose2d(out_channel_n, out_channel_n, (2, 2), (2, 2))
        self.conv9 = DoubleConv(out_channel_n * 2, out_channel_n)
        self.att9 = SelfAttation3D(out_channel_n,(int)(self.h),(int)(self.w))
        self.d_emd_4 = TimeEmbed(time_dim,out_channel_n)

        # head
        self.conv10 = torch.nn.Conv2d(out_channel_n, out_c, kernel_size=(1, 1))  # 加一个0通道bg类别

        # 加上时间步特征
    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2).float() / channels)
        ).to(t.device)
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc        
 

    def forward(self, x,t,context):
        # 对时间步特征进行encode
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        # 4次下采样: double_conv + max_pool
        c1 = self.conv1(x) + self.e_emd_1(t)
        # c1 = self.att1(c1)
        p1 = self.pool1(c1)  # 1/2
        c2 = self.conv2(p1) + self.e_emd_2(t)
        # c2 = self.att2(c2)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2) + self.e_emd_3(t)
        # c3 = self.att3(c3)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3) + self.e_emd_4(t)
        # c4 = self.att4(c4)
        p4 = self.pool4(c4)  # 1/16

        c5 = self.conv5(p4) + context

        # 4次上采样: up+cat+conv
        up6 = self.up6(c5)  # 1/8
        m6 = torch.cat([up6, c4], dim=1)
        c6 = self.conv6(m6) + self.d_emd_1(t)
        # c6 = self.att6(c6)

        up7 = self.up7(c6)  # 1/4
        m7 = torch.cat([up7, c3], dim=1)
        c7 = self.conv7(m7) + self.d_emd_2(t)
        # c7 = self.att7(c7)

        up8 = self.up8(c7)  # 1/2
        m8 = torch.cat([up8, c2], dim=1)
        c8 = self.conv8(m8) + self.d_emd_3(t)
        # c8 = self.att8(c8)

        up9 = self.up9(c8)  # 1/1
        m9 = torch.cat([up9, c1], dim=1)
        c9 = self.conv9(m9) + self.d_emd_4(t)
        # c9 = self.att9(c9)

        # head
        c10 = self.conv10(c9)
        return c10
    
if __name__ == "__main__":
    model = Unet(img_shape=(512,512))
    image = torch.rand(2,3,512,512)
    context = torch.rand(2,192,32,32)
    t = torch.rand(2)
    with torch.no_grad():
        model(image,t,context)