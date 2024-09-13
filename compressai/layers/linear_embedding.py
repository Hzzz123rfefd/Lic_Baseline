from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn as nn
import torch.nn.functional as F
"""
  将图像数据转为序列数据
    init:
        patch_size:下采样倍数
        in_chans：输入通道数量
        embed_dim：输出通道数量
        norm_layer：是否包含norm_layer

    input:
        input_vector:(b,in_chans,w,h)   
        output_vector(b,w/patch_size*h/patch_size,embed_dim)
  exp:
    model = PatchEmbed(img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None)
    x = torch.rand(4,3,224,224);
    model(x).size()
    torch.Size([4, 3136, 96])
  
"""

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size       #[4,4]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None
    def forward(self, x):
        """Forward function."""
        # padding   讲解:   https://blog.csdn.net/qq_34914551/article/details/102940377
        B, C, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))                    # W不是patch_size的倍数 填充为0
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))             # H不是patch_size的倍数 填充为0
        # embed
        x = self.proj(x)  # B C Wh Ww
        # norm
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
        return x