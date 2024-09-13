
import torch.nn as nn

class PatchSplitting(nn.Module):
    """ Patch Expanding Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self,input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(dim, dim * 2, bias=False)
        self.norm = norm_layer(dim)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        B, L, C = x.shape
        H,W = self.resolution
        assert L == H * W, "input feature has wrong size"
        # 增加通道数
        x = self.norm(x)
        x = self.reduction(x)           # B, L, 2C
        # 减小通道数,增加分辨率
        x = x.permute(0, 2, 1).contiguous().view(B, 2*C, H, W)
        x = self.shuffle(x)             # B, C//2 ,2H, 2W
        x = x.permute(0, 2, 3, 1).contiguous().view(B, 4 * L, -1)   #[B,4L,C//2]
        return x