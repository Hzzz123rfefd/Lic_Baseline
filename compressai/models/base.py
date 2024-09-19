import math
import torch
import torch.nn as nn

from compressai.entropy_models import *
# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    """Returns table of logarithmically scales."""
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

class CompressionModel(nn.Module):
    """Base class for constructing an auto-encoder with at least one entropy
    bottleneck module.

    Args:
        entropy_bottleneck_channels (int): Number of channels of the entropy
            bottleneck
    """

    def __init__(self,image_channel,image_height,image_weight,out_channel_m,out_channel_n):
        super().__init__()
        self.image_shape = [image_channel,image_height,image_weight]
        self.image_channel = image_channel
        self.image_height = image_height
        self.image_weight = image_weight
        self.out_channel_m = out_channel_m
        self.out_channel_n = out_channel_n


    def forward(self, *args):
        raise NotImplementedError()

    def from_pretrain(self,checkpoint_path,device):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.load_state_dict(checkpoint["state_dict"])

    def get_z_shape(self):
        return torch.tensor([(int)(self.image_height/64),(int)(self.image_weight/64)])


    def compress():
        pass

    def decompress():
        pass

    def update(self, scale_table=None, force=False, update_quantiles: bool = False):
        """Updates EntropyBottleneck and GaussianConditional CDFs.

        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.

        Args:
            scale_table (torch.Tensor): table of scales (i.e. stdev)
                for initializing the Gaussian distributions
                (default: 64 logarithmically spaced scales from 0.11 to 256)
            force (bool): overwrite previous values (default: False)
            update_quantiles (bool): fast update quantiles (default: False)

        Returns:
            updated (bool): True if at least one of the modules was updated.
        """
        if scale_table is None:
            scale_table = get_scale_table()
        updated = False
        for _, module in self.named_modules():
            if isinstance(module, EntropyBottleneck):
                updated |= module.update(force=force)
            if isinstance(module, GaussianConditional):
                updated |= module.update_scale_table(scale_table, force=force)
        return updated


