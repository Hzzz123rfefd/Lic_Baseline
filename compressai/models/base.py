import torch
import torch.nn as nn

class CompressionModel(nn.Module):
    """Base class for constructing an auto-encoder with at least one entropy
    bottleneck module.

    Args:
        entropy_bottleneck_channels (int): Number of channels of the entropy
            bottleneck
    """

    def __init__(self):
        super().__init__()


    def forward(self, *args):
        raise NotImplementedError()

    def from_pretrain(self,checkpoint_path,device):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.load_state_dict(checkpoint["state_dict"])


    def compress():
        pass

    def decompress():
        pass
    # def update(self, force=False):
    #     """Updates the entropy bottleneck(s) CDF values.

    #     Needs to be called once after training to be able to later perform the
    #     evaluation with an actual entropy coder.

    #     Args:
    #         force (bool): overwrite previous values (default: False)

    #     Returns:
    #         updated (bool): True if one of the EntropyBottlenecks was updated.

    #     """
    #     updated = False
    #     for m in self.children():
    #         if not isinstance(m, EntropyBottleneck):
    #             continue
    #         rv = m.update(force=force)
    #         updated |= rv
    #     return updated

    # def load_state_dict(self, state_dict):
    #     # Dynamically update the entropy bottleneck buffers related to the CDFs
    #     update_registered_buffers(
    #         self.entropy_bottleneck,
    #         "entropy_bottleneck",
    #         ["_quantized_cdf", "_offset", "_cdf_length"],
    #         state_dict,
    #     )
    #     super().load_state_dict(state_dict)

