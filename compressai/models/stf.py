import math
import torch.nn as nn
from compressai.modules import *
from compressai.entropy_models import *
from .base import *
# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    """Returns table of logarithmically scales."""
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

def config_model(args):
    args.image_channel = 3
    args.image_height = 512
    args.image_width = 512
    args.patch_size = 2
    args.embedding_dim = 40
    args.m = 192
    args.n = 128
    args.z_h = 8
    args.z_w = 8
    return args

class STF(CompressionModel):
    def __init__(self,image_channel,image_height,image_weight,patch_size,embedding_dim,out_channel_m,out_channel_n):
        super().__init__()
        self.image_shape = [image_channel,image_height,image_weight]
        self.patch_size = patch_size
        self.embed_dim = embedding_dim
        self.feather_shape = [embedding_dim*8,
                                            (int)(self.image_shape[1]/patch_size/8),
                                            (int)(self.image_shape[2]/patch_size/8)]
        self.image_transform_encoder = Encoder(image_shape = self.image_shape,
                                                                            patch_size = patch_size,
                                                                            embed_dim = embedding_dim,
                                                                            out_channel_m= out_channel_m)
        self.image_transform_decoder = Decoder(image_shape = self.image_shape,
                                                                            patch_size = patch_size,
                                                                            embed_dim = embedding_dim,
                                                                            out_channel_m= out_channel_m)
        self.hyperpriori_encoder = HyperprioriEncoder(feather_shape = [out_channel_m,(int)(self.image_shape[1]/16),(int)(self.image_shape[2]/16)],
                                                                                    out_channel_m = out_channel_m,
                                                                                    out_channel_n = out_channel_n)
        self.hyperpriori_decoder = HyperprioriDecoder(feather_shape = [out_channel_m,(int)(self.image_shape[1]/16),(int)(self.image_shape[2]/16)],
                                                                                    out_channel_m = out_channel_m,
                                                                                    out_channel_n = out_channel_n)
        self.entropy_bottleneck = EntropyBottleneck(out_channel_n)
        self.gaussian_conditional = GaussianConditional(None)
    def forward(self,image):
        """ forward transformation """
        y = self.image_transform_encoder(image)
        """ super prior forward transformation """
        z = self.hyperpriori_encoder(y)
        """ quantization and likelihood estimation of z"""
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        """ lather feature variance"""
        scales_hat = self.hyperpriori_decoder(z_hat)
        """ quantization and likelihood estimation of y"""
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        """ reverse transformation """
        x_hat = self.image_transform_decoder(y_hat)
        output = {
            "reconstruction_image":x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }
        return output
    
    def compress(self, image, device):
        """ forward """
        image = image.to(device)
        image = image.unsqueeze(0)/255
        self.eval()
        with torch.no_grad():
            y = self.image_transform_encoder(image)
            z = self.hyperpriori_encoder(y)

            # self.entropy_bottleneck.update()
            z_strings = self.entropy_bottleneck.compress(z)
            z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
            # with open("z.bin", 'wb') as file:
            #     file.write(z_strings[0])
            # self.gaussian_conditional.update()
            scales_hat = self.hyperpriori_decoder(z_hat)
            indexes = self.gaussian_conditional.build_indexes(scales_hat)
            y_strings = self.gaussian_conditional.compress(y, indexes)
            # with open("y.bin", 'wb') as file:
            #     file.write(y_strings[0])

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}
    
    def decompress(self,strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        self.eval()
        with torch.no_grad():
            z_hat = self.entropy_bottleneck.decompress(strings[1], shape)

            scales_hat = self.hyperpriori_decoder(z_hat)
            indexes = self.gaussian_conditional.build_indexes(scales_hat)
            y_hat = self.gaussian_conditional.decompress(strings[0], indexes)

            x_hat = self.image_transform_decoder(y_hat)

        image = torch.clamp(x_hat,min = 0,max = 1)
        image = image.squeeze(0) * 255
        image = torch.round(image)
        return image

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

