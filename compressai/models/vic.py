import math
import torch
import torch.nn as nn
from compressai.modules import *
from compressai.entropy_models import *

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    """Returns table of logarithmically scales."""
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

def config_model(args):
    args.image_channel = 3
    args.image_height = 720
    args.image_width = 960
    args.m = 192
    args.n = 128
    args.z_h = 12
    args.z_w = 15
    return args

class VIC(nn.Module):
    def __init__(self,image_channel,image_height,image_weight,out_channel_m,out_channel_n):
        super().__init__()
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
        self.hyperpriori_encoder = HyperprioriEncoder(feather_shape = [out_channel_m,(int)(image_height/16),(int)(image_weight/16)],
                                                      out_channel_m = out_channel_m,
                                                      out_channel_n = out_channel_n)
        self.hyperpriori_decoder = HyperprioriDecoder(feather_shape = [out_channel_m,(int)(image_height/16),(int)(image_weight/16)],
                                                      out_channel_m = out_channel_m,
                                                      out_channel_n = out_channel_n)
        self.entropy_bottleneck = EntropyBottleneck(out_channel_n)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self,x):
        """ 正变换 """
        y = self.image_transform_encoder(x)
        """ 超先验正变换 """
        z = self.hyperpriori_encoder(y)
        """ 超先验特征z量化和似然估计"""
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        """ 正变换特征方差"""
        scales_hat = self.hyperpriori_decoder(z_hat)
        """ 正变换特征y量化和似然估计"""
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        """ 反变换"""
        x_hat = self.image_transform_decoder(y_hat)
        output = {
            "reconstruction_image":x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }
        return output
    
    def from_pretrain(self,model_path,device):
        checkpoint = torch.load(model_path, map_location=device)
        self.load_state_dict(checkpoint["state_dict"])


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


