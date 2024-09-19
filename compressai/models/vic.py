import torch
import torch.nn as nn
from compressai.modules import *
from compressai.entropy_models import *
from .base import *


class VIC(CompressionModel):
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
        self.hyperpriori_encoder = HyperprioriEncoder(feather_shape = [out_channel_m,(int)(image_height/16),(int)(image_weight/16)],
                                                      out_channel_m = out_channel_m,
                                                      out_channel_n = out_channel_n)
        self.hyperpriori_decoder = HyperprioriDecoder(feather_shape = [out_channel_m,(int)(image_height/16),(int)(image_weight/16)],
                                                      out_channel_m = out_channel_m,
                                                      out_channel_n = out_channel_n)
        self.entropy_bottleneck = EntropyBottleneck(out_channel_n)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self,x):
        """ forward transformation """
        y = self.image_transform_encoder(x)
        """ super prior forward transformation """
        z = self.hyperpriori_encoder(y)
        """ quantization and likelihood estimation of z"""
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        """ lather feature variance"""
        scales_hat = self.hyperpriori_decoder(z_hat)
        """ quantization and likelihood estimation of y """
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