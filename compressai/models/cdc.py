import numpy as np
import torch.nn as nn
import sys
import os
sys.path.append(os.getcwd())

from compressai.modules import *
from compressai.entropy_models import *
from compressai.models import CompressionModel

class Diffusion(nn.Module):
    def __init__(self,
        image_shape,                 
        in_c,
        out_c,
        time_dim,
        out_channel_n,
        out_channel_m,
        noise_steps, 
        beta_start, 
        beta_end
        ):
        super(Diffusion,self).__init__()
        self.image_shape = image_shape
        self.in_c = in_c
        self.out_c = out_c
        self.out_channel_n = out_channel_n
        self.out_channel_m = out_channel_m
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.unet = Unet(
            img_shape = image_shape,
            in_c = in_c,
            out_c = out_c,
            time_dim = time_dim,
            out_channel_n = out_channel_n,
            out_channel_m = out_channel_m
        )
        # 1 generate betas
        self.betas = self.generate_betas()
        # 2 calulate aerfas
        self.aerfas = 1 - self.betas
        # 3 calulate aerfa_ba
        self.aerfa_ba = torch.cumprod(self.aerfas, dim=0)
    def generate_betas(self):
        betas = np.linspace(self.beta_start, self.beta_end, self.noise_steps , dtype=np.float32)
        return torch.from_numpy(betas)
    def generate_t_steps(self,batch_size):
        return torch.randint(low=1, high=self.noise_steps+1, size=(batch_size,))
    def forward(self,context):
        b,c,h,w = context.shape
        device = context.device
        # generate gaussian noise
        x = torch.randn(b,self.in_c,self.image_shape[0],self.image_shape[1]).to(device)
        self.betas = self.betas.to(device)
        self.aerfas = self.aerfas.to(device)
        self.aerfa_ba = self.aerfa_ba.to(device)
        # reconstruction image
        for i in reversed(range(1, self.noise_steps)):
            # get t steps
            t = (torch.ones(b) * i).float().to(device)
            # predict noisy 
            predicted_noise = self.unet(x, t,context)
            # reverse generate
            index = t.cpu().detach().numpy()
            alpha = self.aerfas[index][:, None, None, None]
            alpha_hat = self.aerfa_ba[index][:, None, None, None]
            beta = self.betas[index][:, None, None, None]
            mu = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat)))* predicted_noise)
            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            # var = torch.sqrt(beta)
            alpha_hat_1 = self.aerfa_ba[index-1][:, None, None, None]
            var = 1 / ((alpha/beta + (1/(1-alpha_hat_1))))
            var = torch.sqrt(var)
            x = mu + var * noise
        x = (x.clamp(0, 1))
        return x


class CDC(CompressionModel):
    def __init__(self,image_channel,image_height,image_weight,out_channel_m,out_channel_n,time_dim,noise_steps,beta_start, beta_end):
        super().__init__(image_channel,image_height,image_weight,out_channel_m,out_channel_n)
        self.time_dim = time_dim
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.image_transform_encoder = nn.Sequential(
            conv(image_channel,out_channel_n),
            GDN(out_channel_n),
            conv(out_channel_n,out_channel_n),
            GDN(out_channel_n),
            conv(out_channel_n,out_channel_n),
            GDN(out_channel_n),
            conv(out_channel_n,out_channel_m)
        )
        self.hyperpriori_encoder = HyperprioriEncoder(
            feather_shape = [out_channel_m,(int)(self.image_shape[1]/16),(int)(self.image_shape[2]/16)],
            out_channel_m = out_channel_m,
            out_channel_n = out_channel_n
        )
        self.hyperpriori_decoder = HyperprioriDecoder(
            feather_shape = [out_channel_m,(int)(self.image_shape[1]/16),(int)(self.image_shape[2]/16)],
            out_channel_m = out_channel_m,
            out_channel_n = out_channel_n
        )
        self.diffusion = Diffusion(
            image_shape = [self.image_height,self.image_weight],                 
            in_c = image_channel,
            out_c = image_channel,
            time_dim = self.time_dim,
            out_channel_n = self.out_channel_n,
            out_channel_m = self.out_channel_m,
            noise_steps = self.noise_steps, 
            beta_start = self.beta_start, 
            beta_end = self.beta_end
        )
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
        """ quantization and likelihood estimation of y """
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        """ reconstruction image """
        x_hat = self.diffusion(y_hat)
        output = {
            "reconstruction_image":x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }
        return output

if __name__ == "__main__":
    model = CDC(
        image_channel = 3,
        image_height = 512,
        image_weight = 512,
        out_channel_m = 192,
        out_channel_n = 128,
        time_dim = 40,
        noise_steps = 20,
        beta_start = 0.0001,
        beta_end = 0.02
    )
    x = torch.rand(2,3,512,512)
    with torch.no_grad():
        model(x)
