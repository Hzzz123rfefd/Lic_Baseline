import argparse
import json
import math

from compressai import * 
from compressai.models import *
from compressai.datasets import *
from compressai.zoo import models
from torch import nn,optim
from tqdm import tqdm
import shutil
import os
import warnings
import torch

warnings.filterwarnings("ignore", message=".*indexing argument.*")


class ModelLoss(nn.Module):
    """model loss"""
    def __init__(self,lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self,output,target):
        N, _, H, W = target["image"].size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        """ reconstruction loss """
        out["reconstruction_loss"] = self.mse(output["reconstruction_image"], target["image"])

        """ all loss """
        out["loss"] = out["reconstruction_loss"] + self.lmbda * out["bpp_loss"]
        return out

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_one_epoch(
    model, criterion, train_dataloader, optimizer, epoch, log_path, clip_max_norm
):
    total_loss = AverageMeter()
    reconstruction_loss = AverageMeter()
    bpp_loss = AverageMeter()
    model.train()
    device = next(model.parameters()).device
    pbar = tqdm(train_dataloader,desc="Processing epoch "+str(epoch), unit="batch")
    for batch_id, image_data, in enumerate(pbar):
        """ get data """
        image_data = image_data.to(device)
        image_data = image_data/255.0

        """ grad zeroing """
        optimizer.zero_grad()

        """ forward """
        used_memory = 0
        after_used_memory = 0
        if device == "cuda":
            used_memory = torch.cuda.memory_allocated(torch.cuda.current_device()) / (1024 ** 3)  
        output = model(image_data)

        """ calculate loss """
        target = {}
        target["image"] = image_data
        out_criterion = criterion(output, target)

        if device == "cuda":
            after_used_memory = torch.cuda.memory_allocated(torch.cuda.current_device()) / (1024 ** 3) 
        out_criterion["loss"].backward()

        """ grad clip """
        if clip_max_norm > 0:
            clip_gradient(optimizer,clip_max_norm)
        
        """ modify parameters """
        optimizer.step()

        total_loss.update(out_criterion["loss"].item())
        reconstruction_loss.update(out_criterion["reconstruction_loss"].item())
        bpp_loss.update(out_criterion["bpp_loss"].item())
        postfix_str = "total_loss: {:.4f}, reconstruction_loss: {:.4f}, bpp:{:.4f}, use_memory: {:.1f}G".format(
            math.sqrt(total_loss.avg), 
            math.sqrt(reconstruction_loss.avg),
            bpp_loss.avg,
            after_used_memory - used_memory
        )
        pbar.set_postfix_str(postfix_str)
        pbar.update()
    with open(log_path, "a") as file:
        file.write(postfix_str+"\n")

def test_epoch(
        epoch, test_dataloader, model, criterion,log_path
):
    model.eval()
    device = next(model.parameters()).device

    total_loss = AverageMeter()
    reconstruction_loss = AverageMeter()
    bpp_loss = AverageMeter()

    with torch.no_grad():
        for batch_id, image_data in enumerate(test_dataloader):
            image_data = image_data.to(device)
            image_data = image_data/255.0
            output = model(image_data)

            target = {}
            target["image"] = image_data
            out_criterion = criterion(output, target)
            total_loss.update(out_criterion["loss"])
            reconstruction_loss.update(out_criterion["reconstruction_loss"])
            bpp_loss.update(out_criterion["bpp_loss"])
    str = (        
        f"Test epoch {epoch}:"
        f" total_loss: {math.sqrt(total_loss.avg):.4f} |"
        f" reconstruction_loss: {math.sqrt(reconstruction_loss.avg):.4f}|"
        f" bpp_loss: {bpp_loss.avg:.4f} \n"
        )
    print(str)
    with open(log_path, "a") as file:
        file.write(str+"\n")
    return total_loss.avg

def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename[:-4]+"_best"+filename[-4:])



def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def main(args):
    """get data"""
    train_dataloader,test_dataloader = get_model_data(
        data_path = args.datasets_path,
        data_size = args.data_size,
        image_channel = args.image_channel,
        image_height = args.image_height,
        image_weight = args.image_weight,
        batch_size = args.batch_size
    )

    """get train device"""
    device = args.device if torch.cuda.is_available() else "cpu"

    """get net struction"""
    with open("cof/" + args.model_name + ".json", 'r') as file:
        net = models[args.model_name](**json.load(file))
    net.to(device)

    """ get optimizer"""
    optimizer = optim.Adam(net.parameters(),args.lr)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer, 
                                                                                            mode = "min", 
                                                                                            factor = args.factor, 
                                                                                            patience = args.patience)

    """get model loss criterion"""
    criterion = ModelLoss(lmbda = args.lamda)

    """ get net parameters"""
    first = False  # a flag, is model existting
    model_name =  "lambda = " + str(args.lamda)
    log_path = args.save_model_dir + args.model_name + "/" + model_name + "/model.pth"
    check_point_path = args.save_model_dir + args.model_name + "/" + model_name + "/model.pth"
    # model is not exist,need to init 
    if not os.path.isdir(args.save_model_dir + args.model_name + "/" + model_name):
        # create model save path
        os.makedirs(args.save_model_dir + args.model_name + "/" + model_name)
        # create model train log
        with open(log_path, "w") as file:
            pass
        # create model file
        save_checkpoint(
            state = {
                "epoch": -1,
                "state_dict": net.state_dict(),
                "loss": float("inf"),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
            },
            is_best = True,
            filename = check_point_path,
        )
        first = True

    # load parameters
    checkpoint = torch.load(check_point_path, map_location=device)
    print("Loading", check_point_path)
    net.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    last_epoch = checkpoint["epoch"] + 1

    if first:
        checkpoint_init = torch.load(args.save_model_dir + args.model_name + "/lambda = 0.0001/model.pth", map_location=device)
        net.load_state_dict(checkpoint_init["state_dict"])


    best_loss = checkpoint["loss"]
    for epoch in range(last_epoch,args.epoch):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            model = net,
            criterion = criterion,
            train_dataloader = train_dataloader,
            optimizer = optimizer,
            epoch = epoch,
            log_path = log_path,
            clip_max_norm = args.clip_max_norm,
        )
        loss = test_epoch(
            epoch = epoch, 
            test_dataloader = test_dataloader,
            model = net, 
            criterion = criterion,
            log_path = log_path
        )
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        save_checkpoint(
            state = {
                "epoch": epoch,
                "state_dict": net.state_dict(),
                "loss": loss,
                "optimizer": optimizer.state_dict(),
                "aux_optimizer": None,
                "lr_scheduler": lr_scheduler.state_dict(),
            },
            is_best = is_best,
            filename = check_point_path,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",type = str,default = "vic")
    parser.add_argument('--datasets_path',type=str, default="data/shuffled_data_512.npy")
    parser.add_argument('--data_size',type=int, default=600)
    parser.add_argument('--image_channel',type=int, default=3)
    parser.add_argument('--image_weight',type=int, default=512)
    parser.add_argument('--image_height',type=int, default=512)
    parser.add_argument('--lamda',type=float,default=0.0001)
    parser.add_argument('--batch_size',type=int, default=16)
    parser.add_argument('--lr',type=float,default=0.0001)
    parser.add_argument('--epoch',type=int,default=1000)
    parser.add_argument('--clip_max_norm',type=float,default=0.5)
    parser.add_argument('--factor',type=float, default=0.3)
    parser.add_argument('--patience',type=int, default=8)
    parser.add_argument('--save_model_dir',type=str,default="./model/")
    parser.add_argument('--device',type=str,default="cuda")
    args = parser.parse_args()
    main(args)