import json
import sys
import os

import cv2
import argparse
import warnings

sys.path.append(os.getcwd())

from compressai.models import *
from compressai.zoo import models

warnings.filterwarnings("ignore", category=UserWarning)


def encode(args):
    """ device """
    device = args.device if torch.cuda.is_available() else "cpu"

    """ load png/raw data """
    x = cv2.imread(args.image_path)
    x = torch.from_numpy(x) 
    x = x.permute(2,0,1)

    """ load model """
    with open("cof/" + args.model_name + ".json", 'r') as file:
        net = models[args.model_name](**json.load(file))
    net.to(device)
    net.from_pretrain(args.pre_model_path,args.device)

    """ compress """
    net.update()
    output = net.compress(x,device=args.device)
    with open(args.output_path_y, 'wb') as file:
        file.write(output["strings"][0][0])
    with open(args.output_path_z, 'wb') as file:
        file.write(output["strings"][1][0])
    print("result has been write to",args.output_path_y,args.output_path_z)

def decode(args):
    """ device """
    device = args.device if torch.cuda.is_available() else "cpu"

    """ load model """
    with open("cof/" + args.model_name + ".json", 'r') as file:
        net = models[args.model_name](**json.load(file))
    net.to(device)
    net.from_pretrain(args.pre_model_path,args.device)

    """ load data """
    output = {"strings":[],
              "shape":net.get_z_shape()}
    # output["strings"]
    # output["strings"] = []
    y_strings = []
    z_strings = []
    with open(args.path_y, 'rb') as file:
        y_strings.append(file.read())
        output["strings"].append(y_strings)
    with open(args.path_z, 'rb') as file:
        z_strings.append(file.read())
        output["strings"].append(z_strings)

    """ decompress """
    net.update()
    image = net.decompress(**output)

    """ save """
    image = image.permute(1,2,0)
    image = image.numpy()
    cv2.imwrite(args.output_image_path, image)
    print("result image has saved to:",args.output_image_path)

def config_encoder(argv):
    parser = argparse.ArgumentParser(description="LIC encoder")
    parser.add_argument("--model_name",type = str,default = "vic")
    parser.add_argument('--image_path', type=str, default= "example\input\image.png",help='image path')
    parser.add_argument('--output_path_y', type=str, default= "example\output\image-y.bin")
    parser.add_argument('--output_path_z', type=str, default= "example\output\image-z.bin")
    parser.add_argument('--pre_model_path',type=str, default="model\lambda = 0.0016\model.pth",help='pre_model path')
    parser.add_argument('--device',type=str, default="cpu")
    args = parser.parse_args(argv)
    return args

def config_decoder(argv):
    parser = argparse.ArgumentParser(description="LIC encoder")
    parser.add_argument("--model_name",type = str,default = "vic")
    parser.add_argument('--path_y', type=str, default= "example/output/image-y.bin")
    parser.add_argument('--path_z', type=str, default= "example\output\image-z.bin")
    parser.add_argument('--output_image_path', type=str, default= "example\output\image-re.png",help='image path')
    parser.add_argument('--pre_model_path',type=str, default="model\lambda = 0.0016\model.pth",help='pre_model path')
    parser.add_argument('--device',type=str, default="cpu")
    args = parser.parse_args(argv)
    return args

def parse_args(argv):
    parser = argparse.ArgumentParser(description="codec")
    parser.add_argument("command", choices=["encode", "decode"])
    args = parser.parse_args(argv)
    return args

def main(argv):
    args = parse_args(argv[0:1])
    argv = argv[1:]
    torch.set_num_threads(1)  # just to be sure
    if args.command == "encode":
        encode(config_encoder(argv))
    elif args.command == "decode":
        decode(config_decoder(argv))

if __name__ == "__main__":
    main(sys.argv[1:])


""""
bash  
    python example/codec.py encode --model_name stf --image_path "data/image.png" --output_path_y "data/y.bin" --output_path_z "data/z.bin" --pre_model_path "model/stf/lambda = 0.01/model.pth" --device "cpu"
    python example/codec.py decode --model_name stf --path_y "data/y.bin" --path_z "data/z.bin" --output_image_path "data/re.png" --pre_model_path "model/stf/lambda = 0.01/model.pth" --device "cpu"
"""