"""Script for training KWT model"""
from argparse import ArgumentParser
from config_parser import get_config
import os
import yaml

import wandb

from utils.loss import LabelSmoothingLoss
from utils.dataset import get_loader
from utils.misc import seed_everything, count_params, get_model
from utils.misc import seed_everything, count_params, get_model, calc_step, log
import torch
from torch import nn
from typing import Callable, Tuple
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

@torch.no_grad()
def evaluate(net: nn.Module, criterion: Callable, dataloader: DataLoader, device: torch.device) -> Tuple[float, float]:
    net.eval()
    correct = 0
    running_loss = 0.0

    for data in dataloader:
        print(data)
        data = data.to(device)
        out = net(data)

        #I wrote this
        print(out)

        temp = out.argmax(1)
        print(temp)



def inference_pipeline(config):

    config["exp"]["save_dir"] = os.path.join(config["exp"]["exp_dir"], config["exp"]["exp_name"])
    os.makedirs(config["exp"]["save_dir"], exist_ok=True)
    
    ######################################
    # save hyperparameters for current run
    ######################################

    config_str = yaml.dump(config)
    print("Using settings:\n", config_str)

    with open(os.path.join(config["exp"]["save_dir"], "settings.txt"), "w+") as f:
        f.write(config_str)
    

    #####################################
    # initialize inference items
    #####################################
    model = get_model(config["hparams"]["model"])
    print("THIS IS JUST FOR ")
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint {args.ckpt}.")
    model = model.to(config["hparams"]["device"])
    
    print(f"Created model with {count_params(model)} parameters.")

    # data
        #inference_list
    with open(config["inference_list_file"],"r") as f:
        inference_list = f.read().rstrip().split("\n")

        print("START PRINTING")
        print(inference_list)
    
    #loss
    if config["hparams"]["l_smooth"]:
        criterion = LabelSmoothingLoss(num_classes=config["hparams"]["model"]["num_classes"], smoothing=config["hparams"]["l_smooth"])
    else:
        criterion = nn.CrossEntropyLoss()

    #inferenceloader 
    inferenceloader = get_loader(inference_list, config, train = False)
    #load
    ckpt = torch.load(os.path.join(config["exp"]["save_dir"], "best.pth"))
    model.load_state_dict(ckpt["model_state_dict"])
    print("Best ckpt loaded.")

    result = evaluate(model, criterion, inferenceloader, config["hparams"]["device"])
    print(result)


def main2(args):
    config = get_config(args.conf)
    seed_everything(config["hparams"]["seed"])
    inference_pipeline(config)

if __name__ == "__main__":
    parser = ArgumentParser("Driver code.")
    parser.add_argument("--conf", type=str, required=True, help="Path to config.yaml file.")
    parser.add_argument("--ckpt", type=str, required=False, help="Path to checkpoint file.", default=None)
    args = parser.parse_args()

    #main(args)
    main2(args)
