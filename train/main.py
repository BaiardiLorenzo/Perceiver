import sched
import torch

from datetime import datetime

from data import get_modelnet40_loaders, get_imagenet_loaders
from lamb import Lamb
from config import PerceiverModelnet40Cfg, PerceiverImageNetCfg, get_perceiver_model
from train import train_evaluate_model


def train_modelnet40():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    batch_size = 16  # 64/512 

    # Load the ModelNet40 dataset
    dl_train, dl_test = get_modelnet40_loaders(batch_size)

    # Create the Perceiver model
    cfg = PerceiverModelnet40Cfg()
    model = get_perceiver_model(cfg).to(device)

    # Parameters for training
    epochs = 120  
    lr = 1e-3
    optimizer = Lamb(params=model.parameters(), lr=lr)

    train_evaluate_model(model, "ModelNet40", dl_train, dl_test, batch_size, lr, epochs, optimizer, sched=None, early_stop=False, device=device)


def train_imagenet():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    batch_size = 16  # 64/512 

    # Load the ImageNet dataset
    dl_train, dl_test = get_imagenet_loaders(batch_size)

    # Create the Perceiver model
    cfg = PerceiverImageNetCfg()
    model = get_perceiver_model(cfg).to(device)

    # Parameters for training
    epochs = 120  
    lr = 4e-3
    optimizer = Lamb(params=model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=[84, 102, 114] , gamma=0.1)

    train_evaluate_model(model, "ImageNet", dl_train, dl_test, batch_size, lr, epochs, optimizer, sched=None, device=device)


if __name__ == '__main__':
    train_modelnet40()