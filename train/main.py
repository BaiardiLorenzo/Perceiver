import torch

import torch_optimizer as optim
from data import get_modelnet40_loaders
from src.config import PerceiverModelNet40Cfg, get_perceiver_model
from train import train_evaluate_model


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    batch_size = 16 # 512 
    gradient_acc_steps = 1  # 32 x 16 = 512

    # Load the ModelNet40 dataset
    dataset_name = "ModelNet40"
    dl_train, dl_test, train_trans = get_modelnet40_loaders(batch_size)

    # Create the Perceiver model
    model, cfg = get_perceiver_model(PerceiverModelNet40Cfg(), device)
    
    # Parameters for training
    early_stop = False
    epochs = 100  
    lr = 1e-3

    opt = optim.Lamb(params=model.parameters(), lr=lr)

    train_evaluate_model(
        model = model, 
        cfg = cfg,
        dataset_name = dataset_name,
        dl_train = dl_train, 
        dl_test = dl_test, 
        train_trans = str(train_trans),
        batch_size = batch_size, 
        gas = gradient_acc_steps,
        lr = lr, 
        epochs = epochs, 
        opt = opt, 
        early_stop = early_stop, 
        device = device,
    )


if __name__ == '__main__':
    main()