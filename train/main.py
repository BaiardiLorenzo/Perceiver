import torch

import torch_optimizer as optim
from data import get_modelnet40_loaders
from src.config import PerceiverModelNet40Cfg, get_perceiver_model
from train import train_evaluate_model


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    batch_size = 16  # 512 

    # Load the ModelNet40 dataset
    dl_train, dl_test = get_modelnet40_loaders(batch_size)

    # Create the Perceiver model
    cfg = PerceiverModelNet40Cfg()
    model = get_perceiver_model(cfg).to(device)

    # Parameters for training
    epochs = 120  
    lr = 1e-3

    optimizer = optim.Lamb(params=model.parameters(), lr=lr)

    train_evaluate_model(model, "ModelNet40", dl_train, dl_test, batch_size, lr, epochs, optimizer, sched=None, early_stop=True, device=device)


if __name__ == '__main__':
    main()