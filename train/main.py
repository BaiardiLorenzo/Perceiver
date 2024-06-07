import torch
import wandb
import torch.nn as nn
from tqdm import tqdm

from src.perceiver import Perceiver
from lamb import LAMB
from torch.utils.data import DataLoader

from torch_geometric.datasets import ModelNet


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    """
    We used an architecture with 2 cross-attentions and 6 self-
    attention layers for each block and otherwise used the same
    architectural settings as ImageNet. We used a higher max-
    imum frequency than for image data to account for the
    irregular sampling structure of point clouds - we used a max
    frequency of 1120 (10×the value used on ImageNet). We
    obtained the best results using 64 frequency bands, and we 
    noticed that values higher than 256 generally led to more
    severe overfitting. We used a batch size of 512 and trained
    with LAMB with a constant learning rate of 1 ×10−3: mod-
    els saturated in performance within 50,000 training steps
    """

    cross_attentions = 2
    self_attentions_for_each_block = 6
    max_freq = 1120
    num_bands = 64
    batch_size = 512 # TODO: Check if this batch size fits 
    optimizer = LAMB() # The LAMB optimizer have a default learning rate of 1e-3
    loss = nn.CrossEntropyLoss()

    loss = 0
    epochs = 10

    model = Perceiver().to(device)

    # Training and test datasets
    train_dataset = ModelNet(root="./dataset", name="40", train=True)
    test_dataset = ModelNet(root="./dataset", name="40", train=False)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    wandb.init(
        project="Perceiver",
        name="Train Model",
        # entity=,
        id="",
        notes="",
        tags=[]
    )

    for epoch in range(epochs):
        train_epoch(model, train_dataloader, optimizer, epoch, device)
        evaluate_epoch(model, test_dataloader, epoch, device)

    wandb.finish()


def train_epoch(model, data, loss, optimizer, scheduler, epoch, device):
    model.train()

    for batch in tqdm(data, desc=f"Epoch {epoch}"):
        optimizer.zero_grad()

        x = batch.to(device)
        y = model(x)

        loss.backward()

        optimizer.step()
        scheduler.step()

        wandb.log({"loss": loss.item()})

def evaluate_epoch(model, data, loss, epoch, device):
    model.eval()

    for batch in tqdm(data, desc=f"Epoch {epoch}"):
        x = batch.to(device)
        y = model(x)

        wandb.log({"loss": loss.item()})
    


if __name__ == '__main__':
    main()