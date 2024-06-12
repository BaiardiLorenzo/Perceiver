import datetime
from matplotlib.pyplot import step
import torch
import wandb
import torch.nn as nn
import numpy as np

from torch import optim
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, classification_report
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch_geometric.utils import to_dense_batch
from tqdm import tqdm
from datetime import datetime

from lamb import Lamb
from src.perceiver import Perceiver


def main():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
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

    batch_size = 32  # 64/512 

    pre_transform, transform = T.NormalizeScale(), T.SamplePoints(2000)

    # Training and test datasets
    train_dataset = ModelNet(root="./dataset", name="40", train=True, transform=transform, pre_transform=pre_transform)
    test_dataset = ModelNet(root="./dataset", name="40", train=False, transform=transform, pre_transform=pre_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    input_dim = 3
    len_shape = 1
    emb_dim = 512  # 512
    latent_dim = 512  # 512
    num_classes = 40

    depth = 2  # 2  
    latent_block = 6 # 6  
    max_freq = 1120  # 1120  
    num_bands = 64  # 64 
    epochs = 50  # 120
    lr = 1e-3

    model = Perceiver(
        input_dim=input_dim,
        len_shape=len_shape,
        emb_dim=emb_dim,
        latent_dim=latent_dim,
        num_classes=num_classes,
        depth=depth,
        latent_blocks=latent_block,
        heads=8,
        fourier_encode=True,
        max_freq=max_freq,
        num_bands=num_bands
    ).to(device)

    optimizer = Lamb(params=model.parameters(), lr=lr)

    wandb.init(
        project="Deep Learning Exam",
        name="Perceiver: "+datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        # Track hyperparameters and run metadata
        config={
            "architecture": "Perceiver",
            "dataset": "ModelNet40",
            "depth": depth,
            "latent_blocks": latent_block,
            "max_freq": max_freq,
            "num_bands": num_bands,
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "optimizer": "Lamb",
            "device": device.type
        }
    )
    wandb.watch(model, nn.CrossEntropyLoss(), log="all")

    losses_accs = []
    class_reports = []
    for epoch in range(epochs):
        loss = train_epoch(model, train_dataloader, optimizer, epoch, device = device)
        (val_acc, class_rep, val_loss) = evaluate_epoch(model, test_dataloader, device = device)
        losses_accs.append((loss, val_acc))
        class_reports.append(class_rep)

        wandb.log({"train/epoch": epoch, "train/loss": loss}, step=epoch)
        wandb.log({"val/epoch": epoch, "val/loss": val_loss, "val/accuracy": val_acc, "classification_report/classification_report": class_rep}, step=epoch)
        print(f"Epoch {epoch}: Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    wandb.unwatch(model)
    wandb.finish()


def train_epoch(model, data, opt, epoch, scheduler=None, device="cuda"):
    model.train()
    losses = []

    for (i, batch) in enumerate(tqdm(data, desc=f"Training epoch {epoch}", leave=True)):
        xs, _ = to_dense_batch(batch.pos, batch=batch.batch)
        xs = xs.to(device)
        ys = batch.y.to(device)

        # Zero out the gradients
        opt.zero_grad()

        # Forward pass
        logits = model(xs)

        # Compute the cross entropy loss
        loss = F.cross_entropy(logits, ys)
        
        # Backward pass
        loss.backward()

        # Update the model parameters
        opt.step()

        # Update the learning rate if a scheduler is provided
        if scheduler is not None:
            scheduler.step()

        # Log the loss
        losses.append(loss.item())

    # Return the average loss
    return np.mean(losses)


def evaluate_epoch(model, data, device="cuda"):
    model.eval()
    predictions = []
    targets = []
    losses = []

    # Disable gradient computation for evaluation
    with torch.no_grad():
        for (i, batch) in enumerate(tqdm(data, desc=f"Evaluating", leave=True)):
            xs, mask = to_dense_batch(batch.pos, batch=batch.batch)
            xs, mask = xs.to(device), mask.to(device)
            ys = batch.y.to(device)

            # Forward pass
            logits = model(xs, mask)

            # Compute the cross entropy loss
            loss = F.cross_entropy(logits, ys)

            # Get the predicted classes
            pred = torch.argmax(logits, 1)

            losses.append(loss.item())

            # Append the predictions and targets
            targets.append(ys.detach().cpu().numpy())
            predictions.append(pred.detach().cpu().numpy())
        
    # Concatenate the predictions and targets
    predictions = np.hstack(predictions)
    targets = np.hstack(targets)

    # Compute the accuracy
    accuracy = accuracy_score(targets, predictions)
    class_report = classification_report(targets, predictions, zero_division=0, digits=3, output_dict=True)
    return accuracy, class_report, np.mean(losses)


if __name__ == '__main__':
    main()