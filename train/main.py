import datetime
import torch
import wandb
import torch.nn as nn
import numpy as np

from torch import optim
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, classification_report
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch
from tqdm import tqdm
from datetime import datetime

from lamb import Lamb
from src.perceiver import Perceiver


def main():
    # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    device = torch.device("cpu")
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

    batch_size = 64  # 512 

    # Training and test datasets
    train_dataset = ModelNet(root="./dataset", name="40", train=True)
    test_dataset = ModelNet(root="./dataset", name="40", train=False)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    input_dim = 3
    len_shape = 1
    emb_dim = 512
    latent_dim = 512
    num_classes = 40

    depth = 2  
    latent_block = 6  
    max_freq = 1120  
    num_bands = 64 
    epochs = 120
    lr = 1e-3

    model = Perceiver(
        input_dim=input_dim,
        len_shape=len_shape,
        emb_dim=emb_dim,  # 512
        latent_dim=latent_dim,  # 512
        batch_size=batch_size,
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
        (val_acc, class_rep) = evaluate_epoch(model, test_dataloader, epoch, device = device)
        losses_accs.append((loss, val_acc))
        class_reports.append(class_rep)

        wandb.log({"epoch": epoch, "loss": loss, "val_acc": val_acc, "class_report": class_rep})
        print(f"Epoch {epoch}: Loss: {loss:.4f}, Val Acc: {val_acc:.4f}")

    wandb.unwatch(model)
    wandb.finish()


def train_epoch(model, data, opt, epoch, scheduler = None, device="cuda"):
    model.train()
    losses = []

    for (i, batch) in enumerate(tqdm(data, desc=f"Training epoch {epoch}", leave=True)):
        xs, mask = to_dense_batch(batch.pos, batch=batch.batch)
        xs, mask = xs.to(device), mask.to(device)
        ys = batch.y.to(device)

        # Zero out the gradients
        opt.zero_grad()

        # Forward pass
        logits = model(xs, mask)

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


def evaluate_epoch(model, data, epoch, device="cuda"):
    model.eval()
    preds = []
    targets = []

    # Disable gradient computation for evaluation
    with torch.no_grad():
        for (i, batch) in enumerate(tqdm(data, desc=f"Evaluating epoch {epoch}", leave=False)):
            xs, mask = to_dense_batch(batch.pos, batch=batch.batch)
            xs, mask = xs.to(device), mask.to(device)
            ys = batch.y.to(device)

            # Forward pass
            logits = model(xs, mask)

            # Get the predicted classes
            pred = logits.argmax(dim=-1)

            # Append the predictions and targets
            preds.append(pred)
            targets.append(ys.detach().cpu().numpy())
        
    # Concatenate the predictions and targets
    preds = np.hstack(preds)
    targets = np.hstack(targets)

    # Compute the accuracy
    accuracy = accuracy_score(targets, preds)
    class_report = classification_report(targets, preds, zero_division=0, digits=3, output_dict=True)
    return accuracy, class_report


if __name__ == '__main__':
    main()