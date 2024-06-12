import datetime
from turtle import st
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
from train.config import PerceiverCfg, get_perceiver_model


def main():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    batch_size = 16  # 64/512 

    # Load the ModelNet40 dataset
    dl_train, dl_test = get_modelnet40_loaders(batch_size)

    # Create the Perceiver model
    cfg = PerceiverCfg()
    model = get_perceiver_model(cfg).to(device)

    # Parameters for training
    epochs = 120  
    lr = 1e-3
    optimizer = Lamb(params=model.parameters(), lr=lr)

    # Initialize wandb
    wandb.init(
        project="Deep Learning Exam",
        name="Perceiver: "+datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        # Track hyperparameters and run metadata
        config={
            "architecture": "Perceiver",
            "depth": cfg.depth,
            "latent_blocks": cfg.latent_blocks,
            "max_freq": cfg.max_freq,
            "num_bands": cfg.num_bands,
            "dataset": "ModelNet40",
            "epochs": epochs,
            "lr": lr,
            "optimizer": "Lamb",
            "batch_size": batch_size,
            "device": device.type
        }
    )
    wandb.watch(model, nn.CrossEntropyLoss(), log="all", log_freq=10)

    model_path = "Perceiver_bs:"+str(batch_size)+"_lr:"+str(lr)+"_epochs:"+str(epochs)+".pth"

    # Train and evaluate the model
    results = {"train": [], "val": [], "class_rep": []}
    early_stop_counter = 0
    state_dict = None
    for epoch in range(epochs):
        # Train the model 
        train_loss = train_one_epoch(model, dl_train, epoch, optimizer, device=device)

        # Evaluate the model
        val_loss, val_acc, class_rep = evaluate_epoch(model, dl_test, device=device)

        results["train"].append(train_loss)
        results["val"].append((val_loss, val_acc))
        results["class_rep"].append(class_rep)

        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save the best model
        if val_acc > max(results["val"], key=lambda x: x[1])[1]:
            state_dict = model.state_dict()

        # Early stopping
        if val_loss < min(results["val"], key=lambda x: x[0])[0]:
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter == 5:
                print("Early stopping...")
                break
        
        # Log the results
        wandb.log({"train/epoch": epoch, "train/loss": train_loss}, step=epoch)
        wandb.log({"val/epoch": epoch, "val/loss": val_loss, "val/accuracy": val_acc}, step=epoch)
        wandb.log({"class_rep/classification_report": class_rep}, step=epoch)

    wandb.unwatch(model)
    wandb.finish()

    # Save the model
    torch.save(state_dict, model_path)


def get_modelnet40_loaders(batch_size: int):
    """
    Get ModelNet40 dataset loaders

    The ModelNet40 dataset is a collection of 3D CAD models from 40 categories.
    Each model is represented as a point cloud with 2000 points.
    The dataset is split into 9,843 training samples and 2,468 testing samples.

    Every point cloud is represented as a torch_geometric.data.Data object with the following attributes:
    - pos: Tensor of shape (N, 3) where N is the number of points and 3 is the number of dimensions
    - y: Integer representing the class of the object
    - batch: Tensor of shape (N,) where N is the number of points. This is used to indicate which point belongs to which object.

    Args:
    batch_size: int: Batch size

    Returns:
    DataLoader: Training DataLoader
    DataLoader: Validation DataLoader
    DataLoader: Testing DataLoader
    """
    normalize, point_sampling = T.NormalizeScale(), T.SamplePoints(2000)

    # Lenght of ModelNet40 dataset is 9,843 for training and 2,468 for testing
    ds_train = ModelNet(root="./dataset", name="40", train=True, transform=point_sampling, pre_transform=normalize)
    ds_test = ModelNet(root="./dataset", name="40", train=False, transform=point_sampling, pre_transform=normalize)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=True)

    return dl_train, dl_test


def train_one_epoch(model: nn.Module, data: DataLoader, epoch: int, opt: optim.Optimizer, device="cuda"):
    model.train()

    losses = []

    for (i, batch) in enumerate(tqdm(data, desc=f"Training epoch {epoch}", leave=True)):
        # Get the input and target data and move it to the device
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

        # Log the loss
        losses.append(loss.item())

    # Return the average loss
    return np.mean(losses)


def evaluate_epoch(model: nn.Module, data: DataLoader, device="cuda"):
    model.eval()

    predictions = []
    targets = []
    losses = []

    # Disable gradient computation for evaluation
    with torch.no_grad():
        for (i, batch) in enumerate(tqdm(data, desc=f"Evaluating", leave=True)):
            # Get the input and target data and move it to the device
            xs, _ = to_dense_batch(batch.pos, batch=batch.batch)
            xs = xs.to(device)
            ys = batch.y.to(device)

            # Forward pass
            logits = model(xs)

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