from sympy import per
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import wandb
from datetime import datetime

from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch

from torch.nn import functional as F
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm


def train_evaluate_model(
        model: nn.Module, 
        dataset_name: str,
        dl_train: DataLoader, 
        dl_test: DataLoader, 
        batch_size: int, 
        lr: int, 
        epochs: int, 
        optimizer: optim.Optimizer, 
        sched: optim.lr_scheduler = None, 
        early_stop: bool = False,
        device="cuda",
    ):

    model_path = "Perceiver::"+datetime.now().strftime("%Y-%m-%d %H:%M:%S")+"_"+dataset_name+"_bs:"+str(batch_size)+"_lr:"+str(lr)+"_epochs:"+str(epochs)

    # Initialize wandb
    wandb.init(
        project="Deep Learning Exam",
        name=model_path,
        # Track hyperparameters and run metadata
        config={
            "architecture": "Perceiver",
            "dataset": dataset_name,
            "epochs": epochs,
            "lr": lr,
            "optimizer": "Lamb",
            "batch_size": batch_size,
            "device": device.type
        }
    )
    wandb.watch(model, nn.CrossEntropyLoss(), log="all", log_freq=10)

    # Train and evaluate the model
    results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "class_rep": []}
    epochs_done = 0
    early_stopping = EarlyStopping(patience=10, monitor='val_loss')
    state_dict = None

    for epoch in range(epochs):
        # Train the model 
        train_loss, train_acc, _ = train_one_epoch(model, dl_train, epoch, optimizer, sched, device=device)

        # Evaluate the model
        val_loss, val_acc, class_rep = evaluate_epoch(model, dl_test, device=device)

        epochs_done += 1

        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save the best model
        if val_acc > max(results["val_acc"], default=0):
            state_dict = model.state_dict()
        
        # Append the results
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_loss)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
        results["class_rep"].append(class_rep)

        # Log the results
        wandb.log({"train/epoch": epoch, "train/loss": train_loss, "train/accuracy": train_acc}, step=epoch)
        wandb.log({"val/epoch": epoch, "val/loss": val_loss, "val/accuracy": val_acc}, step=epoch)
        wandb.log({"class_rep/classification_report": class_rep}, step=epoch)

        # Check early stopping criteria
        early_stopping(val_loss)
        if early_stopping.early_stop and early_stop:
            print("Early stopping")
            break

    wandb.unwatch(model)
    wandb.finish()

    model_states_path = "train/model_states/"+model_path+".pth"
    # model_states_path = model_path+".pth"

    # Save the best model
    if state_dict is not None:
        torch.save(state_dict, model_states_path)
        print(f"Model saved to {model_states_path}")


def train_one_epoch(model: nn.Module, data: DataLoader, epoch: int, opt: optim.Optimizer, sched: optim.lr_scheduler = None, device="cuda"):
    model.train()

    losses = []

    targets = []
    predictions = []

    for (i, batch) in enumerate(tqdm(data, desc=f"Training epoch {epoch}", leave=True)):
        # Get the input and target data and move it to the device
        if hasattr(batch, "pos"):
            xs, mask = to_dense_batch(batch.pos, batch=batch.batch)
        xs = xs.to(device)
        ys = batch.y.to(device)

        # Zero out the gradients
        opt.zero_grad()

        # Forward pass
        logits = model(xs)

        # Compute the cross entropy loss
        loss = F.cross_entropy(logits, ys)

        # Get the predicted classes
        pred = torch.argmax(logits, 1)
        
        # Backward pass
        loss.backward()

        # Update the model parameters
        opt.step()

        # Update the learning rate
        if sched is not None:
            sched.step()

        # Log the loss
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

    # Return the average loss
    return np.mean(losses), accuracy, class_report


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
    return np.mean(losses), accuracy, class_report


class EarlyStopping:
    def __init__(self, patience=5, delta=0, monitor='val_loss'):
        self.patience = patience
        self.delta = delta
        self.monitor = monitor
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_metric):
        if self.best_score is None:
            self.best_score = val_metric
        elif val_metric > self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_metric
            self.counter = 0