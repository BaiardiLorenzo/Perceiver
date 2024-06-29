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
        cfg,
        dataset_name: str,
        class_names: list,
        dl_train: DataLoader, 
        dl_test: DataLoader, 
        train_trans: str,
        batch_size: int, 
        gas: int,
        lr: int, 
        epochs: int, 
        opt: optim.Optimizer, 
        early_stop: bool = False,
        device="cuda",
    ):

    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    test_name = "Perceiver::"+time+"_"+dataset_name+"_epochs:"+str(epochs)+"_bs:"+str(batch_size)+"_gradaccu:"+str(gas)+"_lr:"+str(lr)
    test_name_cfg = test_name + "_cfg:" + str(cfg) + "_dataset_transform:" + train_trans

    wandb_init(test_name_cfg, model, cfg, dataset_name, train_trans, epochs, batch_size, gas, lr, device)

    # Train and evaluate the model
    train_results = {"loss": [], "acc": [], "class_rep": []}
    val_results = {"loss": [], "acc": [], "class_rep": []}
    max_val_acc = 0
    early_stopping = EarlyStopping(patience=10, monitor='val_loss', active=early_stop)
    state_dict = None

    for epoch in range(epochs):
        # Train the model 
        train_loss, train_acc, train_class_rep = train_epoch(model, dl_train, class_names, epoch, gas, opt, device=device)

        # Evaluate the model
        val_loss, val_acc, val_class_rep = evaluate_epoch(model, dl_test, class_names, epoch, device=device)

        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save the best model
        if val_acc > max_val_acc:
            max_val_acc = val_acc
            state_dict = model.state_dict()
        
        # Append the results
        train_results["loss"].append(train_loss)
        train_results["acc"].append(train_acc)
        train_results["class_rep"].append(train_class_rep)
        val_results["loss"].append(val_loss)
        val_results["acc"].append(val_acc)
        val_results["class_rep"].append(val_class_rep)

        # Check early stopping criteria
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    wandb.unwatch(model)
    wandb.finish()

    model_states_path = "train/model_states/"+test_name+".pth"

    # Save the best model
    if state_dict is not None:
        torch.save(state_dict, model_states_path)
        print(f"Model saved to {model_states_path}")


def wandb_init(test_name, model, cfg, dataset, train_trans, epochs, bs, gas, lr, device, project="Deep Learning Exam"):
    # Initialize wandb
    wandb.init(
        project=project,
        name=test_name,
        # Track hyperparameters and run metadata
        config={
            "architecture": "Perceiver",
            "dataset": dataset,
            "dataset_transform": train_trans,
            "latent_length": cfg.latent_length,
            "latent_dim": cfg.latent_dim,
            "latent_blocks": cfg.latent_blocks,
            "heads": cfg.heads,
            "perceiver_block": cfg.perceiver_block,
            "share_weights": cfg.share_weights,
            "ff_pos_encoding": cfg.ff_pos_encoding,
            "max_freq": cfg.max_freq,
            "num_bands": cfg.num_bands,
            "epochs": epochs,
            "batch_size": bs,
            "gradient_accumulation": gas,
            "lr": lr,
            "optimizer": "Lamb",
            "device": device.type
        }
    )
    wandb.watch(model, nn.CrossEntropyLoss(), log="all")


def train_epoch(model: nn.Module, data: DataLoader, class_names: list, epoch: int, gas: int, opt: optim.Optimizer, device="cuda"):
    model.train()

    losses = []
    gts = []
    preds = []

    # Zero out the gradients
    opt.zero_grad()
    
    for (i, batch) in enumerate(tqdm(data, desc=f"Training epoch {epoch}", leave=True)):
        # Get the input and target data and move it to the device
        xs, _ = to_dense_batch(batch.pos, batch=batch.batch)
        xs, ys = xs.to(device), batch.y.to(device)

        # Forward pass
        logits = model(xs)

        # Get the predicted classes
        pred = torch.argmax(logits, 1)
        
        # Compute the cross entropy loss
        loss = F.cross_entropy(logits, ys)/gas

        # Backward pass
        loss.backward()

        # Gradient accumulation
        if (i+1) % gas == 0 or (i+1) == len(data):
            # Update the model parameters
            opt.step()

            # Zero out the gradients
            opt.zero_grad()

        # Append the loss, predictions and ground truths
        losses.append(loss.item())
        preds.extend(pred.detach().cpu().numpy())
        gts.extend(ys.detach().cpu().numpy())

    # Compute the accuracy and classification report
    accuracy = accuracy_score(gts, preds)
    class_report = classification_report(gts, preds, zero_division=0, digits=3, output_dict=True)
    class_report_db = classification_report(gts, preds, zero_division=0, digits=3).splitlines()

    report_table = []
    for line in class_report_db[2:(len(class_names)+2)]:
       report_table.append(line.split())

    # Log the results
    loss = np.mean(losses)
    wandb.log({
        "train/epoch": epoch, 
        "train/loss": loss, 
        "train/accuracy": accuracy, 
        "train_class_rep/classification_report": class_report,
        "train/confusion_matrix": wandb.plot.confusion_matrix(y_true=gts, preds=preds, class_names=class_names),
        "train/classification_report": wandb.Table(data=report_table, columns=["Class", "Precision", "Recall", "F1-score", "Support"])
        },
        step=epoch)

    # Return the average loss, accuracy and classification report
    return loss, accuracy, class_report


def evaluate_epoch(model: nn.Module, data: DataLoader, class_names: list, epoch: int, device="cuda"):
    model.eval()

    losses = []
    preds = []
    gts = []
    # Disable gradient computation for evaluation
    with torch.no_grad():
        for (_, batch) in enumerate(tqdm(data, desc=f"Evaluating", leave=True)):
            # Get the input and target data and move it to the device
            xs, _ = to_dense_batch(batch.pos, batch=batch.batch)
            xs, ys = xs.to(device), batch.y.to(device)

            # Forward pass
            logits = model(xs)

            # Compute the cross entropy loss
            loss = F.cross_entropy(logits, ys)

            # Get the predicted classes
            pred = torch.argmax(logits, 1)

            # Append the loss, predictions and ground truths
            losses.append(loss.item())
            preds.extend(pred.detach().cpu().numpy())
            gts.extend(ys.detach().cpu().numpy())

    # Compute the accuracy and classification report
    accuracy = accuracy_score(gts, preds)
    class_report = classification_report(gts, preds, zero_division=0, digits=3, output_dict=True)

    class_report_db = classification_report(gts, preds, zero_division=0, digits=3).splitlines()
    report_table = []
    for line in class_report_db[2:(len(class_names)+2)]:
        report_table.append(line.split())

    # Log the results
    loss = np.mean(losses)
    wandb.log({
        "val/epoch": epoch, 
        "val/loss": loss, 
        "val/accuracy": accuracy,
        "val_class_rep/classification_report": class_report,
        "val/confusion_matrix": wandb.plot.confusion_matrix(y_true=gts, preds=preds, class_names=class_names),
        "val/classification_report": wandb.Table(data=report_table, columns=["Class", "Precision", "Recall", "F1-score", "Support"])
        }, 
        step=epoch)
    
    return loss, accuracy, class_report


class EarlyStopping:
    def __init__(self, patience=5, delta=0, monitor='val_loss', active=True):
        self.patience = patience
        self.delta = delta
        self.monitor = monitor
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.active = active

    def __call__(self, val_metric):
        if self.best_score is None:
            self.best_score = val_metric
        elif val_metric > self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and self.active:
                self.early_stop = True
        else:
            self.best_score = val_metric
            self.counter = 0