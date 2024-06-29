import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from src.config import PerceiverModelNet40Cfg
from src.config import get_perceiver_model

from data import get_modelnet40_loaders

def train_evaluate_model(
        model: nn.Module, 
        cfg,
        dataset_name: str,
        class_names: list,
        dl_test: DataLoader, 
        train_trans: str,
        batch_size: int, 
        gas: int,
        lr: int, 
        epochs: int, 
        device="cuda",
    ):

    test_name = "Perceiver::CONFUSION_MATRIX_FF"
    test_name_cfg = test_name 

    wandb_init(test_name_cfg, model, cfg, dataset_name, train_trans, epochs, batch_size, gas, lr, device)

    # Evaluate the model
    _, _, val_class_rep = evaluate_epoch(model, dl_test, class_names, 1, device=device)

    print(f'Accuracy report on TEST:\n {val_class_rep}')

    wandb.unwatch(model)
    wandb.finish()


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


def load_confusion_matrix():
    device = torch.device("cpu")
    path = "train/model_states/Perceiver(NO_FF)::2024-06-25 16:39:54_ModelNet40_epochs:100_bs:16_gradaccu:1_lr:0.001.pth"
    model, _ = get_perceiver_model(PerceiverModelNet40Cfg(), device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()

    # Load the dataset
    _, dl_test, _, _ = get_modelnet40_loaders(16)

    # Get the class names
    modelnet40_class_names = ["airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl", "car", "chair", "cone", "cup", "curtain", "desk", "door", "dresser", "flower_pot", "glass_box", "guitar", "keyboard", "lamp", "laptop", "mantel", "monitor", "night_stand", "person", "piano", "plant", "radio", "range_hood", "sink", "sofa", "stairs", "stool", "table", "tent", "toilet", "tv_stand", "vase", "wardrobe", "xbox"]

    train_evaluate_model(model, PerceiverModelNet40Cfg(), "ModelNet40", modelnet40_class_names, dl_test, "None", 16, 1, 0, 0, device)


if __name__ == "__main__":
    load_confusion_matrix()
