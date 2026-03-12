"""Command-line entrypoint for training the CV starter model.

The script ties all pieces together: it loads hyperparameters from YAML, builds
the dataloaders, instantiates a pretrained backbone, and handles the training
loop with logging, learning-rate scheduling, early stopping, and checkpointing.
"""

import argparse
import csv
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

from utils import set_seed, get_device, load_yaml, save_checkpoint, save_json
from data import build_dataloaders
from model import build_model


# ---------------------------------------------------
# TRAIN
# ---------------------------------------------------

def train_one_epoch(model, loader, crit, opt, device):
    """Run one training epoch and return loss + accuracy."""
    
    model.train()

    loss_sum = 0.0
    correct = 0
    n = 0

    for x, y in tqdm(loader, desc="train", leave=False):

        x, y = x.to(device), y.to(device)

        opt.zero_grad(set_to_none=True)

        logits = model(x)

        loss = crit(logits, y)

        loss.backward()

        opt.step()

        preds = logits.argmax(dim=1)

        correct += (preds == y).sum().item()

        loss_sum += loss.item() * y.size(0)
        n += y.size(0)

    train_loss = loss_sum / n
    train_acc = correct / n

    return train_loss, train_acc


# ---------------------------------------------------
# VALIDATION
# ---------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, crit, device, num_classes):
    """Evaluate the model on validation set."""

    model.eval()

    acc_metric = MulticlassAccuracy(num_classes=num_classes).to(device)

    loss_sum = 0.0
    n = 0

    for x, y in tqdm(loader, desc="val", leave=False):

        x, y = x.to(device), y.to(device)

        logits = model(x)

        loss = crit(logits, y)

        loss_sum += loss.item() * y.size(0)
        n += y.size(0)

        acc_metric.update(logits, y)

    val_loss = loss_sum / n
    val_acc = acc_metric.compute().item()

    return val_loss, val_acc


# ---------------------------------------------------
# MAIN TRAINING PIPELINE
# ---------------------------------------------------

def main(cfg_path):

    cfg = load_yaml(cfg_path)

    set_seed(cfg["seed"])

    device = get_device()

    print(f"Using device: {device}")

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    # log file
    (out / "log.csv").write_text(
        "epoch,train_loss,train_acc,val_loss,val_acc,lr\n"
    )

    train_loader, val_loader, num_classes, classes = build_dataloaders(cfg)

    model = build_model(cfg, num_classes).to(device)

    crit = nn.CrossEntropyLoss()

    params = [p for p in model.parameters() if p.requires_grad]

    # optimizer
    if cfg["train"]["optimizer"].lower() == "sgd":

        opt = SGD(
            params,
            lr=cfg["train"]["lr"],
            momentum=cfg["train"]["momentum"],
            weight_decay=cfg["train"]["weight_decay"],
        )

    else:

        opt = AdamW(
            params,
            lr=cfg["train"]["lr"],
            weight_decay=cfg["train"]["weight_decay"],
        )

    # scheduler
    if cfg["train"]["scheduler"] == "cosine":

        sch = CosineAnnealingLR(
            opt,
            T_max=cfg["train"]["t_max"]
        )

    else:

        sch = None

    best_acc = 0.0
    best_path = out / "best.pt"

    patience = cfg["early_stopping"]["patience"]
    waited = 0

    print("\nStarting training...\n")

    for epoch in range(1, cfg["train"]["epochs"] + 1):

        print(f"Epoch {epoch}/{cfg['train']['epochs']}")

        # train
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            crit,
            opt,
            device,
        )

        # validation
        val_loss, val_acc = evaluate(
            model,
            val_loader,
            crit,
            device,
            num_classes,
        )

        # scheduler step
        if sch:
            sch.step()

        # learning rate
        lr = opt.param_groups[0]["lr"]

        # console log
        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"LR: {lr:.6f}"
        )

        # save log
        with open(out / "log.csv", "a", newline="") as f:

            csv.writer(f).writerow([
                epoch,
                f"{train_loss:.4f}",
                f"{train_acc:.4f}",
                f"{val_loss:.4f}",
                f"{val_acc:.4f}",
                f"{lr:.6f}",
            ])

        # early stopping + checkpoint
        if val_acc > best_acc + cfg["early_stopping"]["min_delta"]:

            best_acc = val_acc
            waited = 0

            save_checkpoint(model, best_path)

            print("New best model saved")

        else:

            waited += 1

            print(f"No improvement ({waited}/{patience})")

            if waited > patience:

                print("\nEarly stopping triggered\n")
                break

        print("-" * 50)

    save_json(
        {
            "best_val_acc": best_acc,
            "classes": classes,
        },
        out / "metrics.json",
    )

    print(
        f"\nTraining finished\nBest validation accuracy: {best_acc:.4f}\nCheckpoint saved at: {best_path}"
    )


# ---------------------------------------------------
# CLI
# ---------------------------------------------------

if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        type=str,
        default="configs/cv_cifar10.yaml",
    )

    args = ap.parse_args()

    main(args.config)
