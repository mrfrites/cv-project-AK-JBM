"""
Lightweight sanity check for the CV starter.

Loads CIFAR-10 using the configured transforms, runs a single forward/backward
pass, and stores metrics under outputs/smoke_metrics.json. Helpful for CPU-only
validation before longer GPU runs (e.g., on Colab/Kaggle T4).
"""

from pathlib import Path
import torch
import torch.nn as nn

from utils import load_yaml, set_seed, get_device, save_json
from data import build_dataloaders
from model import build_model


def run_smoke(cfg_path: str = "configs/cv_cifar10_fast.yaml") -> Path:
    cfg = load_yaml(cfg_path)
    set_seed(cfg["seed"])
    device = get_device()

    # --------------------------
    # 1️⃣ Charger le dataloader
    # --------------------------
    train_loader, _, num_classes, classes = build_dataloaders(cfg)
    model = build_model(cfg, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    # --------------------------
    # 2️⃣ Afficher les premières images du batch pour vérification
    # --------------------------
    import matplotlib.pyplot as plt
    x, y = next(iter(train_loader))  # un batch
    # Dénormalisation pour affichage
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    x_vis = x[:5] * IMAGENET_STD + IMAGENET_MEAN  # les 5 premières images
    x_vis = x_vis.permute(0,2,3,1).numpy()  # [B,H,W,C] pour matplotlib

    fig, axes = plt.subplots(1, 5, figsize=(12,3))
    for i in range(5):
        axes[i].imshow(x_vis[i])
        axes[i].set_title(classes[y[i]])
        axes[i].axis('off')
    plt.show()

    # --------------------------
    # 3️⃣ Continuer le smoke test
    # --------------------------
    x, y = x.to(device), y.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "smoke_metrics.json"
    save_json(
        {
            "loss": float(loss.item()),
            "batch_size": int(x.size(0)),
            "num_classes": num_classes,
            "device": str(device),
        },
        out_path,
    )
    return out_path
