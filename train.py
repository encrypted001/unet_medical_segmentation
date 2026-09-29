"""Train the Mohit Kumar medical image segmentation model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from src.data import SegmentationDataset
from src.metrics import dice_score, dice_loss
from src.model import UNet


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def main(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SegmentationDataset(args.images, args.masks, args.size, augment=True)
    val_count = max(1, int(len(dataset) * args.validation))
    train_count = len(dataset) - val_count
    if train_count < 1:
        raise ValueError("Dataset is too small for the requested validation split.")
    train_ds, val_ds = random_split(dataset, [train_count, val_count], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    model = UNet(num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    history = []; best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train(); running = 0.0
        for batch in train_loader:
            x, y = batch["image"].to(device), batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y) + dice_loss(logits.softmax(1)[:, 1], y)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            running += loss.item()
        model.eval(); scores = []
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch["image"].to(device), batch["mask"].to(device)
                pred = model(x).argmax(1); scores.append(dice_score(pred, y).item())
        val_dice = float(np.mean(scores))
        row = {"epoch": epoch, "train_loss": running / max(1, len(train_loader)), "val_dice": val_dice}
        history.append(row); print(row)
        checkpoint = {"model_state": model.state_dict(), "epoch": epoch, "val_dice": val_dice}
        torch.save(checkpoint, out / "last.pt")
        if val_dice > best:
            best = val_dice; torch.save(checkpoint, out / "best.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Best validation Dice: {best:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train Mohit Kumar's medical segmentation U-Net")
    p.add_argument("--images", default="data/images"); p.add_argument("--masks", default="data/masks")
    p.add_argument("--output", default="checkpoints"); p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4); p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--validation", type=float, default=0.2); p.add_argument("--size", type=int, default=256)
    p.add_argument("--workers", type=int, default=0); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true")
    main(p.parse_args())
