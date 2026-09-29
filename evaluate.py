"""Evaluate a trained segmentation checkpoint with Dice and IoU."""
import argparse, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.data import SegmentationDataset
from src.metrics import dice_score, iou_score
from src.model import UNet


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = SegmentationDataset(args.images, args.masks, args.size, augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    model = UNet(num_classes=2).to(device)
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state", ckpt)); model.eval()
    dices, ious = [], []
    with torch.no_grad():
        for batch in loader:
            x, y = batch["image"].to(device), batch["mask"].to(device)
            pred = model(x).argmax(1)
            for p, t in zip(pred, y):
                dices.append(dice_score(p, t).item()); ious.append(iou_score(p, t).item())
    result = {"samples": len(ds), "dice": sum(dices)/len(dices), "iou": sum(ious)/len(ious)}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2))

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--model",default="checkpoints/best.pt"); p.add_argument("--images",default="data/images"); p.add_argument("--masks",default="data/masks"); p.add_argument("--output",default="results/metrics.json"); p.add_argument("--batch-size",type=int,default=4); p.add_argument("--size",type=int,default=256); main(p.parse_args())
