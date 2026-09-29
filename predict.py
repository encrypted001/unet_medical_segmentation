"""Run segmentation inference on one or more images."""
import argparse
from pathlib import Path
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from src.model import UNet


def predict(model, path, size, device):
    image = Image.open(path).convert("RGB")
    original = image.size
    x = TF.to_tensor(TF.resize(image, [size, size], antialias=True)).unsqueeze(0).to(device)
    with torch.no_grad(): mask = model(x).softmax(1)[:, 1].squeeze().cpu()
    mask = Image.fromarray((mask.numpy() > 0.5).astype("uint8") * 255).resize(original, Image.Resampling.NEAREST)
    return mask


def main(args):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=UNet(num_classes=2).to(device)
    ckpt=torch.load(args.model,map_location=device,weights_only=False); model.load_state_dict(ckpt.get("model_state",ckpt)); model.eval()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    for name in args.input:
        mask=predict(model,name,args.size,device); dest=out/(Path(name).stem+"_mask.png"); mask.save(dest); print(dest)

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--model",default="checkpoints/best.pt"); p.add_argument("--input",nargs="+",required=True); p.add_argument("--output",default="results/predictions"); p.add_argument("--size",type=int,default=256); main(p.parse_args())
