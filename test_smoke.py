"""Fast architecture smoke test."""
import torch
from src.model import UNet

model = UNet(num_classes=2)
x = torch.randn(1, 3, 128, 128)
y = model(x)
assert y.shape == (1, 2, 128, 128), y.shape
print("Smoke test passed:", tuple(y.shape))
