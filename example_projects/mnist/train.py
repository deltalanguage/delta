#!/usr/bin/env python3
"""MNIST with Delta - Minimal Example"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import delta

# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Lambda(lambda x: x.view(-1))  # Flatten
])

train = DataLoader(
    datasets.MNIST("./data", train=True, download=True, transform=transform),
    batch_size=64, shuffle=True
)

test = DataLoader(
    datasets.MNIST("./data", train=False, transform=transform),
    batch_size=64
)

# ─────────────────────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────────────────────

print("═" * 50)
print("  MNIST with Delta")
print("═" * 50)

model = delta.compile("model.delta")
model.to("auto")
model.fit(train, epochs=10, lr=0.001)

# ─────────────────────────────────────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────────────────────────────────────

import torch
model.eval()
correct = 0
for x, y in test:
    x, y = x.to(model._device), y.to(model._device)
    with torch.no_grad():
        preds = model(x).argmax(1)
        correct += (preds == y).sum().item()

print(f"\nTest accuracy: {100 * correct / len(test.dataset):.2f}%")
