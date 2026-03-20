"""
src/dataset.py
==============
Shared dataloader module — extracted from src/preprocessing/preprocess.ipynb
Matches exactly the original preprocessing code, no modifications.
Import this in any branch that needs train_dl or valid_dl.
"""

import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

# ── Paths ───────────────────────────────────────────────
train_dir = "../data/train"
valid_dir = "../data/valid"

# ── Datasets ────────────────────────────────────────────
# Exactly as in preprocess.ipynb
train = ImageFolder(train_dir, transform=transforms.ToTensor())
valid = ImageFolder(valid_dir, transform=transforms.ToTensor())

# ── Config ──────────────────────────────────────────────
random_seed = 7
torch.manual_seed(random_seed)

batch_size = 32

# ── DataLoaders ─────────────────────────────────────────
# Exactly as in preprocess.ipynb
train_dl = DataLoader(train, batch_size, shuffle=True,
                      num_workers=2, pin_memory=True)

valid_dl = DataLoader(valid, batch_size,
                      num_workers=2, pin_memory=True)

# ── Quick info ──────────────────────────────────────────
num_classes = len(train.classes)  # 38
classes = train.classes

if __name__ == "__main__":
    print(f"Train size   : {len(train)}")
    print(f"Valid size   : {len(valid)}")
    print(f"Num classes  : {num_classes}")
    print(f"Batch size   : {batch_size}")

    img, label = train[0]
    print(f"Image shape  : {img.shape}")  # torch.Size([3, 256, 256])
    print(f"First class  : {classes[label]}")
