import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image, ImageDraw
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

def get_data_transforms():
    """
    Returns torchvision transforms for training and validation/testing.
    Data augmentations are tailored for driver face analysis:
    - Subtle rotation (driver head tilt)
    - Color jitter (changing cabin lighting conditions)
    - Horizontal flip
    """
    train_transforms = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=12),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    ])

    eval_transforms = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    ])

    return train_transforms, eval_transforms


def get_dataloaders(train_dir=config.TRAIN_DIR, val_dir=config.VAL_DIR, batch_size=config.BATCH_SIZE):
    """
    Builds and returns PyTorch DataLoaders for train and validation datasets.
    """
    train_tf, eval_tf = get_data_transforms()

    train_dataset = datasets.ImageFolder(root=str(train_dir), transform=train_tf)
    val_dataset = datasets.ImageFolder(root=str(val_dir), transform=eval_tf)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # 0 is safe on Windows
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_loader, val_loader, train_dataset.classes


def check_dataset_status():
    """
    Scans dataset directories and returns count of images per class.
    """
    stats = {"train": {}, "val": {}, "test": {}}
    for split, path in [("train", config.TRAIN_DIR), ("val", config.VAL_DIR), ("test", config.TEST_DIR)]:
        for c in config.CLASSES:
            cls_folder = path / c
            if cls_folder.exists():
                count = len(list(cls_folder.glob("*.jpg"))) + len(list(cls_folder.glob("*.png")))
            else:
                count = 0
            stats[split][c] = count
    return stats


def generate_synthetic_starter_data(samples_per_class=15):
    """
    Generates synthetic starter images with simulated face/eye/mouth markers
    to test and verify the entire training and inference pipeline immediately
    before real webcam data is captured.
    """
    print("[Dataset Utils] Generating synthetic sample images for pipeline testing...")
    
    for split_dir, n_samples in [(config.TRAIN_DIR, samples_per_class), (config.VAL_DIR, max(4, samples_per_class // 3))]:
        for cls_name in config.CLASSES:
            folder = split_dir / cls_name
            folder.mkdir(parents=True, exist_ok=True)
            
            # Generate sample variations
            for i in range(n_samples):
                img = Image.new("RGB", config.IMAGE_SIZE, color=(180, 160, 140))
                draw = ImageDraw.Draw(img)

                # Simulated face oval
                draw.ellipse([30, 20, 194, 204], fill=(220, 190, 165), outline=(170, 130, 110), width=3)

                # Eyes
                if cls_name == "Sleepy":
                    # Closed eyes (horizontal slits)
                    draw.line([65, 85, 95, 85], fill=(50, 30, 20), width=4)
                    draw.line([130, 85, 160, 85], fill=(50, 30, 20), width=4)
                else:
                    # Open eyes
                    draw.ellipse([65, 75, 95, 95], fill=(255, 255, 255), outline=(50, 30, 20), width=2)
                    draw.ellipse([75, 80, 85, 90], fill=(40, 20, 10))
                    draw.ellipse([130, 75, 160, 95], fill=(255, 255, 255), outline=(50, 30, 20), width=2)
                    draw.ellipse([140, 80, 150, 90], fill=(40, 20, 10))

                # Mouth
                if cls_name == "Yawning":
                    # Wide open mouth
                    draw.ellipse([85, 130, 140, 185], fill=(40, 10, 10), outline=(120, 40, 40), width=3)
                    draw.ellipse([95, 145, 130, 175], fill=(20, 5, 5))
                else:
                    # Normal mouth
                    draw.arc([85, 135, 140, 160], start=0, end=180, fill=(120, 40, 40), width=3)

                # Add some random noise
                filename = f"sample_{cls_name.lower()}_{i:03d}.jpg"
                img.save(folder / filename)

    print("[Dataset Utils] Synthetic starter dataset successfully generated.")
