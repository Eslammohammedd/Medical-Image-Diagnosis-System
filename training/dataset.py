import torch
import os
from torchvision import datasets
from torch.utils.data import DataLoader
from training.transforms import get_transforms

def get_dataloaders(root_dir, batch_size=16, img_size=224, num_workers=2):
    """
    Creates dataloaders from an ImageFolder structure (like Roboflow downloads).
    Expected structure: root_dir/train/class_name/image.jpg

    Optimized for laptop GPU:
      - pin_memory=True  → faster CPU→GPU transfers
      - prefetch_factor=2 → pre-loads next batches without overloading RAM
      - num_workers=2     → safe for laptops (avoids CPU overload)
      - persistent_workers → keeps workers alive between epochs (faster)
    """
    train_dir = f"{root_dir}/train"
    val_dir = f"{root_dir}/valid" # Roboflow often uses 'valid'
    if not os.path.exists(val_dir):
        val_dir = f"{root_dir}/val" # Fallback to 'val'

    train_transform = get_transforms(img_size=img_size, is_train=True)
    val_transform = get_transforms(img_size=img_size, is_train=False)

    train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
    val_ds = datasets.ImageFolder(val_dir, transform=val_transform)

    use_cuda = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,         # Faster CPU→GPU transfer
        prefetch_factor=2,           # Pre-load 2 batches ahead
        persistent_workers=True if num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        prefetch_factor=2,
        persistent_workers=True if num_workers > 0 else False,
    )

    # Print dataset stats
    print(f"📊 Train: {len(train_ds)} images | Val: {len(val_ds)} images")
    print(f"📊 Batch size: {batch_size} | Workers: {num_workers} | Pin memory: {use_cuda}")

    return train_loader, val_loader, train_ds.classes
