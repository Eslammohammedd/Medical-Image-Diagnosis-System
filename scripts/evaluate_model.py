#!/usr/bin/env python
"""
scripts/evaluate_model.py
──────────────────────────
Evaluates a trained DenseNet121 checkpoint on the test split and prints
a full classification report + per-class AUC.

Usage
-----
    python scripts/evaluate_model.py \\
        --weights models/weights/best_model.pth \\
        --data    Classification-5 \\
        --batch   16
"""
import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# Add project root to path so imports work regardless of CWD
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from models.architectures.densenet121 import get_model
from training.transforms import get_transforms
from training.metrics import MetricTracker, get_classification_report
from utils.gpu_utils import get_device
from utils.logger import get_logger

log = get_logger(__name__)


def evaluate(weights_path: str, data_root: str, batch_size: int = 16):
    device = get_device()
    data_path = Path(data_root)

    # ── Load test data ────────────────────────────────────────────────────────
    test_dir = data_path / "test"
    if not test_dir.exists():
        test_dir = data_path / "valid"
        log.warning("No 'test' split found, falling back to 'valid'.")

    transform   = get_transforms(is_train=False)
    test_ds     = datasets.ImageFolder(str(test_dir), transform=transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = test_ds.classes
    log.info(f"Classes: {classes}")
    log.info(f"Test samples: {len(test_ds)}")

    # ── Load model ────────────────────────────────────────────────────────────
    model = get_model(num_classes=len(classes), pretrained=False)
    if Path(weights_path).exists():
        model.load_state_dict(torch.load(weights_path, map_location=device))
        log.info(f"Loaded weights: {weights_path}")
    else:
        log.error(f"Weights not found: {weights_path}")
        sys.exit(1)

    model.to(device)
    model.eval()

    # ── Run evaluation ────────────────────────────────────────────────────────
    tracker = MetricTracker(class_names=classes)
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            tracker.update(outputs, labels)

    metrics = tracker.compute()
    print("\n" + "=" * 50)
    print("📊 Evaluation Results")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:<15}: {v:.4f}")
    print("=" * 50)

    # Full classification report
    all_outputs = torch.cat(tracker.all_outputs)
    all_targets = torch.cat(tracker.all_targets)
    report = get_classification_report(all_outputs, all_targets, classes)
    print("\n📋 Per-Class Report:\n")
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ChestX-Ray AI model")
    parser.add_argument("--weights", default="models/weights/best_model.pth")
    parser.add_argument("--data",    default="Classification-5")
    parser.add_argument("--batch",   type=int, default=16)
    args = parser.parse_args()

    evaluate(args.weights, args.data, args.batch)
