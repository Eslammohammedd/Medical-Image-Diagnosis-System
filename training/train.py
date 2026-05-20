import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim

# Fix Windows console encoding
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.architectures.densenet121 import get_model
from training.dataset import get_dataloaders
from training.trainer import ModelTrainer
from training.losses import get_loss_fn


def freeze_backbone(model, freeze=True):
    """
    Freezes/unfreezes DenseNet backbone layers.
    Freezing = only train the classifier head (Transfer Learning).
    Much faster and prevents overfitting on small datasets.
    """
    for param in model.backbone.parameters():
        param.requires_grad = not freeze
    status = "FROZEN" if freeze else "UNFROZEN"
    print(f"🧊 Backbone layers: {status}")


def main(config_path):
    # 1. Load Config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # 2. GPU Safety Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"🏥 ChestX-Ray AI — Training Pipeline")
    print(f"{'='*55}")
    print(f"⚙️  Device: {device}")

    if torch.cuda.is_available():
        # Limit VRAM to 80% — protects laptop GPU
        torch.cuda.set_per_process_memory_fraction(0.80)
        torch.backends.cudnn.benchmark = True
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🖥️  GPU: {gpu_name} ({vram:.1f} GB)")
        print(f"🛡️  VRAM limited to 80% | cuDNN benchmark ON")

    # 3. Setup Data
    data_root = "Classification-5"
    if not os.path.exists(data_root):
        print(f"❌ Dataset folder not found: {data_root}")
        return

    train_loader, val_loader, classes = get_dataloaders(
        data_root,
        batch_size=cfg['data']['batch_size'],
        img_size=cfg['data']['img_size'],
        num_workers=cfg['data']['num_workers']
    )

    num_classes = len(classes)
    print(f"🏷️  Classes ({num_classes}): {list(classes)}")

    # 4. Setup Model
    model = get_model(num_classes=num_classes, pretrained=cfg['model']['pretrained'])

    # Load checkpoint if resuming from a previous stage
    checkpoint_path = cfg.get('model', {}).get('checkpoint')
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"📦 Loaded checkpoint: {checkpoint_path}")

    # Freeze backbone for Transfer Learning (first stage only)
    unfreeze_all = cfg.get('training', {}).get('unfreeze_all', False)
    if not unfreeze_all:
        freeze_backbone(model, freeze=True)
    else:
        freeze_backbone(model, freeze=False)

    # 5. Setup Loss & Optimizer
    loss_name = cfg.get('training', {}).get('loss_fn', 'CrossEntropyLoss')
    criterion = get_loss_fn(loss_name)
    print(f"📉 Loss: {loss_name}")

    # Only optimize trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in trainable_params)
    print(f"🧠 Parameters: {trainable_count:,} trainable / {total_params:,} total")

    optimizer = optim.AdamW(
        trainable_params,
        lr=cfg['training']['lr'],
        weight_decay=cfg['training']['weight_decay']
    )

    # LR Scheduler (optional)
    scheduler = None
    scheduler_name = cfg.get('training', {}).get('scheduler')
    if scheduler_name == 'CosineAnnealingLR':
        t_max = cfg['training'].get('t_max', cfg['training']['epochs'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
        print(f"📈 Scheduler: CosineAnnealingLR (T_max={t_max})")

    # 6. Initialize Trainer (with Mixed Precision + overheat protection)
    trainer = ModelTrainer(model, optimizer, criterion, device, use_amp=True)

    # 7. Training Loop
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = cfg.get('early_stopping', {}).get('patience', 999)

    print(f"\n🚀 Starting: {cfg['experiment']['name']}")
    print(f"   Epochs: {cfg['training']['epochs']} | LR: {cfg['training']['lr']}")
    print(f"   Early stopping patience: {early_stop_patience}")
    print(f"{'='*55}\n")

    for epoch in range(1, cfg['training']['epochs'] + 1):
        print(f"--- Epoch {epoch}/{cfg['training']['epochs']} ---")
        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.validate(val_loader)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"📊 Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")

        # Step scheduler
        if scheduler:
            scheduler.step()

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            trainer.save_checkpoint(epoch, name=f"{cfg['experiment']['name']}_real_weights.pth")
            trainer.save_checkpoint(epoch, name="best_model.pth")
            print(f"🏆 New best val loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"⏳ No improvement ({patience_counter}/{early_stop_patience})")

        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\n🛑 Early stopping at epoch {epoch} (no improvement for {early_stop_patience} epochs)")
            break

        print()

    # 8. Final summary
    print(f"\n{'='*55}")
    print(f"✅ Training Complete!")
    print(f"   Best Val Loss: {best_val_loss:.4f}")
    print(f"   Weights saved: models/weights/best_model.pth")
    print(f"{'='*55}")

    # Free GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    main(args.config)
