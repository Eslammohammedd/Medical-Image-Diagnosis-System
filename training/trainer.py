import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import os
import subprocess

class ModelTrainer:
    """
    Handles the training and validation loops for PyTorch models.
    Features:
      - Mixed Precision (FP16) for faster training and lower VRAM usage
      - GPU temperature monitoring to protect laptop hardware
      - Automatic emergency stop if GPU overheats (85°C+)
    """
    def __init__(self, model, optimizer, criterion, device, save_dir="models/weights",
                 use_amp=True, max_gpu_temp=85):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.save_dir = save_dir
        self.max_gpu_temp = max_gpu_temp

        # Mixed Precision setup — makes training ~2x faster on RTX GPUs
        self.use_amp = use_amp and device.type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None
        if self.use_amp:
            print("⚡ Mixed Precision (AMP) enabled — faster + lower VRAM usage")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def _get_gpu_temp(self) -> float | None:
        """Reads GPU temperature via nvidia-smi."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3,
            )
            if result.returncode == 0:
                return float(result.stdout.strip())
        except Exception:
            pass
        return None

    def _check_overheat(self) -> bool:
        """Returns True if GPU is overheating and training should stop."""
        temp = self._get_gpu_temp()
        if temp is not None and temp >= self.max_gpu_temp:
            print(f"\n🔴 GPU OVERHEATING: {temp}°C >= {self.max_gpu_temp}°C — STOPPING!")
            return True
        return False

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (images, labels) in enumerate(pbar):
            # Temperature check every 50 batches
            if batch_idx % 50 == 0 and self._check_overheat():
                print("Emergency stop: GPU temperature exceeded safe limit.")
                return total_loss / max(batch_idx, 1)

            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                # Mixed Precision forward pass
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                # Scaled backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard FP32 training
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()

            # Show loss + GPU temp in progress bar
            temp = self._get_gpu_temp()
            temp_str = f" | {temp:.0f}°C" if temp else ""
            pbar.set_postfix_str(f"loss={loss.item():.4f}{temp_str}")

        return total_loss / len(train_loader)

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                if self.use_amp:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                total_loss += loss.item()

        return total_loss / len(val_loader)

    def save_checkpoint(self, epoch, name="best_model.pth"):
        path = os.path.join(self.save_dir, name)
        torch.save(self.model.state_dict(), path)
        print(f"💾 Checkpoint saved to {path}")
