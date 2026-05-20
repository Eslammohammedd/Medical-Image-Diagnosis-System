#!/usr/bin/env python
"""
scripts/export_onnx.py
───────────────────────
Exports a trained PyTorch checkpoint to ONNX format for edge deployment
(e.g., NVIDIA TensorRT, OpenVINO, ONNX Runtime).

Usage
-----
    python scripts/export_onnx.py \\
        --weights models/weights/best_model.pth \\
        --output  models/weights/chestxray_ai.onnx \\
        --classes 5
"""
import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from models.architectures.densenet121 import get_model
from utils.logger import get_logger

log = get_logger(__name__)


def export(weights_path: str, output_path: str, num_classes: int = 5, img_size: int = 224):
    device = torch.device("cpu")  # ONNX export always on CPU

    model = get_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    log.info(f"Model loaded from {weights_path}")

    dummy_input = torch.randn(1, 3, img_size, img_size)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    size_mb = Path(output_path).stat().st_size / 1e6
    log.info(f"✅ ONNX model saved → {output_path}  ({size_mb:.1f} MB)")
    print(f"✅ Export complete: {output_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--weights", default="models/weights/best_model.pth")
    parser.add_argument("--output",  default="models/weights/chestxray_ai.onnx")
    parser.add_argument("--classes", type=int, default=5)
    parser.add_argument("--size",    type=int, default=224)
    args = parser.parse_args()

    if not Path(args.weights).exists():
        print(f"❌ Weights not found: {args.weights}")
        sys.exit(1)

    export(args.weights, args.output, args.classes, args.size)
