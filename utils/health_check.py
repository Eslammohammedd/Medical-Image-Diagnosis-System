"""
utils/health_check.py
──────────────────────
System-readiness diagnostic.  Run from the project root:

    python utils/health_check.py

or call run_health_check() from within Streamlit for an in-app status page.
"""
import os
import sys
import shutil
import requests
import torch
from pathlib import Path
from dotenv import load_dotenv

# Fix Windows console encoding (cp1256 can't display emoji)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


def run_health_check() -> list[str]:
    """
    Checks GPU, API keys, model weights, dataset, and disk space.
    Returns a list of human-readable status strings.
    """
    load_dotenv()
    status = []

    print("🔍 Starting Medical AI Health Check…\n")

    # ── 1. GPU ────────────────────────────────────────────────────────────────
    gpu_ok = torch.cuda.is_available()
    status.append(
        f"🖥️  GPU: {'✅ ' + torch.cuda.get_device_name(0) if gpu_ok else '⚠️  CPU Mode (slower inference)'}"
    )

    # ── 2. Gemini API ─────────────────────────────────────────────────────────
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if len(gemini_key) > 20:
        try:
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models"
                f"?key={gemini_key}"
            )
            r = requests.get(url, timeout=8)
            if r.status_code == 200:
                status.append("📡 Gemini API: ✅ Connected")
            else:
                status.append(f"📡 Gemini API: ⚠️  HTTP {r.status_code}")
        except Exception as e:
            status.append(f"📡 Gemini API: ❌ Unreachable ({e})")
    else:
        status.append("📡 Gemini API: ❌ Key missing or too short")

    # ── 3. Roboflow API ───────────────────────────────────────────────────────
    rf_key = os.getenv("ROBOFLOW_API_KEY", "")
    if rf_key:
        status.append("🎯 Roboflow API: ✅ Key configured")
    else:
        status.append("🎯 Roboflow API: ❌ Key not set in .env")

    # ── 4. Model Weights ──────────────────────────────────────────────────────
    weight_candidates = [
        Path("models/weights/best_model.pth"),
        Path("models/weights/medical_xray_local_v1_real_weights.pth"),
    ]
    found_weight = next((p for p in weight_candidates if p.exists()), None)
    if found_weight:
        size_mb = found_weight.stat().st_size / 1e6
        status.append(f"🧠 Model Weights: ✅ {found_weight.name} ({size_mb:.1f} MB)")
    else:
        status.append(
            "🧠 Model Weights: ⚠️  Not found — run: "
            "python training/train.py --config configs/stage1_chexpert.yaml"
        )

    # ── 5. Dataset ────────────────────────────────────────────────────────────
    dataset_path = Path("Classification-5")
    if dataset_path.exists():
        splits = [s for s in ("train", "valid", "test") if (dataset_path / s).exists()]
        status.append(f"📂 Dataset: ✅ Found — splits: {', '.join(splits)}")
    else:
        status.append("📂 Dataset: ❌ Classification-5/ folder missing")

    # ── 6. Disk Space ─────────────────────────────────────────────────────────
    try:
        _, _, free = shutil.disk_usage(Path.cwd().anchor)
        free_gb = free / 1e9
        icon = "✅" if free_gb > 5 else "⚠️ "
        status.append(f"💾 Disk Space: {icon} {free_gb:.1f} GB free")
    except Exception:
        status.append("💾 Disk Space: ⚠️  Could not read")

    print("\n".join(status))
    print("\n✅ Health check complete.")
    return status


if __name__ == "__main__":
    run_health_check()
