"""
utils/config.py
──────────────
Centralised project configuration.  All hard-coded paths, hyper-parameters,
and feature flags live here so the rest of the codebase stays clean.
"""
import os
from pathlib import Path

# ── Root Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
DATA_ROOT      = PROJECT_ROOT / "Classification-5"
MODEL_WEIGHTS  = PROJECT_ROOT / "models" / "weights"
LOG_DIR        = PROJECT_ROOT / "logs"
ASSETS_DIR     = PROJECT_ROOT / "dashboard" / "assets"

# ── Dataset Settings ──────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Bacterial Pneumonia",
    "Covid-19",
    "Normal",
    "Pneumonia",
    "Viral Pneumonia",
]
NUM_CLASSES = len(CLASS_NAMES)

IMAGE_SIZE  = 224         # pixels (square)
BATCH_SIZE  = 16
NUM_WORKERS = 2           # safe default for Windows

# ── Model ─────────────────────────────────────────────────────────────────────
DEFAULT_ARCHITECTURE = "densenet121"
BEST_WEIGHTS_NAME    = "best_model.pth"
BEST_WEIGHTS_PATH    = MODEL_WEIGHTS / BEST_WEIGHTS_NAME

# ── Training ──────────────────────────────────────────────────────────────────
LEARNING_RATE  = 1e-4
WEIGHT_DECAY   = 1e-5
NUM_EPOCHS     = 15
LOSS_FN        = "CrossEntropyLoss"   # or 'FocalLoss'

# ── Inference ─────────────────────────────────────────────────────────────────
CONFIDENCE_HIGH   = 0.60   # red  (high risk)
CONFIDENCE_MEDIUM = 0.30   # amber (medium risk)

# ── API ───────────────────────────────────────────────────────────────────────
GEMINI_MODEL  = "gemini-2.5-flash"
GEMINI_TIMEOUT = 30       # seconds

ROBOFLOW_DETECTION_MODEL    = "train-3jix3/4"
ROBOFLOW_CLASSIFY_MODEL     = "classification-upura/5"
ROBOFLOW_TIMEOUT            = 20

# ── Ensure directories exist at import time ───────────────────────────────────
for _dir in (MODEL_WEIGHTS, LOG_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
