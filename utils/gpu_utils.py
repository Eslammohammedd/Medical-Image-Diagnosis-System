"""
utils/gpu_utils.py
──────────────────
GPU / device utilities for consistent hardware selection across training
and inference scripts.  Includes safety features for laptop GPUs.
"""
import torch
import platform
import subprocess
from utils.logger import get_logger

log = get_logger(__name__)

# ── Safety Constants ──────────────────────────────────────────────────────────
MAX_GPU_TEMP_C       = 85     # Emergency stop threshold
WARN_GPU_TEMP_C      = 78     # Warning threshold
VRAM_FRACTION        = 0.80   # Use only 80% of VRAM (protects laptop GPU)


def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Returns the best available compute device.

    Priority: CUDA GPU → Apple MPS (M-series) → CPU
    """
    if prefer_gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            log.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
            return device
        # Apple Silicon
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            log.info("Using Apple MPS (Metal Performance Shaders).")
            return device

    device = torch.device("cpu")
    log.info("CUDA not available. Falling back to CPU.")
    return device


def apply_gpu_safety():
    """
    Applies laptop-safe GPU settings:
      1. Limits VRAM usage to 80%
      2. Enables cuDNN benchmark for consistent performance
    Call ONCE at the start of training.
    """
    if not torch.cuda.is_available():
        return

    # Limit VRAM to protect against OOM and thermal throttling
    torch.cuda.set_per_process_memory_fraction(VRAM_FRACTION)
    log.info(f"VRAM limited to {VRAM_FRACTION:.0%} of total")

    # cuDNN benchmark — finds the fastest conv algorithm for fixed input sizes
    torch.backends.cudnn.benchmark = True
    log.info("cuDNN benchmark mode enabled")


def get_gpu_temperature() -> float | None:
    """
    Reads GPU temperature via nvidia-smi.
    Returns temperature in °C, or None if unavailable.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return None


def check_gpu_temperature() -> str:
    """
    Checks GPU temperature and returns a status string.
    Returns 'OK', 'WARN', or 'CRITICAL'.
    """
    temp = get_gpu_temperature()
    if temp is None:
        return "UNKNOWN"
    if temp >= MAX_GPU_TEMP_C:
        log.warning(f"GPU CRITICAL: {temp}°C >= {MAX_GPU_TEMP_C}°C — STOP TRAINING!")
        return "CRITICAL"
    if temp >= WARN_GPU_TEMP_C:
        log.warning(f"GPU WARNING: {temp}°C — consider reducing batch size")
        return "WARN"
    log.debug(f"GPU temp: {temp}°C — OK")
    return "OK"


def get_device_info() -> dict:
    """
    Returns a dict with device information suitable for the
    health-check endpoint and the Streamlit sidebar.
    """
    info = {
        "platform": platform.system(),
        "python":   platform.python_version(),
        "torch":    torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": "CPU",
        "vram_gb": None,
        "gpu_temp_c": None,
    }

    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["vram_gb"] = round(props.total_memory / (1024 ** 3), 1)
        info["gpu_temp_c"] = get_gpu_temperature()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info["device_name"] = "Apple MPS"

    return info


def get_vram_usage() -> dict | None:
    """Returns current VRAM usage in MB, or None if no GPU."""
    if not torch.cuda.is_available():
        return None
    return {
        "allocated_mb": round(torch.cuda.memory_allocated(0) / 1e6, 1),
        "reserved_mb":  round(torch.cuda.memory_reserved(0) / 1e6, 1),
        "total_mb":     round(torch.cuda.get_device_properties(0).total_memory / 1e6, 1),
    }


def free_gpu_memory():
    """Clears the CUDA memory cache – useful between inference calls."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        log.debug("CUDA memory cache cleared.")


def set_seed(seed: int = 42):
    """Sets random seeds for reproducibility."""
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    log.info(f"Random seed set to {seed}.")
