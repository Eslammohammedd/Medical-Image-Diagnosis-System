import io
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torch


# ImageNet statistics (used during DenseNet121 pre-training)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def load_image(source) -> Image.Image:
    """
    Accepts a file path (str), file-like object, or bytes and returns
    a PIL Image in RGB mode.
    """
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    img = Image.open(source)
    return img.convert('RGB')


def resize_and_pad(img: Image.Image, target_size: int = 224) -> Image.Image:
    """
    Resize the image so the longest side equals target_size,
    then pad to make it square (letter-boxing).  Avoids distortion
    that can fool the classifier.
    """
    w, h = img.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)

    padded = Image.new('RGB', (target_size, target_size), (0, 0, 0))
    offset_x = (target_size - new_w) // 2
    offset_y = (target_size - new_h) // 2
    padded.paste(img_resized, (offset_x, offset_y))
    return padded


def get_inference_transform(img_size: int = 224) -> T.Compose:
    """Standard transform pipeline for model inference (no augmentation)."""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def preprocess_for_model(source, img_size: int = 224) -> torch.Tensor:
    """
    End-to-end preprocessing: load → convert → transform → add batch dim.
    Returns a (1, 3, H, W) tensor ready for model.forward().
    """
    img = load_image(source)
    transform = get_inference_transform(img_size)
    tensor = transform(img)          # (3, H, W)
    return tensor.unsqueeze(0)       # (1, 3, H, W)


def tensor_to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Reverses normalization and converts a (3, H, W) tensor to a
    uint8 HxWx3 numpy array (for visualization / Grad-CAM overlay).
    """
    mean = np.array(IMAGENET_MEAN)
    std  = np.array(IMAGENET_STD)
    img  = tensor.detach().cpu().numpy().transpose(1, 2, 0)  # HxWxC
    img  = (img * std + mean).clip(0, 1)
    return (img * 255).astype(np.uint8)
