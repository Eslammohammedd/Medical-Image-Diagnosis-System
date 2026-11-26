import io

import numpy as np
from PIL import Image

from inference.preprocessor import (
    get_inference_transform,
    load_image,
    preprocess_for_model,
    resize_and_pad,
    tensor_to_numpy_image,
)


def _png_bytes(width=20, height=10):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (120, 80, 40)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_load_image_returns_rgb():
    image = load_image(_png_bytes())
    assert image.mode == "RGB"
    assert image.size == (20, 10)


def test_resize_and_pad_is_square():
    image = load_image(_png_bytes())
    padded = resize_and_pad(image, target_size=32)
    assert padded.size == (32, 32)


def test_inference_transform_shape():
    transform = get_inference_transform(32)
    image = load_image(_png_bytes())
    tensor = transform(image)
    assert tuple(tensor.shape) == (3, 32, 32)


def test_preprocess_for_model_adds_batch_dimension():
    tensor = preprocess_for_model(_png_bytes(), img_size=32)
    assert tuple(tensor.shape) == (1, 3, 32, 32)


def test_tensor_to_numpy_image_roundtrip():
    tensor = preprocess_for_model(_png_bytes(), img_size=32)[0]
    array = tensor_to_numpy_image(tensor)
    assert array.shape == (32, 32, 3)
    assert array.dtype == np.uint8
