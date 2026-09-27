from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def save_heatmap_overlay(
    original: np.ndarray,
    grayscale_cam: np.ndarray,
    output_path: str | Path,
) -> None:
    """
    Save a simple Grad-CAM overlay.

    `original` and `grayscale_cam` are expected in HxW and [0,1].
    The exact blending/display policy can be refined after the first trained
    model and reviewed visually on DXA images.
    """
    original = np.asarray(original, dtype=np.float32)
    cam = np.asarray(grayscale_cam, dtype=np.float32)

    if original.ndim != 2:
        raise ValueError("original must be HxW grayscale.")
    if cam.ndim != 2:
        raise ValueError("grayscale_cam must be HxW.")

    h, w = original.shape
    img = Image.fromarray(np.clip(original * 255, 0, 255).astype(np.uint8))
    cam_img = Image.fromarray(np.clip(cam * 255, 0, 255).astype(np.uint8))
    cam_img = cam_img.resize((w, h), Image.Resampling.BILINEAR)

    # Keep the first implementation dependency-light. The external
    # pytorch-grad-cam package can later be used to compute grayscale_cam.
    cam_arr = np.asarray(cam_img, dtype=np.float32) / 255.0

    # Red-channel emphasis without relying on matplotlib.
    rgb = np.stack([original, original, original], axis=-1)
    rgb[..., 0] = np.maximum(rgb[..., 0], cam_arr)
    rgb[..., 1] *= (1.0 - 0.45 * cam_arr)
    rgb[..., 2] *= (1.0 - 0.45 * cam_arr)

    Image.fromarray(np.clip(rgb * 255, 0, 255).astype(np.uint8)).save(output_path)
