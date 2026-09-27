from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom
import torch
from PIL import Image


@dataclass
class DICOMImage:
    pixels: np.ndarray
    study_uid: str | None
    series_uid: str | None
    sop_instance_uid: str | None
    manufacturer: str | None
    manufacturer_model: str | None
    photometric_interpretation: str | None
    pixel_spacing: tuple[float, float] | None
    was_multiframe: bool = False


def _safe_str(value):
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def load_dicom(path: str | Path) -> DICOMImage:
    path = Path(path)
    ds = pydicom.dcmread(path, force=False)
    arr = np.asarray(ds.pixel_array)
    was_multiframe = arr.ndim > 2

    if arr.ndim == 3:
        arr = arr[0]
    elif arr.ndim != 2:
        raise ValueError(f"Unsupported Pixel Array shape {arr.shape} in {path}")

    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.all():
        fill = float(np.median(arr[finite])) if finite.any() else 0.0
        arr[~finite] = fill

    photometric = _safe_str(getattr(ds, "PhotometricInterpretation", None))
    if photometric == "MONOCHROME1":
        pmin, pmax = float(arr.min()), float(arr.max())
        if pmax > pmin:
            arr = pmax - arr

    spacing = None
    raw = getattr(ds, "PixelSpacing", None)
    if raw is not None and len(raw) >= 2:
        try:
            spacing = (float(raw[0]), float(raw[1]))
        except (TypeError, ValueError):
            pass

    return DICOMImage(
        pixels=arr,
        study_uid=_safe_str(getattr(ds, "StudyInstanceUID", None)),
        series_uid=_safe_str(getattr(ds, "SeriesInstanceUID", None)),
        sop_instance_uid=_safe_str(getattr(ds, "SOPInstanceUID", None)),
        manufacturer=_safe_str(getattr(ds, "Manufacturer", None)),
        manufacturer_model=_safe_str(getattr(ds, "ManufacturerModelName", None)),
        photometric_interpretation=photometric,
        pixel_spacing=spacing,
        was_multiframe=was_multiframe,
    )


def robust_normalize(arr, low=1.0, high=99.0):
    arr = np.asarray(arr, dtype=np.float32)
    lo, hi = np.percentile(arr, [low, high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def resize_preserve_aspect(image, size=512, pad_value=0.0):
    image = np.asarray(image, dtype=np.float32)
    h, w = image.shape
    if h <= 0 or w <= 0:
        raise ValueError("Empty image.")

    scale = min(size / h, size / w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    pil = Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8))
    pil = pil.resize((new_w, new_h), Image.Resampling.BILINEAR)
    resized = np.asarray(pil, dtype=np.float32) / 255.0

    canvas = np.full((size, size), pad_value, dtype=np.float32)
    top = (size - new_h) // 2
    left = (size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


def dicom_to_tensor(path, image_size=512):
    dicom = load_dicom(path)
    image = robust_normalize(dicom.pixels)
    image = resize_preserve_aspect(image, size=image_size)
    tensor = torch.from_numpy(image).unsqueeze(0).repeat(3, 1, 1)
    return tensor.float(), dicom
