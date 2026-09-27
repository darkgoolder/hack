from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import pydicom

from app.data.schema import LABEL_COLUMNS


class DXAImageDataset(Dataset):
    """PyTorch Dataset for one-DICOM-per-sample training.

    The Dataset does not infer anatomy from file names. It uses the validated manifest
    labels for supervision and reads only DICOM pixel data at training time.
    """

    def __init__(self, manifest_csv: str | Path, transform: Any | None = None):
        self.df = pd.read_csv(manifest_csv).reset_index(drop=True)
        self.transform = transform

        required = {"image_path", "anatomical_region", "laterality", "folder_id", *LABEL_COLUMNS}
        missing = sorted(required - set(self.df.columns))
        if missing:
            raise ValueError(f"Missing columns in manifest: {missing}")

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def _load_dicom(path: Path) -> np.ndarray:
        ds = pydicom.dcmread(str(path), force=False)
        if not hasattr(ds, "PixelData"):
            raise ValueError(f"DICOM has no PixelData: {path}")
        image = ds.pixel_array.astype(np.float32)
        if image.ndim != 2:
            raise ValueError(f"Expected a 2D DXA image, got shape={image.shape} for {path}")
        if not np.isfinite(image).all():
            raise ValueError(f"Non-finite pixel values in {path}")
        return image

    @staticmethod
    def _normalize(image: np.ndarray) -> np.ndarray:
        lo = np.percentile(image, 1.0)
        hi = np.percentile(image, 99.0)
        if hi <= lo:
            return np.zeros_like(image, dtype=np.float32)
        image = np.clip(image, lo, hi)
        return ((image - lo) / (hi - lo)).astype(np.float32)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        path = Path(str(row["image_path"]))
        image = self._normalize(self._load_dicom(path))

        # Grayscale is repeated to 3 channels to support standard pretrained backbones.
        tensor = torch.from_numpy(image).unsqueeze(0)
        tensor = tensor.repeat(3, 1, 1)

        if self.transform is not None:
            tensor = self.transform(tensor)

        targets: dict[str, torch.Tensor] = {}
        for column in LABEL_COLUMNS:
            value = row[column]
            # -1 marks a label that is not applicable to this anatomy.
            targets[column] = torch.tensor(-1.0 if pd.isna(value) else float(value), dtype=torch.float32)

        targets["quality_class"] = torch.tensor(
            float(self._derive_quality(row)), dtype=torch.float32
        )

        return {
            "image": tensor,
            "targets": targets,
            "folder_id": str(row["folder_id"]),
            "image_path": str(path),
            "anatomical_region": str(row["anatomical_region"]),
            "laterality": None if pd.isna(row["laterality"]) else str(row["laterality"]),
        }

    @staticmethod
    def _derive_quality(row: pd.Series) -> int:
        anatomy = str(row["anatomical_region"])
        if anatomy == "spine":
            values = [row[c] for c in ("spine_layout", "spine_axis", "spine_artifact")]
        elif anatomy in {"left_hip", "right_hip"}:
            values = [row[c] for c in ("hip_position_rotation", "hip_roi")]
        else:
            raise ValueError(f"Unsupported anatomy: {anatomy}")
        numeric = [int(v) for v in values if not pd.isna(v)]
        return int(any(v == 1 for v in numeric)) if numeric else 0
