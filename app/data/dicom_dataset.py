from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from app.preprocessing.dicom import dicom_to_tensor

QUALITY_TARGETS = (
    "spine_layout",
    "spine_axis",
    "spine_artifact",
    "hip_position_rotation",
    "hip_roi",
)


def build_train_transform():
    # No spatial transforms in the baseline: rotation/orientation and image
    # margins carry clinical signal and are themselves targets.
    return v2.Compose([
        v2.RandomApply(
            [v2.ColorJitter(brightness=0.08, contrast=0.08)],
            p=0.5,
        )
    ])


def build_eval_transform():
    return v2.Identity()


class DXAQualityDataset(Dataset):
    def __init__(
        self,
        manifest_csv: str | Path,
        split: str | None = None,
        image_size: int = 512,
        training: bool = False,
        root: str | Path | None = None,
    ):
        self.manifest_csv = Path(manifest_csv)
        self.df = pd.read_csv(self.manifest_csv)
        self.root = Path(root) if root is not None else None

        if split is not None:
            if "split" not in self.df.columns:
                raise ValueError("Manifest does not contain 'split' column.")
            self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        required = {
            "folder_id", "image_path", "anatomical_region",
            "quality_class", *QUALITY_TARGETS,
        }
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

        self.image_size = int(image_size)
        self.transform = build_train_transform() if training else build_eval_transform()
        self.anatomy_to_index = {"spine": 0, "left_hip": 1, "right_hip": 2}

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _masked_binary(value: Any):
        if pd.isna(value) or value == "":
            return 0.0, 0.0
        value = float(value)
        if value not in (0.0, 1.0):
            raise ValueError(f"Binary target must be 0/1/blank, got {value!r}")
        return value, 1.0

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_path = Path(row["image_path"])
        if self.root is not None and not image_path.is_absolute():
            image_path = self.root / image_path
        image, dicom_meta = dicom_to_tensor(image_path, self.image_size)
        image = self.transform(image)

        anatomy = str(row["anatomical_region"]).strip().lower()
        if anatomy not in self.anatomy_to_index:
            raise ValueError(f"Unknown anatomical_region={anatomy!r}")

        targets, masks = [], []
        for name in QUALITY_TARGETS:
            target, mask = self._masked_binary(row[name])
            targets.append(target)
            masks.append(mask)

        overall, overall_mask = self._masked_binary(row["quality_class"])
        targets.append(overall)
        masks.append(overall_mask)

        return {
            "image": image,
            "anatomy": torch.tensor(self.anatomy_to_index[anatomy], dtype=torch.long),
            "targets": torch.tensor(targets, dtype=torch.float32),
            "target_mask": torch.tensor(masks, dtype=torch.float32),
            "folder_id": str(row["folder_id"]),
            "image_path": str(image_path),
            "study_uid": str(row.get("study_uid", "")),
            "sop_instance_uid": str(row.get("sop_instance_uid", "")),
            "dicom_metadata": {
                "manufacturer": dicom_meta.manufacturer,
                "manufacturer_model": dicom_meta.manufacturer_model,
                "pixel_spacing": dicom_meta.pixel_spacing,
                "was_multiframe": dicom_meta.was_multiframe,
            },
        }
