from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch


def positive_weight(series: pd.Series) -> float:
    series = pd.to_numeric(series, errors="coerce").dropna()
    pos = int((series == 1).sum())
    neg = int((series == 0).sum())
    if pos == 0:
        return 1.0
    return max(neg / pos, 1.0)


def compute_weights(train_csv: str | Path) -> dict[str, torch.Tensor]:
    df = pd.read_csv(train_csv)

    spine = torch.tensor(
        [positive_weight(df["spine_layout"]),
         positive_weight(df["spine_axis"]),
         positive_weight(df["spine_artifact"])],
        dtype=torch.float32,
    )
    hip = torch.tensor(
        [positive_weight(df["hip_position_rotation"]),
         positive_weight(df["hip_roi"])],
        dtype=torch.float32,
    )
    quality = torch.tensor(
        [positive_weight(df["quality_class"])],
        dtype=torch.float32,
    )

    counts = {
        name: {
            "positive": int((pd.to_numeric(df[name], errors="coerce") == 1).sum()),
            "negative": int((pd.to_numeric(df[name], errors="coerce") == 0).sum()),
            "missing": int(pd.to_numeric(df[name], errors="coerce").isna().sum()),
        }
        for name in (
            "spine_layout",
            "spine_axis",
            "spine_artifact",
            "hip_position_rotation",
            "hip_roi",
            "quality_class",
        )
    }

    anatomy_counts = df["anatomical_region"].value_counts()
    anatomy_weight = torch.tensor(
        [
            len(df) / max(3 * int(anatomy_counts.get("spine", 0)), 1),
            len(df) / max(3 * int(anatomy_counts.get("left_hip", 0)), 1),
            len(df) / max(3 * int(anatomy_counts.get("right_hip", 0)), 1),
        ],
        dtype=torch.float32,
    )

    return {
        "spine_pos_weight": spine,
        "hip_pos_weight": hip,
        "quality_pos_weight": quality,
        "anatomy_class_weight": anatomy_weight,
        "counts": counts,
    }
