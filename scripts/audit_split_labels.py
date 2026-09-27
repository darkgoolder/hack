from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

TARGETS = [
    "spine_layout", "spine_axis", "spine_artifact",
    "hip_position_rotation", "hip_roi", "quality_class",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("split_dir", type=Path)
    args = parser.parse_args()

    for split in ("train", "val", "test"):
        path = args.split_dir / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path)
        print(f"\n=== {split.upper()} ===")
        print(f"Images: {len(df)}")
        print(f"Folders: {df['folder_id'].nunique()}")

        for target in TARGETS:
            s = df[target]
            print(
                f"{target:24s}"
                f"0={int((s == 0).sum()):3d} "
                f"1={int((s == 1).sum()):3d} "
                f"blank={int(s.isna().sum()):3d}"
            )


if __name__ == "__main__":
    main()
