from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("split_dir", type=Path)
    args = parser.parse_args()

    sets: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        path = args.split_dir / f"{split}.csv"
        df = pd.read_csv(path)
        sets[split] = set(df["folder_id"].astype(str))
        print(f"{split}: {len(df)} images, {len(sets[split])} folders")

    intersections = {
        "train_val": sets["train"] & sets["val"],
        "train_test": sets["train"] & sets["test"],
        "val_test": sets["val"] & sets["test"],
    }
    for name, values in intersections.items():
        print(f"{name} overlap: {len(values)}")
        if values:
            raise SystemExit(f"LEAKAGE DETECTED: {sorted(values)[:10]}")

    print("OK: no folder appears in more than one split.")


if __name__ == "__main__":
    main()
