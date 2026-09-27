from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("split_dir", type=Path)
    args = parser.parse_args()

    sets = {}
    for split in ("train", "val", "test"):
        df = pd.read_csv(args.split_dir / f"{split}.csv")
        sets[split] = set(df["folder_id"].astype(str))
        print(f"{split}: {len(df)} images, {len(sets[split])} folders")

    ok = True
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sets[a] & sets[b]
        print(f"{a}_{b} overlap: {len(overlap)}")
        ok &= not overlap

    if not ok:
        raise SystemExit("ERROR: folder leakage detected.")
    print("OK: no folder appears in more than one split.")


if __name__ == "__main__":
    main()
