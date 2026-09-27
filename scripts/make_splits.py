from __future__ import annotations

import argparse
from pathlib import Path

from app.data.split_dataset import make_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Create folder-safe train/val/test splits.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/manifests/splits"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    stats = make_splits(args.manifest, args.output_dir, seed=args.seed)
    print(f"Splits created in: {args.output_dir.resolve()}")
    for split, n in stats["folders"].items():
        print(f"  {split}: {n} folders / {stats['images'][split]} images")


if __name__ == "__main__":
    main()
