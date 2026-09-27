from __future__ import annotations

import argparse
from pathlib import Path

from app.data.dicom_dataset import DXAQualityDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--root", type=Path, default=Path("data/raw/dataset"))
    args = parser.parse_args()

    ds = DXAQualityDataset(
        args.manifest, split=args.split,
        image_size=args.image_size, training=False,
        root=args.root,
    )

    n = min(args.count, len(ds))
    print(f"Dataset split={args.split}, samples={len(ds)}, smoke_test_count={n}")

    for i in range(n):
        sample = ds[i]
        print(
            f"[{i}] anatomy={sample['anatomy'].item()} "
            f"shape={tuple(sample['image'].shape)} "
            f"targets={sample['targets'].tolist()} "
            f"mask={sample['target_mask'].tolist()} "
            f"file={sample['image_path']}"
        )

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
