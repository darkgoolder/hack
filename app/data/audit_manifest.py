from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

SPINE_TARGETS = ["spine_layout", "spine_axis", "spine_artifact"]
HIP_TARGETS = ["hip_position_rotation", "hip_roi"]


def derive_quality(row: pd.Series) -> float:
    if row["anatomical_region"] == "spine":
        values = [row[c] for c in SPINE_TARGETS if pd.notna(row[c])]
    elif row["anatomical_region"] in {"left_hip", "right_hip"}:
        values = [row[c] for c in HIP_TARGETS if pd.notna(row[c])]
    else:
        values = []
    if not values:
        return float("nan")
    return float(int(any(v == 1 for v in values)))


def build_report(df: pd.DataFrame) -> dict:
    work = df.copy()
    work["derived_quality"] = work.apply(derive_quality, axis=1)
    missing_quality = work[work["quality_class"].isna()].copy()
    mismatch = work[
        work["quality_class"].notna()
        & work["derived_quality"].notna()
        & (work["quality_class"] != work["derived_quality"])
    ].copy()
    return {
        "images": int(len(work)),
        "folders": int(work["folder_id"].nunique()),
        "unreadable_dicom": int((work["dicom_readable"] == False).sum()),
        "quality_class_missing": int(len(missing_quality)),
        "quality_class_vs_detail_mismatch": int(len(mismatch)),
        "projection_missing": int(work["projection"].isna().sum()),
        "anatomy_counts": {str(k): int(v) for k, v in work["anatomical_region"].value_counts(dropna=False).items()},
        "mismatch_rows": mismatch[[
            "folder_id", "file_name", "anatomical_region", "quality_class",
            "derived_quality", "study_comments", "violation_type"
        ]].fillna("").to_dict(orient="records"),
        "missing_quality_rows": missing_quality[[
            "folder_id", "file_name", "anatomical_region", "quality_class",
            "study_comments", "violation_type"
        ]].fillna("").to_dict(orient="records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/manifests/dataset_audit.json"))
    args = parser.parse_args()
    df = pd.read_csv(args.manifest)
    report = build_report(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Images: {report['images']}")
    print(f"Folders: {report['folders']}")
    print(f"Unreadable DICOM: {report['unreadable_dicom']}")
    print(f"Missing quality_class: {report['quality_class_missing']}")
    print(f"Detail-vs-total mismatches: {report['quality_class_vs_detail_mismatch']}")
    print(f"Missing projection metadata: {report['projection_missing']}")
    print(f"Audit written to: {args.output}")


if __name__ == "__main__":
    main()
