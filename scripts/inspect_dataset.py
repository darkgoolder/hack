from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def as_int(value: str) -> int | None:
    if value == "" or value.lower() == "none":
        return None
    return int(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print dataset statistics from a generated manifest")
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()

    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    print(f"Images: {len(rows)}")
    print(f"Folders: {len({row['folder_id'] for row in rows})}")
    print("Anatomy:")
    for anatomy, count in Counter(row["anatomy"] for row in rows).items():
        print(f"  {anatomy}: {count}")

    targets = [
        ("spine_layout", "Spine layout"),
        ("spine_axis", "Spine axis"),
        ("spine_artifact", "Spine artifact"),
        ("hip_position_rotation", "Hip position/rotation"),
        ("hip_roi", "Hip ROI"),
    ]
    print("Targets:")
    for column, label in targets:
        counts = Counter(as_int(row[column]) for row in rows if row[column] != "")
        print(f"  {label}: 0={counts.get(0, 0)} 1={counts.get(1, 0)} blank={sum(1 for row in rows if row[column] == '')}")

    print("Quality class:")
    qc = Counter(as_int(row["quality_class"]) for row in rows if row["quality_class"] != "")
    print(f"  0={qc.get(0, 0)} 1={qc.get(1, 0)}")

    comment_counts = Counter(row["expert_comment"] for row in rows if row["expert_comment"])
    print("Comments:")
    if comment_counts:
        for comment, count in comment_counts.most_common():
            print(f"  {count:>3} × {comment}")
    else:
        print("  none")

    tag_counts = Counter()
    for row in rows:
        for tag in filter(None, row["auxiliary_tags"].split(";")):
            tag_counts[tag] += 1
    print("Auxiliary tags:")
    if tag_counts:
        for tag, count in tag_counts.most_common():
            print(f"  {count:>3} × {tag}")
    else:
        print("  none")

    unreadable = [row for row in rows if row["dicom_readable"].lower() != "true"]
    print(f"Unreadable DICOM: {len(unreadable)}")


if __name__ == "__main__":
    main()
