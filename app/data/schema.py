from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Final

@dataclass
class ImageRecord:
    """One DICOM file + expert labels + DICOM metadata.

    Mirrors the schema expected by ``split_dataset.py``/``schema.py``.
    """
    folder_id: str
    image_path: Path
    file_name: str
    anatomy: str
    laterality: str | None = None
    projection: str | None = None
    study_uid: str | None = None
    series_uid: str | None = None
    sop_instance_uid: str | None = None
    spine_layout: int | None = None
    spine_axis: int | None = None
    spine_artifact: int | None = None
    hip_position_rotation: int | None = None
    hip_roi: int | None = None
    quality_class: int | None = None
    violation_type: str | None = None
    expert_comment: str | None = None
    auxiliary_tags: list[str] | None = None
    dicom_readable: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        # pathlib.Path -> POSIX-строка, чтобы CSV был кроссплатформенным
        data["image_path"] = str(data["image_path"]).replace("\\", "/")
        # список тегов -> строка через ';' (удобно для CSV)
        if isinstance(data.get("auxiliary_tags"), list):
            data["auxiliary_tags"] = ";".join(data["auxiliary_tags"])
        return data

# Canonical manifest columns produced/consumed by the project.
CORE_COLUMNS: Final[tuple[str, ...]] = (
    "folder_id",
    "image_path",
    "anatomical_region",
    "laterality",
)

LABEL_COLUMNS: Final[tuple[str, ...]] = (
    "spine_layout",
    "spine_axis",
    "spine_artifact",
    "hip_position_rotation",
    "hip_roi",
)

OPTIONAL_COLUMNS: Final[tuple[str, ...]] = (
    "study_uid",
    "series_uid",
    "sop_instance_uid",
    "projection",
    "quality_class",
    "study_comments",
    "auxiliary_tags",
)

ANATOMY_VALUES: Final[set[str]] = {"spine", "left_hip", "right_hip"}


@dataclass(frozen=True)
class SplitRecord:
    folder_id: str
    split: str


@dataclass(frozen=True)
class AnatomyLabels:
    spine: str = "spine"
    left_hip: str = "left_hip"
    right_hip: str = "right_hip"