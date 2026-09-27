from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ImageRecord:
    folder_id: str
    image_path: Path
    file_name: str
    anatomy: str
    laterality: str | None
    projection: str | None
    study_uid: str | None
    series_uid: str | None
    sop_instance_uid: str | None

    # Expert targets. None means "not applicable / image absent".
    spine_layout: int | None = None
    spine_axis: int | None = None
    spine_artifact: int | None = None
    hip_position_rotation: int | None = None
    hip_roi: int | None = None

    # Derived labels / comments.
    quality_class: int | None = None
    violation_type: str | None = None
    expert_comment: str | None = None
    auxiliary_tags: list[str] = field(default_factory=list)

    # Audit information.
    dicom_readable: bool = True
    error: str | None = None

    def applicable_targets(self) -> dict[str, int | None]:
        if self.anatomy == "spine":
            return {
                "spine_layout": self.spine_layout,
                "spine_axis": self.spine_axis,
                "spine_artifact": self.spine_artifact,
            }
        if self.anatomy in {"left_hip", "right_hip"}:
            return {
                "hip_position_rotation": self.hip_position_rotation,
                "hip_roi": self.hip_roi,
            }
        return {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "folder_id": self.folder_id,
            "image_path": str(self.image_path),
            "file_name": self.file_name,
            "anatomy": self.anatomy,
            "laterality": self.laterality,
            "projection": self.projection,
            "study_uid": self.study_uid,
            "series_uid": self.series_uid,
            "sop_instance_uid": self.sop_instance_uid,
            "spine_layout": self.spine_layout,
            "spine_axis": self.spine_axis,
            "spine_artifact": self.spine_artifact,
            "hip_position_rotation": self.hip_position_rotation,
            "hip_roi": self.hip_roi,
            "quality_class": self.quality_class,
            "violation_type": self.violation_type,
            "expert_comment": self.expert_comment,
            "auxiliary_tags": ";".join(self.auxiliary_tags),
            "dicom_readable": self.dicom_readable,
            "error": self.error,
        }
