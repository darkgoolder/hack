from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required. Install dependencies from requirements.txt") from exc


class CommentParser:
    def __init__(self, config_path: Path) -> None:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.rules: list[tuple[str, list[str]]] = [
            (str(item["pattern"]).lower(), [str(tag) for tag in item.get("tags", [])])
            for item in config.get("comments", [])
        ]
        self.ambiguous = {str(x).lower() for x in config.get("ambiguous_comment_tags", [])}

    @staticmethod
    def normalize(text: str | None) -> str:
        if not text:
            return ""
        text = text.replace("ё", "е").strip().lower()
        text = re.sub(r"\s+", " ", text)
        return text

    def parse(self, text: str | None) -> list[str]:
        normalized = self.normalize(text)
        if not normalized:
            return []
        tags: list[str] = []
        for pattern, rule_tags in self.rules:
            if pattern in normalized:
                for tag in rule_tags:
                    if tag not in tags:
                        tags.append(tag)
        return tags

    def is_ambiguous(self, text: str | None) -> bool:
        normalized = self.normalize(text)
        return any(item in normalized for item in self.ambiguous)

COMMENT_TAGS = {
    "сколиоз": {"SPINE_SCOLIOSIS"},
    "перелом": {"FRACTURE"},
    "правое бедро эндопротезирования": {"RIGHT_HIP_PROSTHESIS", "HIP_PROSTHESIS"},
    "эндопротезирования ТБС": {"HIP_PROSTHESIS"},
    "хороший пример отклонения оси при КТ п-ка": {"SPINE_AXIS_DEVIATION"},
    "l6, люмбализация": {"SPINE_L6_LUMBALIZATION"},
    "отклонение оси, нет малых вертелов": {
        "SPINE_AXIS_DEVIATION",
        "HIP_LESSER_TROCHANTER_NOT_VISIBLE",
    },
    "хороший пример ротации": {"HIP_ROTATION"},
}


def normalize_comment(comment: object) -> str:
    if comment is None:
        return ""
    return str(comment).strip().lower()


def parse_comment(comment: object) -> tuple[set[str], bool]:
    """Return auxiliary tags and whether the comment is ambiguous."""
    normalized = normalize_comment(comment)
    if not normalized:
        return set(), False

    tags = set()
    for phrase, phrase_tags in COMMENT_TAGS.items():
        if phrase in normalized:
            tags.update(phrase_tags)

    ambiguous = "требует внимание" in normalized
    return tags, ambiguous