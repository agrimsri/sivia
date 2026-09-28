"""Pydantic model and validators for SIVIA Kit Specification."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class KitClass(BaseModel):
    id: int = Field(ge=0, description="Zero-indexed class identifier")
    name: str = Field(min_length=1, description="Unique class name")
    prompts: list[str] = Field(min_length=1, description="List of zero-shot text prompts")

    @field_validator("prompts")
    @classmethod
    def validate_prompts(cls, v: list[str]) -> list[str]:
        cleaned = [p.strip() for p in v if p.strip()]
        if not cleaned:
            raise ValueError("Class must have at least one non-empty prompt")
        return cleaned


class KitRules(BaseModel):
    min_score: float = Field(default=0.35, ge=0.0, le=1.0)
    max_count: dict[str, int] = Field(default_factory=dict)

    @field_validator("max_count")
    @classmethod
    def validate_max_count(cls, v: dict[str, int]) -> dict[str, int]:
        for k, count in v.items():
            if count <= 0:
                raise ValueError(f"Max count for class '{k}' must be > 0, got {count}")
        return v


SlotBox = Annotated[list[float], Field(min_length=4, max_length=4)]


class KitSpec(BaseModel):
    kit_name: str = Field(min_length=1)
    classes: list[KitClass] = Field(min_length=1)
    required: list[str] = Field(default_factory=list)
    optional: list[str] = Field(default_factory=list)
    slots: dict[str, SlotBox] = Field(default_factory=dict)
    rules: KitRules = Field(default_factory=KitRules)

    @field_validator("slots")
    @classmethod
    def validate_slots(cls, slots: dict[str, list[float]]) -> dict[str, list[float]]:
        for slot_name, bbox in slots.items():
            if len(bbox) != 4:
                raise ValueError(f"Slot '{slot_name}' bbox must have 4 elements [x1, y1, x2, y2]")
            x1, y1, x2, y2 = bbox
            if not (0.0 <= x1 < x2 <= 1.0):
                raise ValueError(
                    f"Slot '{slot_name}' x coordinates invalid: 0 <= {x1} < {x2} <= 1.0 required"
                )
            if not (0.0 <= y1 < y2 <= 1.0):
                raise ValueError(
                    f"Slot '{slot_name}' y coordinates invalid: 0 <= {y1} < {y2} <= 1.0 required"
                )
        return slots

    @model_validator(mode="after")
    def validate_kit_consistency(self) -> KitSpec:
        class_ids = [c.id for c in self.classes]
        if len(class_ids) != len(set(class_ids)):
            raise ValueError(f"Duplicate class IDs found: {class_ids}")

        class_names = {c.name for c in self.classes}
        if len(class_names) != len(self.classes):
            raise ValueError("Duplicate class names found in classes list")

        # Validate required and optional lists
        for req in self.required:
            if req not in class_names:
                raise ValueError(f"Required class '{req}' is not defined in classes list")
        for opt in self.optional:
            if opt not in class_names:
                raise ValueError(f"Optional class '{opt}' is not defined in classes list")

        intersection = set(self.required) & set(self.optional)
        if intersection:
            raise ValueError(f"Classes cannot be both required and optional: {intersection}")

        # Validate slot class names
        for slot_class in self.slots:
            if slot_class not in class_names:
                raise ValueError(f"Slot defined for unknown class: '{slot_class}'")

        # Validate max_count class names
        for count_class in self.rules.max_count:
            if count_class not in class_names:
                raise ValueError(f"Max count defined for unknown class: '{count_class}'")

        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> KitSpec:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Kit spec file not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def get_class_id_by_name(self, name: str) -> int:
        for c in self.classes:
            if c.name == name:
                return c.id
        raise KeyError(f"Class '{name}' not found in kit '{self.kit_name}'")

    def get_class_name_by_id(self, class_id: int) -> str:
        for c in self.classes:
            if c.id == class_id:
                return c.name
        raise KeyError(f"Class id {class_id} not found in kit '{self.kit_name}'")
