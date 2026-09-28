"""Prompt configuration manager and prompt variation handler for open-vocabulary teachers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class PromptManager:
    """Manages prompt engineering variations and class label mappings."""

    def __init__(self, config_path: str | Path = "configs/labeling/prompts.yaml") -> None:
        self.config_path = Path(config_path)
        self.raw_config = self._load_config()
        self.active_set = self.raw_config.get("active_set", "default")
        self.classes: list[dict[str, Any]] = self.raw_config.get("classes", [])

    def _load_config(self) -> dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Prompt config not found at {self.config_path}")
        with open(self.config_path) as f:
            return yaml.safe_load(f) or {}

    def get_prompt_sets(self) -> list[str]:
        """Return list of available prompt variation sets."""
        if not self.classes:
            return []
        first_prompts = self.classes[0].get("prompts", {})
        return list(first_prompts.keys())

    def get_class_names(self) -> dict[int, str]:
        """Return mapping of class_id -> class_name."""
        return {item["id"]: item["name"] for item in self.classes}

    def get_prompts_for_set(self, prompt_set_name: str | None = None) -> dict[int, str]:
        """Return class_id -> prompt string for a given prompt set (e.g. 'default', 'descriptive')."""
        target_set = prompt_set_name or self.active_set
        mapping: dict[int, str] = {}
        for item in self.classes:
            cid = item["id"]
            prompts = item.get("prompts", {})
            if target_set in prompts:
                mapping[cid] = prompts[target_set]
            elif "default" in prompts:
                mapping[cid] = prompts["default"]
            else:
                mapping[cid] = item["name"]
        return mapping

    def get_all_prompts_for_class(self, class_id: int) -> dict[str, str]:
        """Return all prompt variations for a given class ID."""
        for item in self.classes:
            if item["id"] == class_id:
                return dict(item.get("prompts", {}))
        return {}

    def sweep_prompt_sets(
        self,
        eval_fn: Any,
        prompt_sets: list[str] | None = None,
    ) -> dict[str, Any]:
        """Evaluate multiple prompt variation sets using an evaluation function.

        Args:
            eval_fn: Callable taking class_prompts dict and returning evaluation metrics dict.
            prompt_sets: List of prompt set names to sweep (default: all available).

        Returns:
            Dictionary summarizing results per prompt set and identifying the optimal set.
        """
        targets = prompt_sets or self.get_prompt_sets()
        results: dict[str, Any] = {}
        best_set = ""
        best_score = -1.0

        for p_set in targets:
            prompts = self.get_prompts_for_set(p_set)
            metrics = eval_fn(prompts, p_set)
            results[p_set] = metrics

            score = metrics.get("mAP_50", 0.0)
            if score > best_score:
                best_score = score
                best_set = p_set

        return {
            "prompt_sweep_results": results,
            "best_prompt_set": best_set,
            "best_mAP_50": best_score,
        }
