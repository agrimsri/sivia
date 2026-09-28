"""COCO-standard evaluation harness for object detection teachers and students."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class CocoEvaluator:
    """Evaluates detector predictions against ground truth COCO annotations."""

    def __init__(self, ground_truth_coco: str | Path | dict[str, Any]) -> None:
        if isinstance(ground_truth_coco, dict):
            # Write to in-memory/temp or pass via COCO object
            self.coco_gt = COCO()
            self.coco_gt.dataset = ground_truth_coco
            self.coco_gt.createIndex()
        else:
            self.gt_path = Path(ground_truth_coco)
            with contextlib.redirect_stdout(io.StringIO()):
                self.coco_gt = COCO(str(self.gt_path))

        # Build category map
        self.cats = self.coco_gt.loadCats(self.coco_gt.getCatIds())
        self.cat_id_to_name = {c["id"]: c["name"] for c in self.cats}

    def evaluate(
        self,
        predictions: list[dict[str, Any]],
        iou_type: str = "bbox",
    ) -> dict[str, Any]:
        """Run standard COCO evaluation on detection predictions.

        Args:
            predictions: List of dicts, each with keys:
                image_id: int
                category_id: int
                bbox: [x, y, w, h]
                score: float
            iou_type: "bbox" or "segm".

        Returns:
            Dictionary with mAP_50, mAP_50_95, per_class_ap, coverage, etc.
        """
        if not predictions:
            return {
                "mAP_50": 0.0,
                "mAP_50_95": 0.0,
                "coverage": 0.0,
                "per_class_ap_50": dict.fromkeys(self.cat_id_to_name.values(), 0.0),
                "num_predictions": 0,
            }

        # Suppress COCO stdout printing
        with contextlib.redirect_stdout(io.StringIO()):
            coco_dt = self.coco_gt.loadRes(predictions)
            coco_eval = COCOeval(self.coco_gt, coco_dt, iouType=iou_type)
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

        # Overall summary stats
        # stats[0]: AP @ IoU=0.50:0.95
        # stats[1]: AP @ IoU=0.50
        map_50_95 = float(coco_eval.stats[0]) if coco_eval.stats[0] >= 0 else 0.0
        map_50 = float(coco_eval.stats[1]) if coco_eval.stats[1] >= 0 else 0.0

        # Calculate per-class AP@0.5
        # coco_eval.eval['precision'] has shape (T, R, K, A, M):
        # T: IoU thresholds (idx 0 is 0.50)
        # R: Recall thresholds (101 points)
        # K: Categories
        # A: Areas (0: all)
        # M: Max detections (2: 100)
        per_class_ap: dict[str, float] = {}
        if hasattr(coco_eval, "eval") and coco_eval.eval is not None:
            prec = coco_eval.eval["precision"]
            cat_ids = coco_eval.params.catIds
            for k_idx, cat_id in enumerate(cat_ids):
                cat_name = self.cat_id_to_name.get(cat_id, f"class_{cat_id}")
                prec_at_50 = prec[0, :, k_idx, 0, 2]
                valid_p = prec_at_50[prec_at_50 > -1]
                class_ap = float(np.mean(valid_p)) if len(valid_p) > 0 else 0.0
                per_class_ap[cat_name] = round(class_ap, 4)
        else:
            for name in self.cat_id_to_name.values():
                per_class_ap[name] = round(map_50, 4)

        # Calculate coverage: fraction of GT images that received >= 1 detection
        gt_image_ids = set(self.coco_gt.getImgIds())
        dt_image_ids = {p["image_id"] for p in predictions}
        coverage = float(len(dt_image_ids.intersection(gt_image_ids)) / max(1, len(gt_image_ids)))

        return {
            "mAP_50": round(map_50, 4),
            "mAP_50_95": round(map_50_95, 4),
            "coverage": round(coverage, 4),
            "per_class_ap_50": per_class_ap,
            "num_predictions": len(predictions),
        }
