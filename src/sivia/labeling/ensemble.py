"""Teacher ensemble fusion using Weighted Boxes Fusion (WBF) and SAM mask refinement.

Combines open-vocabulary detections from OWLv2 and Grounding DINO, computes teacher
agreement IoU, integrates SAM instance segmentation, and calibrates consensus scores.
"""

from __future__ import annotations

import numpy as np
from ensemble_boxes import weighted_boxes_fusion

from sivia.labeling.sam import SamPredictor, compute_box_iou
from sivia.labeling.types import EnsembleDetection, RawDetection


class TeacherEnsemble:
    """Multi-teacher fusion engine with WBF and consensus scoring."""

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        iou_threshold: float = 0.55,
        skip_box_threshold: float = 0.15,
        conf_threshold: float = 0.25,
        sam_refinement: bool = True,
    ) -> None:
        self.weights = weights or {"owlv2": 1.0, "grounding_dino": 1.0}
        self.iou_threshold = iou_threshold
        self.skip_box_threshold = skip_box_threshold
        self.conf_threshold = conf_threshold
        self.sam_refinement = sam_refinement

    def fuse(
        self,
        teacher_detections: dict[str, list[RawDetection]],
        image_width: int,
        image_height: int,
        sam_predictor: SamPredictor | None = None,
        image_array: np.ndarray | None = None,
        class_names: dict[int, str] | None = None,
    ) -> list[EnsembleDetection]:
        """Fuse multi-teacher detections for an image.

        Args:
            teacher_detections: Mapping of teacher_name -> list of RawDetection.
            image_width: Image width in pixels.
            image_height: Image height in pixels.
            sam_predictor: Optional SAM predictor for mask generation.
            image_array: Optional image RGB numpy array for SAM.
            class_names: Optional class_id -> name mapping.

        Returns:
            List of fused EnsembleDetection objects.
        """
        model_names = list(teacher_detections.keys())
        if not model_names:
            return []

        # Check if all models produced empty detections
        total_dets = sum(len(dets) for dets in teacher_detections.values())
        if total_dets == 0:
            return []

        # Prepare normalized boxes, scores, and labels for WBF
        boxes_list: list[list[list[float]]] = []
        scores_list: list[list[float]] = []
        labels_list: list[list[int]] = []
        model_weights: list[float] = []

        w = float(max(1, image_width))
        h = float(max(1, image_height))

        for name in model_names:
            dets = teacher_detections[name]
            b_model: list[list[float]] = []
            s_model: list[float] = []
            l_model: list[int] = []

            for d in dets:
                # Normalize [x1, y1, x2, y2] to [0.0, 1.0] and clip
                nx1 = max(0.0, min(1.0, d.box_xyxy[0] / w))
                ny1 = max(0.0, min(1.0, d.box_xyxy[1] / h))
                nx2 = max(0.0, min(1.0, d.box_xyxy[2] / w))
                ny2 = max(0.0, min(1.0, d.box_xyxy[3] / h))

                if nx2 > nx1 and ny2 > ny1:
                    b_model.append([nx1, ny1, nx2, ny2])
                    s_model.append(float(d.score))
                    l_model.append(int(d.class_id))

            boxes_list.append(b_model)
            scores_list.append(s_model)
            labels_list.append(l_model)
            model_weights.append(self.weights.get(name, 1.0))

        # Check if at least one model has valid boxes
        if not any(len(b) > 0 for b in boxes_list):
            return []

        # Run Weighted Boxes Fusion
        fused_boxes_norm, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights=model_weights,
            iou_thr=self.iou_threshold,
            skip_box_thr=self.skip_box_threshold,
        )

        fused_detections: list[EnsembleDetection] = []
        boxes_for_sam: list[list[float]] = []

        for f_box_norm, f_score, f_label in zip(
            fused_boxes_norm, fused_scores, fused_labels, strict=False
        ):
            if f_score < self.conf_threshold:
                continue

            # Un-normalize back to pixel coordinates
            px1 = float(f_box_norm[0] * w)
            py1 = float(f_box_norm[1] * h)
            px2 = float(f_box_norm[2] * w)
            py2 = float(f_box_norm[3] * h)
            box_pixel = [px1, py1, px2, py2]
            cid = int(f_label)

            # Determine contributing models and teacher agreement IoU
            contributing_models, agreement_iou = self._compute_agreement(
                target_box=box_pixel,
                class_id=cid,
                teacher_detections=teacher_detections,
            )

            # Calibrate ensemble confidence: reward consensus agreement
            calibrated_score = float(f_score)
            if agreement_iou > 0.0:
                # Slight consensus boost for high agreement
                calibrated_score = min(1.0, calibrated_score * (0.95 + 0.1 * agreement_iou))
            else:
                # Moderate penalty for single-teacher unconfirmed detections
                calibrated_score = calibrated_score * 0.90

            cname = class_names[cid] if class_names and cid in class_names else f"class_{cid}"

            fused_detections.append(
                EnsembleDetection(
                    class_id=cid,
                    class_name=cname,
                    box_xyxy=box_pixel,
                    score=float(calibrated_score),
                    agreement_iou=float(agreement_iou),
                    source_models=contributing_models,
                )
            )
            boxes_for_sam.append(box_pixel)

        # Optional SAM refinement for masks and mask_box_iou
        if self.sam_refinement and sam_predictor and boxes_for_sam and image_array is not None:
            mask_results = sam_predictor.predict_masks(image_array, boxes_for_sam)
            for det, m_res in zip(fused_detections, mask_results, strict=False):
                det.mask_rle = m_res.mask_rle
                det.mask_box = m_res.mask_box
                det.mask_box_iou = m_res.mask_box_iou

        return fused_detections

    def _compute_agreement(
        self,
        target_box: list[float],
        class_id: int,
        teacher_detections: dict[str, list[RawDetection]],
    ) -> tuple[list[str], float]:
        """Find matching teacher boxes and calculate agreement IoU."""
        matching_teachers: dict[str, list[float]] = {}

        for m_name, dets in teacher_detections.items():
            best_iou = 0.0
            best_box: list[float] | None = None
            for d in dets:
                if d.class_id == class_id:
                    iou = compute_box_iou(target_box, d.box_xyxy)
                    if iou > best_iou:
                        best_iou = iou
                        best_box = d.box_xyxy

            if best_box is not None and best_iou >= 0.30:
                matching_teachers[m_name] = best_box

        contributing = list(matching_teachers.keys())

        # If at least 2 distinct teachers matched, compute agreement IoU
        if len(matching_teachers) >= 2:
            model_keys = list(matching_teachers.keys())
            box_a = matching_teachers[model_keys[0]]
            box_b = matching_teachers[model_keys[1]]
            agreement_iou = compute_box_iou(box_a, box_b)
        else:
            agreement_iou = 0.0

        return contributing, agreement_iou
