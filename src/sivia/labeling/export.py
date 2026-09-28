"""Dataset exporter for SIVIA producing canonical COCO JSON and derived YOLO txt datasets.

Includes versioned directory layout, manifest provenance tracking, and round-trip verification.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import pycocotools.coco as coco_utils
import yaml

from sivia.store.repository import Sample, SiviaStore


class DatasetExporter:
    """Exports SQLite samples and labels to versioned COCO and YOLO dataset structures."""

    def __init__(self, store: SiviaStore, output_base_dir: str | Path = "data/datasets") -> None:
        self.store = store
        self.output_base_dir = Path(output_base_dir)

    def export(
        self,
        version: str = "v1",
        label_source: str | None = None,
        class_names: dict[int, str] | None = None,
        copy_images: bool = True,
    ) -> Path:
        """Export dataset to data/datasets/{version}/.

        Args:
            version: Dataset version string (e.g. "v1", "v2").
            label_source: Filter labels by source (e.g. 'ensemble', 'human'). If None, takes all non-rejected.
            class_names: Mapping of class_id -> name.
            copy_images: Whether to copy or symlink image files into dataset images/ folder.

        Returns:
            Path to the exported dataset directory.
        """
        dest_dir = self.output_base_dir / version
        dest_dir.mkdir(parents=True, exist_ok=True)

        images_dir = dest_dir / "images"
        labels_dir = dest_dir / "labels"
        annotations_dir = dest_dir / "annotations"

        for split in ["train", "val", "test"]:
            (images_dir / split).mkdir(parents=True, exist_ok=True)
            (labels_dir / split).mkdir(parents=True, exist_ok=True)
        annotations_dir.mkdir(parents=True, exist_ok=True)

        if class_names is None:
            class_names = {
                0: "screwdriver",
                1: "tape_roll",
                2: "sensor_module",
                3: "usb_cable",
                4: "multimeter",
                5: "pliers",
            }

        categories = [
            {"id": cid, "name": name, "supercategory": "kit_object"}
            for cid, name in sorted(class_names.items())
        ]

        # Gather samples by split
        splits = ["train", "val", "test"]
        split_samples: dict[str, list[Sample]] = {
            s: self.store.list_samples(split=s)
            for s in splits  # type: ignore[arg-type]
        }

        # Track manifest stats
        manifest: dict[str, Any] = {
            "version": version,
            "created_at": datetime.now(UTC).isoformat(),
            "total_images": 0,
            "total_annotations": 0,
            "splits": {},
            "class_distribution": dict.fromkeys(class_names.values(), 0),
            "source_distribution": {},
            "hashes": {},
        }

        anno_global_id = 1

        for split in splits:
            samples = split_samples[split]
            coco_images: list[dict[str, Any]] = []
            coco_annotations: list[dict[str, Any]] = []
            split_anno_count = 0

            for sample in samples:
                assert sample.id is not None
                src_path = Path(sample.path)
                if not src_path.exists():
                    continue

                # Read image size
                img_bgr = cv2.imread(str(src_path))
                if img_bgr is None:
                    continue
                img_h, img_w = img_bgr.shape[:2]

                # Destination file names
                target_filename = f"{sample.id:06d}_{src_path.name}"
                dest_img_path = images_dir / split / target_filename

                if copy_images and not dest_img_path.exists():
                    shutil.copy2(src_path, dest_img_path)

                coco_images.append(
                    {
                        "id": sample.id,
                        "file_name": target_filename,
                        "width": img_w,
                        "height": img_h,
                    }
                )

                # Fetch labels for this sample
                labels = self.store.list_labels(sample_id=sample.id)
                # Filter by status and source
                valid_labels = [
                    lab
                    for lab in labels
                    if lab.status != "rejected"
                    and (label_source is None or lab.source == label_source)
                ]

                yolo_lines: list[str] = []

                for lbl in valid_labels:
                    # Bounding box xyxy in pixels
                    x1, y1, x2, y2 = lbl.bbox_xyxy
                    w_box = max(0.0, x2 - x1)
                    h_box = max(0.0, y2 - y1)
                    area = w_box * h_box

                    # COCO bbox: [x, y, width, height]
                    coco_bbox = [round(x1, 2), round(y1, 2), round(w_box, 2), round(h_box, 2)]

                    # Segmentation
                    seg = []
                    if lbl.mask_rle:
                        seg = lbl.mask_rle

                    coco_annotations.append(
                        {
                            "id": anno_global_id,
                            "image_id": sample.id,
                            "category_id": lbl.class_id,
                            "bbox": coco_bbox,
                            "area": round(area, 2),
                            "iscrowd": 0,
                            "score": lbl.score,
                            "agreement_iou": lbl.agreement_iou,
                            "segmentation": seg,
                        }
                    )
                    anno_global_id += 1
                    split_anno_count += 1

                    # Update manifest stats
                    c_name = class_names.get(lbl.class_id, f"class_{lbl.class_id}")
                    manifest["class_distribution"][c_name] = (
                        manifest["class_distribution"].get(c_name, 0) + 1
                    )
                    manifest["source_distribution"][lbl.source] = (
                        manifest["source_distribution"].get(lbl.source, 0) + 1
                    )

                    # YOLO normalized format: <class_id> <x_center> <y_center> <width> <height>
                    xc_norm = min(1.0, max(0.0, ((x1 + x2) / 2.0) / img_w))
                    yc_norm = min(1.0, max(0.0, ((y1 + y2) / 2.0) / img_h))
                    w_norm = min(1.0, max(0.0, w_box / img_w))
                    h_norm = min(1.0, max(0.0, h_box / img_h))
                    yolo_lines.append(
                        f"{lbl.class_id} {xc_norm:.6f} {yc_norm:.6f} {w_norm:.6f} {h_norm:.6f}"
                    )

                # Write YOLO label file for this image
                label_txt_path = (labels_dir / split / target_filename).with_suffix(".txt")
                with open(label_txt_path, "w") as f:
                    f.write("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))

            # Save COCO instances JSON for this split
            coco_dict = {
                "info": {
                    "description": f"SIVIA Dataset {version}",
                    "version": version,
                    "date_created": datetime.now(UTC).isoformat(),
                },
                "licenses": [],
                "images": coco_images,
                "annotations": coco_annotations,
                "categories": categories,
            }

            coco_json_path = annotations_dir / f"instances_{split}.json"
            json_bytes = json.dumps(coco_dict, indent=2).encode("utf-8")
            with open(coco_json_path, "wb") as f:
                f.write(json_bytes)

            manifest["hashes"][f"instances_{split}.json"] = hashlib.sha256(json_bytes).hexdigest()
            manifest["splits"][split] = {
                "images": len(coco_images),
                "annotations": split_anno_count,
            }
            manifest["total_images"] += len(coco_images)
            manifest["total_annotations"] += split_anno_count

        # Write data.yaml for YOLO training
        yolo_yaml = {
            "path": str(dest_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": dict(sorted(class_names.items())),
        }
        with open(dest_dir / "data.yaml", "w") as f:
            yaml.safe_dump(yolo_yaml, f, sort_keys=False)

        # Write manifest.json
        with open(dest_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        return dest_dir

    @staticmethod
    def validate_export(dataset_dir: str | Path) -> bool:
        """Validate COCO and YOLO integrity of exported dataset."""
        d_path = Path(dataset_dir)
        manifest_file = d_path / "manifest.json"
        if not manifest_file.exists():
            return False

        annotations_dir = d_path / "annotations"
        labels_dir = d_path / "labels"

        for split in ["train", "val", "test"]:
            coco_file = annotations_dir / f"instances_{split}.json"
            if not coco_file.exists():
                return False

            # Verify pycocotools parses the JSON correctly
            try:
                coco = coco_utils.COCO(str(coco_file))
            except Exception:
                return False

            # Count YOLO label lines across the split
            split_label_dir = labels_dir / split
            yolo_box_count = 0
            for txt_file in split_label_dir.glob("*.txt"):
                with open(txt_file) as f:
                    lines = [ln.strip() for ln in f if ln.strip()]
                    yolo_box_count += len(lines)

            coco_box_count = len(coco.getAnnIds())
            if yolo_box_count != coco_box_count:
                return False

        return True
