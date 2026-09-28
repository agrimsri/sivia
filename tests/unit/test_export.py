"""Unit tests for dataset export to COCO JSON and YOLO txt formats."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from sivia.labeling.export import DatasetExporter
from sivia.store.repository import Label, Sample, SiviaStore


def test_dataset_export_round_trip(tmp_path: Path) -> None:
    db_file = tmp_path / "test.db"
    store = SiviaStore(db_file)

    # Create dummy images
    img1_path = tmp_path / "img1.jpg"
    img2_path = tmp_path / "img2.jpg"
    cv2.imwrite(str(img1_path), np.zeros((100, 100, 3), dtype=np.uint8))
    cv2.imwrite(str(img2_path), np.zeros((100, 100, 3), dtype=np.uint8))

    s1 = Sample(
        id=None,
        path=str(img1_path),
        source_video="v1",
        frame_idx=0,
        ts=0.0,
        sha256="sha1",
        phash="p1",
        split="train",
        origin="capture",
    )
    s2 = Sample(
        id=None,
        path=str(img2_path),
        source_video="v2",
        frame_idx=0,
        ts=0.0,
        sha256="sha2",
        phash="p2",
        split="val",
        origin="capture",
    )
    id1 = store.add_sample(s1)
    id2 = store.add_sample(s2)

    l1 = Label(
        id=None,
        sample_id=id1,
        class_id=0,
        bbox_xyxy=[10.0, 20.0, 50.0, 60.0],
        source="ensemble",
        score=0.9,
        status="auto",
    )
    l2 = Label(
        id=None,
        sample_id=id2,
        class_id=1,
        bbox_xyxy=[20.0, 30.0, 70.0, 80.0],
        source="ensemble",
        score=0.85,
        status="auto",
    )
    store.add_label(l1)
    store.add_label(l2)

    exporter = DatasetExporter(store=store, output_base_dir=tmp_path / "datasets")
    out_dir = exporter.export(version="v_test", copy_images=True)

    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "data.yaml").exists()
    assert (out_dir / "annotations" / "instances_train.json").exists()
    assert (out_dir / "annotations" / "instances_val.json").exists()

    with open(out_dir / "manifest.json") as f:
        manifest = json.load(f)
    assert manifest["version"] == "v_test"
    assert manifest["total_images"] == 2
    assert manifest["total_annotations"] == 2

    # Verify validity with exporter's check
    assert exporter.validate_export(out_dir) is True
