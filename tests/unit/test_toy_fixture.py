"""Unit tests verifying the integrity of the toy smoke test fixture."""

import json
from pathlib import Path

from PIL import Image


def test_toy_fixture_structure():
    """Verify toy images and COCO annotations conform to expectations."""
    fixture_dir = Path("tests/fixtures/toy")
    images_dir = fixture_dir / "images"
    ann_file = fixture_dir / "annotations.json"

    assert images_dir.is_dir()
    assert ann_file.is_file()

    images = list(images_dir.glob("*.jpg"))
    assert len(images) == 10

    with open(ann_file) as f:
        coco = json.load(f)

    assert "images" in coco
    assert "annotations" in coco
    assert "categories" in coco

    assert len(coco["images"]) == 10
    assert len(coco["annotations"]) == 10
    assert len(coco["categories"]) == 6

    # Verify each image opens and dimensions match COCO metadata
    for img_meta in coco["images"]:
        img_path = images_dir / img_meta["file_name"]
        assert img_path.exists()
        with Image.open(img_path) as im:
            w, h = im.size
            assert w == img_meta["width"] == 320
            assert h == img_meta["height"] == 320

    # Verify bounding boxes
    for ann in coco["annotations"]:
        x, y, w, h = ann["bbox"]
        assert x >= 0 and y >= 0
        assert w > 0 and h > 0
        assert (x + w) <= 320
        assert (y + h) <= 320
