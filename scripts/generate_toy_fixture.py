"""Generate toy dataset fixture (10 images + 10 COCO boxes) for smoke tests."""

import json
from pathlib import Path

from PIL import Image, ImageDraw


def generate_toy_dataset(output_dir: Path | str = "tests/fixtures/toy") -> None:
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    categories = [
        {"id": 0, "name": "screwdriver", "supercategory": "tool"},
        {"id": 1, "name": "tape_roll", "supercategory": "supplies"},
        {"id": 2, "name": "sensor_module", "supercategory": "electronics"},
        {"id": 3, "name": "usb_cable", "supercategory": "electronics"},
        {"id": 4, "name": "multimeter", "supercategory": "tool"},
        {"id": 5, "name": "pliers", "supercategory": "tool"},
    ]

    images = []
    annotations = []

    width, height = 320, 320

    # 10 diverse toy images with drawn bounding boxes
    colors = [
        (220, 50, 50),
        (50, 180, 50),
        (50, 50, 220),
        (220, 180, 40),
        (180, 40, 220),
        (40, 200, 200),
        (200, 100, 50),
        (100, 100, 100),
        (150, 50, 100),
        (80, 150, 50),
    ]

    for i in range(10):
        img_id = i + 1
        file_name = f"toy_{img_id:03d}.jpg"
        img_path = images_dir / file_name

        # Create synthetic image with background noise/pattern
        bg_color = (230 + (i * 2) % 25, 230 - (i * 3) % 20, 230 + (i * 1) % 20)
        img = Image.new("RGB", (width, height), color=bg_color)
        draw = ImageDraw.Draw(img)

        # Place a colored rectangle
        cat_id = i % len(categories)
        box_w = 40 + (i * 5) % 50
        box_h = 30 + (i * 7) % 40
        x1 = 20 + (i * 25) % (width - box_w - 40)
        y1 = 20 + (i * 30) % (height - box_h - 40)
        x2 = x1 + box_w
        y2 = y1 + box_h

        draw.rectangle([x1, y1, x2, y2], fill=colors[i], outline=(0, 0, 0), width=2)
        img.save(img_path, format="JPEG", quality=95)

        images.append(
            {
                "id": img_id,
                "file_name": file_name,
                "width": width,
                "height": height,
            }
        )

        annotations.append(
            {
                "id": img_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [float(x1), float(y1), float(box_w), float(box_h)],  # COCO [x, y, w, h]
                "area": float(box_w * box_h),
                "iscrowd": 0,
            }
        )

    coco_data = {
        "info": {
            "description": "SIVIA Toy Fixture Dataset for Smoke Tests",
            "version": "1.0",
            "year": 2026,
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    json_path = output_path / "annotations.json"
    with open(json_path, "w") as f:
        json.dump(coco_data, f, indent=2)

    print(f"Generated {len(images)} toy images in {images_dir} and annotations at {json_path}")


if __name__ == "__main__":
    generate_toy_dataset()
