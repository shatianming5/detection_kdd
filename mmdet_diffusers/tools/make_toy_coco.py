#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _make_image(path: Path, *, width: int, height: int) -> None:
    try:
        from PIL import Image, ImageDraw
    except Exception as e:
        raise RuntimeError("Pillow is required to generate the toy dataset. Install `pillow`.") from e

    img = Image.new("RGB", (width, height), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)
    # Draw a simple box to make the image non-empty.
    draw.rectangle([width * 0.2, height * 0.2, width * 0.8, height * 0.8], outline=(220, 20, 60), width=3)
    img.save(path, format="JPEG", quality=90)


def _write_coco(path: Path, *, image_name: str, width: int, height: int, bbox_xywh: list[float]) -> None:
    x, y, w, h = bbox_xywh
    ann = {
        "id": 1,
        "image_id": 1,
        "category_id": 1,
        "bbox": [x, y, w, h],
        "area": float(w * h),
        "iscrowd": 0,
    }
    coco = {
        "images": [{"id": 1, "file_name": image_name, "width": int(width), "height": int(height)}],
        "annotations": [ann],
        "categories": [{"id": 1, "name": "person"}],
    }
    path.write_text(json.dumps(coco, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a minimal COCO-format toy dataset for MMDetection smoke tests.")
    ap.add_argument("--out", default="datasets/toy_coco", help="Output directory (default: datasets/toy_coco)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    out = Path(args.out)
    img_dir = out / "images"
    ann_dir = out / "annotations"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    image_name = "000000000001.jpg"
    image_path = img_dir / image_name
    _make_image(image_path, width=args.width, height=args.height)

    # One centered bbox.
    bbox = [args.width * 0.25, args.height * 0.25, args.width * 0.5, args.height * 0.5]
    _write_coco(ann_dir / "instances_train.json", image_name=image_name, width=args.width, height=args.height, bbox_xywh=bbox)
    _write_coco(ann_dir / "instances_val.json", image_name=image_name, width=args.width, height=args.height, bbox_xywh=bbox)

    print(f"Wrote toy COCO dataset to: {out.resolve()}")


if __name__ == "__main__":
    main()

