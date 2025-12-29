import argparse
import json
import os
from pathlib import Path


COCO_ID_BY_NAME = {
    "person": 1,
    "car": 3,
    "motorcycle": 4,
}

NAME_SYNONYMS = {
    "motorbike": "motorcycle",
}


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp_path, path)


def _canonical_name(name: str) -> str:
    return NAME_SYNONYMS.get(name, name)


def _build_id_map(categories: list[dict]) -> dict[int, int]:
    id_map: dict[int, int] = {}
    for cat in categories:
        src_id = int(cat["id"])
        src_name = str(cat["name"])
        canonical = _canonical_name(src_name)
        if canonical not in COCO_ID_BY_NAME:
            raise ValueError(
                f"Unsupported category name {src_name!r} (canonical={canonical!r}); "
                f"supported={sorted(COCO_ID_BY_NAME.keys())}"
            )
        id_map[src_id] = COCO_ID_BY_NAME[canonical]
    return id_map


def remap_coco_json(src_json: Path, dst_json: Path) -> None:
    data = _load_json(src_json)
    id_map = _build_id_map(data.get("categories", []))

    for ann in data.get("annotations", []):
        src_cat_id = int(ann["category_id"])
        if src_cat_id not in id_map:
            raise ValueError(f"Annotation category_id={src_cat_id} not found in categories for {src_json}")
        ann["category_id"] = id_map[src_cat_id]

    # Use COCO ids + canonical names.
    used_dst_ids = sorted(set(id_map.values()))
    data["categories"] = [
        {"id": coco_id, "name": name}
        for name, coco_id in COCO_ID_BY_NAME.items()
        if coco_id in used_dst_ids
    ]
    data["categories"].sort(key=lambda c: c["id"])

    _write_json(dst_json, data)


def ensure_symlink(link_path: Path, target_path: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        return
    link_path.parent.mkdir(parents=True, exist_ok=True)
    link_path.symlink_to(target_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a DETR-friendly COCO view using COCO category ids.")
    parser.add_argument("--src", required=True, help="Source COCO root (expects train2017/ val2017/ annotations/)")
    parser.add_argument("--dst", required=True, help="Destination COCO root to create")
    args = parser.parse_args()

    src_root = Path(args.src).resolve()
    dst_root = Path(args.dst).resolve()

    src_ann_dir = src_root / "annotations"
    dst_ann_dir = dst_root / "annotations"

    src_train_json = src_ann_dir / "instances_train2017.json"
    src_val_json = src_ann_dir / "instances_val2017.json"
    if not src_train_json.exists() or not src_val_json.exists():
        raise FileNotFoundError(
            "Missing COCO annotations under src; expected "
            f"{src_train_json} and {src_val_json}"
        )

    dst_train_json = dst_ann_dir / "instances_train2017.json"
    dst_val_json = dst_ann_dir / "instances_val2017.json"

    # Reuse the existing image trees via symlinks (avoid copying 10k images).
    ensure_symlink(dst_root / "train2017", src_root / "train2017")
    ensure_symlink(dst_root / "val2017", src_root / "val2017")

    # Remap jsons (idempotent).
    remap_coco_json(src_train_json, dst_train_json)
    remap_coco_json(src_val_json, dst_val_json)

    print(f"Prepared DETR COCO root: {dst_root}")


if __name__ == "__main__":
    main()

