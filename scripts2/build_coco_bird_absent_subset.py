#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
from typing import Any, Dict, List, Set


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MS COCO bird-absent subset manifests.")
    parser.add_argument("--coco-json", required=True, help="Path to COCO instances json.")
    parser.add_argument("--image-root", required=True, help="Path to split image root.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument("--target-class-name", default="bird", help="Class name to treat as excluded target.")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    coco = load_json(args.coco_json)
    categories: List[Dict[str, Any]] = list(coco.get("categories", []))
    images: List[Dict[str, Any]] = list(coco.get("images", []))
    annotations: List[Dict[str, Any]] = list(coco.get("annotations", []))

    cat_by_id = {int(c["id"]): c for c in categories}
    target_cats = [c for c in categories if str(c.get("name", "")).lower() == args.target_class_name.lower()]
    if not target_cats:
        raise ValueError("Target class not found: {}".format(args.target_class_name))
    if len(target_cats) != 1:
        raise ValueError("Expected exactly one target class, got {}".format(len(target_cats)))
    target_cat = target_cats[0]
    target_cat_id = int(target_cat["id"])

    image_ids_with_target: Set[int] = set()
    anns_kept_full: List[Dict[str, Any]] = []
    for ann in annotations:
        image_id = int(ann["image_id"])
        category_id = int(ann["category_id"])
        if category_id == target_cat_id:
            image_ids_with_target.add(image_id)
        else:
            anns_kept_full.append(ann)

    kept_images: List[Dict[str, Any]] = [im for im in images if int(im["id"]) not in image_ids_with_target]
    kept_image_ids: Set[int] = {int(im["id"]) for im in kept_images}
    anns_kept_full = [ann for ann in anns_kept_full if int(ann["image_id"]) in kept_image_ids]

    missing_files = []
    for im in kept_images:
        p = os.path.join(args.image_root, str(im["file_name"]))
        if not os.path.isfile(p):
            missing_files.append(str(im["file_name"]))

    subset_full = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": kept_images,
        "annotations": anns_kept_full,
        "categories": categories,
    }
    subset_bird_negative = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": kept_images,
        "annotations": [],
        "categories": [target_cat],
    }
    summary = {
        "source_json": args.coco_json,
        "image_root": args.image_root,
        "target_class_name": args.target_class_name,
        "target_category_id": target_cat_id,
        "num_images_total": len(images),
        "num_images_with_target": len(image_ids_with_target),
        "num_images_bird_absent": len(kept_images),
        "num_annotations_total": len(annotations),
        "num_annotations_non_target_on_absent_images": len(anns_kept_full),
        "num_missing_files": len(missing_files),
        "missing_files_preview": missing_files[:20],
        "rule": "Keep images that have zero annotations with category name equal to target_class_name.",
    }

    save_json(os.path.join(args.out_dir, "instances_bird_absent_full.json"), subset_full)
    save_json(os.path.join(args.out_dir, "instances_bird_absent_bird_only_negative.json"), subset_bird_negative)
    save_json(os.path.join(args.out_dir, "bird_absent_summary.json"), summary)
    with open(os.path.join(args.out_dir, "bird_absent_image_ids.txt"), "w") as f:
        for image_id in sorted(kept_image_ids):
            f.write(str(image_id) + "\n")

    print("Saved:", os.path.join(args.out_dir, "instances_bird_absent_full.json"))
    print("Saved:", os.path.join(args.out_dir, "instances_bird_absent_bird_only_negative.json"))
    print("Saved:", os.path.join(args.out_dir, "bird_absent_summary.json"))
    print("Saved:", os.path.join(args.out_dir, "bird_absent_image_ids.txt"))
    print("Bird-absent images:", len(kept_images))
    print("Images with target:", len(image_ids_with_target))


if __name__ == "__main__":
    main()
