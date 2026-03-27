#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def maybe_check_file(image_root: Optional[str], file_name: str) -> bool:
    if not image_root:
        return False
    return os.path.isfile(os.path.join(image_root, file_name))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Caltech Camera Traps split manifests from local metadata.")
    parser.add_argument("--images-json", required=True)
    parser.add_argument("--bboxes-json", required=True)
    parser.add_argument("--splits-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--image-root", default="", help="Optional local image root for existence check.")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    images_meta = load_json(args.images_json)
    bboxes_meta = load_json(args.bboxes_json)
    splits_meta = load_json(args.splits_json)
    image_root = args.image_root.strip() or None

    all_images: List[Dict[str, Any]] = list(images_meta.get("images", []))
    bbox_images: List[Dict[str, Any]] = list(bboxes_meta.get("images", []))
    bbox_anns: List[Dict[str, Any]] = list(bboxes_meta.get("annotations", []))
    categories: List[Dict[str, Any]] = list(bboxes_meta.get("categories", []))

    bbox_image_ids: Set[str] = {str(im["id"]) for im in bbox_images}
    anns_by_image: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ann in bbox_anns:
        anns_by_image[str(ann["image_id"])].append(ann)

    split_to_locations = {
        split_name: {str(x) for x in split_vals}
        for split_name, split_vals in splits_meta.get("splits", {}).items()
    }

    save_json(
        os.path.join(args.out_dir, "caltech_split_locations.json"),
        {
            "info": splits_meta.get("info", {}),
            "split_to_locations": {k: sorted(v) for k, v in split_to_locations.items()},
        },
    )

    top_summary: Dict[str, Any] = {
        "images_json": args.images_json,
        "bboxes_json": args.bboxes_json,
        "splits_json": args.splits_json,
        "image_root": image_root,
        "num_all_images_metadata": len(all_images),
        "num_bbox_images": len(bbox_images),
        "num_bbox_annotations": len(bbox_anns),
        "num_categories": len(categories),
        "splits": {},
    }

    for split_name, location_ids in split_to_locations.items():
        split_dir = os.path.join(args.out_dir, split_name)
        ensure_dir(split_dir)

        split_images = [im for im in all_images if str(im.get("location")) in location_ids]
        split_image_ids = {str(im["id"]) for im in split_images}
        split_bbox_images = [im for im in bbox_images if str(im["id"]) in split_image_ids]
        split_bbox_image_ids = {str(im["id"]) for im in split_bbox_images}
        split_bbox_anns = [ann for ann in bbox_anns if str(ann["image_id"]) in split_bbox_image_ids]

        missing_bbox_images = [im for im in split_images if str(im["id"]) not in split_bbox_image_ids]
        available_files = 0
        missing_files = 0
        if image_root:
            for im in split_images:
                if maybe_check_file(image_root, str(im["file_name"])):
                    available_files += 1
                else:
                    missing_files += 1

        split_manifest = {
            "info": images_meta.get("info", {}),
            "licenses": images_meta.get("licenses", []),
            "images": split_images,
            "annotations": split_bbox_anns,
            "categories": categories,
        }
        split_summary = {
            "split": split_name,
            "location_ids": sorted(location_ids),
            "num_locations": len(location_ids),
            "num_images_metadata": len(split_images),
            "num_bbox_images": len(split_bbox_images),
            "num_missing_bbox_images": len(missing_bbox_images),
            "num_bbox_annotations": len(split_bbox_anns),
            "num_categories": len(categories),
            "image_root_checked": image_root is not None,
            "num_available_files": available_files,
            "num_missing_files": missing_files,
            "missing_bbox_image_preview": [
                {
                    "id": im["id"],
                    "file_name": im["file_name"],
                    "location": im.get("location"),
                }
                for im in missing_bbox_images[:20]
            ],
        }

        save_json(os.path.join(split_dir, "manifest_coco_bbox.json"), split_manifest)
        save_json(os.path.join(split_dir, "summary.json"), split_summary)
        with open(os.path.join(split_dir, "image_ids.txt"), "w") as f:
            for image_id in sorted(split_image_ids):
                f.write(str(image_id) + "\n")

        top_summary["splits"][split_name] = split_summary

    save_json(os.path.join(args.out_dir, "prep_summary.json"), top_summary)
    print("Saved:", os.path.join(args.out_dir, "prep_summary.json"))
    for split_name in split_to_locations:
        print("Saved:", os.path.join(args.out_dir, split_name, "manifest_coco_bbox.json"))
        print("Saved:", os.path.join(args.out_dir, split_name, "summary.json"))
        print("Saved:", os.path.join(args.out_dir, split_name, "image_ids.txt"))


if __name__ == "__main__":
    main()
