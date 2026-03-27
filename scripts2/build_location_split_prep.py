#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import zipfile
from typing import Any, Dict, List, Optional, Set


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def load_json_or_zipped_json(path: str) -> Dict[str, Any]:
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".json")]
            if not names:
                raise ValueError("No json file found in zip: {}".format(path))
            if len(names) != 1:
                raise ValueError("Expected exactly one json file in zip: {}".format(path))
            return json.loads(z.read(names[0]))
    return load_json(path)


def save_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def normalize_split_map(splits_json: Dict[str, Any], split_key: Optional[str]) -> Dict[str, List[str]]:
    if split_key:
        split_src = splits_json[split_key]
    else:
        split_src = splits_json

    out: Dict[str, List[str]] = {}
    for k, v in split_src.items():
        if k == "info" or k == "dataset":
            continue
        if not isinstance(v, list):
            continue
        out[str(k)] = [str(x) for x in v]
    return out


def maybe_check_file(image_root: Optional[str], file_name: str) -> bool:
    if not image_root:
        return False
    return os.path.isfile(os.path.join(image_root, file_name))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare split manifests grouped by image location.")
    parser.add_argument("--bbox-json", required=True)
    parser.add_argument("--splits-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split-key", default="", help="Optional top-level key like 'splits'.")
    parser.add_argument("--image-root", default="", help="Optional local image root for existence check.")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    bbox_meta = load_json_or_zipped_json(args.bbox_json)
    splits_meta = load_json(args.splits_json)
    image_root = args.image_root.strip() or None

    images: List[Dict[str, Any]] = list(bbox_meta.get("images", []))
    annotations: List[Dict[str, Any]] = list(bbox_meta.get("annotations", []))
    categories: List[Dict[str, Any]] = list(bbox_meta.get("categories", []))

    anns_by_image: Dict[str, List[Dict[str, Any]]] = {}
    for ann in annotations:
        image_id = str(ann["image_id"])
        anns_by_image.setdefault(image_id, []).append(ann)

    split_map = normalize_split_map(splits_meta, args.split_key.strip() or None)

    top_summary: Dict[str, Any] = {
        "bbox_json": args.bbox_json,
        "splits_json": args.splits_json,
        "image_root": image_root,
        "num_images": len(images),
        "num_annotations": len(annotations),
        "num_categories": len(categories),
        "splits": {},
    }

    save_json(os.path.join(args.out_dir, "split_locations.json"), {"splits": split_map})

    for split_name, location_ids in split_map.items():
        split_dir = os.path.join(args.out_dir, split_name)
        ensure_dir(split_dir)
        location_set: Set[str] = {str(x) for x in location_ids}
        split_images = [im for im in images if str(im.get("location")) in location_set]
        split_image_ids = {str(im["id"]) for im in split_images}
        split_anns = [ann for ann in annotations if str(ann["image_id"]) in split_image_ids]

        available_files = 0
        missing_files = 0
        if image_root:
            for im in split_images:
                if maybe_check_file(image_root, str(im["file_name"])):
                    available_files += 1
                else:
                    missing_files += 1

        manifest = {
            "info": bbox_meta.get("info", {}),
            "licenses": bbox_meta.get("licenses", []),
            "images": split_images,
            "annotations": split_anns,
            "categories": categories,
        }
        summary = {
            "split": split_name,
            "num_locations": len(location_set),
            "num_images": len(split_images),
            "num_annotations": len(split_anns),
            "num_categories": len(categories),
            "image_root_checked": image_root is not None,
            "num_available_files": available_files,
            "num_missing_files": missing_files,
            "location_ids_preview": sorted(list(location_set))[:20],
        }
        save_json(os.path.join(split_dir, "manifest_coco_bbox.json"), manifest)
        save_json(os.path.join(split_dir, "summary.json"), summary)
        with open(os.path.join(split_dir, "image_ids.txt"), "w") as f:
            for image_id in sorted(split_image_ids):
                f.write(str(image_id) + "\n")
        top_summary["splits"][split_name] = summary

    save_json(os.path.join(args.out_dir, "prep_summary.json"), top_summary)
    print("Saved:", os.path.join(args.out_dir, "prep_summary.json"))
    for split_name in split_map:
        print("Saved:", os.path.join(args.out_dir, split_name, "manifest_coco_bbox.json"))
        print("Saved:", os.path.join(args.out_dir, split_name, "summary.json"))
        print("Saved:", os.path.join(args.out_dir, split_name, "image_ids.txt"))


if __name__ == "__main__":
    main()
