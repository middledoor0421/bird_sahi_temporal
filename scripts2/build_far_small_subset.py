#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
from typing import Any, Dict, List, Set


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build far-small image_id subset from COCO-style GT.")
    parser.add_argument("--gt", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--category-id", type=int, required=True)
    parser.add_argument("--tau", type=float, required=True, help="bbox_area / image_area threshold")
    args = parser.parse_args()

    gt = load_json(args.gt)
    images = gt.get("images", [])
    anns = gt.get("annotations", [])
    img_meta: Dict[int, Dict[str, int]] = {
        int(im["id"]): {
            "width": int(im.get("width", 0)),
            "height": int(im.get("height", 0)),
        }
        for im in images
    }

    image_ids: Set[int] = set()
    ann_count = 0
    for ann in anns:
        if int(ann.get("category_id", -1)) != int(args.category_id):
            continue
        bbox = ann.get("bbox", None)
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        img_id = int(ann["image_id"])
        meta = img_meta.get(img_id, None)
        if meta is None:
            continue
        img_area = max(1, int(meta["width"]) * int(meta["height"]))
        box_area = max(0.0, float(bbox[2])) * max(0.0, float(bbox[3]))
        if (box_area / float(img_area)) <= float(args.tau):
            image_ids.add(img_id)
            ann_count += 1

    save_json(
        args.out,
        {
            "tau": float(args.tau),
            "category_id": int(args.category_id),
            "image_ids": sorted(int(x) for x in image_ids),
            "num_images": int(len(image_ids)),
            "num_annotations": int(ann_count),
        },
    )


if __name__ == "__main__":
    main()
