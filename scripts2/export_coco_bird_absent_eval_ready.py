#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
from typing import Any, Dict, List


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: str, data: Any) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def rel_symlink(src: str, dst: str) -> None:
    ensure_dir(os.path.dirname(dst))
    if os.path.lexists(dst):
        return
    rel_src = os.path.relpath(src, os.path.dirname(dst))
    os.symlink(rel_src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export COCO bird-absent subset into shared eval-ready format.")
    parser.add_argument("--coco-json", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--split", default="val2017")
    parser.add_argument("--annotation-prefix", default="fbdsv24_vid")
    args = parser.parse_args()

    coco = load_json(args.coco_json)
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])
    info = coco.get("info", {})
    licenses = coco.get("licenses", [])

    out_root = os.path.abspath(args.out_root)
    out_img_root = os.path.join(out_root, "images")
    out_ann_root = os.path.join(out_root, "annotations")
    ensure_dir(out_img_root)
    ensure_dir(out_ann_root)

    img_id_map: Dict[int, int] = {}
    exported_images: List[Dict[str, Any]] = []
    for idx, im in enumerate(images, start=1):
        old_id = int(im["id"])
        img_id_map[old_id] = idx
        src = os.path.join(os.path.abspath(args.image_root), str(im["file_name"]))
        dst_rel = str(im["file_name"])
        dst = os.path.join(out_img_root, dst_rel)
        if os.path.isfile(src):
            rel_symlink(src, dst)
        exp = dict(im)
        exp["id"] = idx
        exp["file_name"] = os.path.join("images", dst_rel)
        exported_images.append(exp)

    exported_annotations: List[Dict[str, Any]] = []
    for idx, ann in enumerate(annotations, start=1):
        old_img_id = int(ann.get("image_id"))
        if old_img_id not in img_id_map:
            continue
        exp = dict(ann)
        exp["id"] = idx
        exp["image_id"] = int(img_id_map[old_img_id])
        exported_annotations.append(exp)

    sequences: List[Dict[str, Any]] = []
    for seq_idx, im in enumerate(exported_images, start=1):
        image_id = int(im["id"])
        sequences.append({
            "id": int(seq_idx),
            "video_id": str(image_id),
            "seq_id": str(image_id),
            "frames": [{
                "image_id": int(image_id),
                "file_name": str(im["file_name"]),
                "rel_path": str(im["file_name"]),
                "frame_num": 0,
            }],
        })

    gt_out = os.path.join(out_ann_root, "{}_{}.json".format(args.annotation_prefix, args.split))
    seq_out = os.path.join(out_ann_root, "{}_{}_sequences.json".format(args.annotation_prefix, args.split))
    summary_out = os.path.join(out_ann_root, "coco_bird_absent_export_summary.json")

    save_json(gt_out, {
        "info": info,
        "licenses": licenses,
        "images": exported_images,
        "annotations": exported_annotations,
        "categories": categories,
    })
    save_json(seq_out, sequences)
    save_json(summary_out, {
        "split": str(args.split),
        "num_images": len(exported_images),
        "num_annotations": len(exported_annotations),
        "num_sequences": len(sequences),
    })

    print("Saved:", gt_out)
    print("Saved:", seq_out)
    print("Saved:", summary_out)


if __name__ == "__main__":
    main()
