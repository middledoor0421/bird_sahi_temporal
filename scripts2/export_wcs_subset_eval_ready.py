#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: str, data: Any) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export WCS subset manifest into shared eval-ready COCO-video format.")
    parser.add_argument("--subset-root", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--annotation-prefix", default="fbdsv24_vid")
    args = parser.parse_args()

    subset_root = os.path.abspath(args.subset_root)
    src_manifest = os.path.join(subset_root, "annotations", "manifest_coco_bbox.json")
    if not os.path.isfile(src_manifest):
        raise FileNotFoundError("Subset manifest not found: {}".format(src_manifest))

    manifest = load_json(src_manifest)
    images = manifest.get("images", [])
    annotations = manifest.get("annotations", [])
    categories = manifest.get("categories", [])
    licenses = manifest.get("licenses", [])
    info = manifest.get("info", {})

    img_id_map: Dict[str, int] = {}
    exported_images: List[Dict[str, Any]] = []
    for idx, im in enumerate(images, start=1):
        old_id = str(im["id"])
        img_id_map[old_id] = int(idx)
        exported = dict(im)
        exported["id"] = int(idx)
        exported["file_name"] = os.path.join("images", str(im["file_name"]))
        exported_images.append(exported)

    exported_annotations: List[Dict[str, Any]] = []
    for idx, ann in enumerate(annotations, start=1):
        old_img_id = str(ann.get("image_id"))
        if old_img_id not in img_id_map:
            continue
        exported = dict(ann)
        exported["id"] = int(idx)
        exported["image_id"] = int(img_id_map[old_img_id])
        exported_annotations.append(exported)

    by_seq: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for im in exported_images:
        seq_id = str(im.get("seq_id", im.get("location", "unknown")))
        by_seq[seq_id].append(im)

    sequences: List[Dict[str, Any]] = []
    for seq_idx, (seq_id, seq_images) in enumerate(sorted(by_seq.items()), start=1):
        ordered = sorted(
            seq_images,
            key=lambda x: (
                int(x.get("frame_num", 0)),
                int(x.get("id", 0)),
            )
        )
        frames = []
        for im in ordered:
            frames.append({
                "image_id": int(im["id"]),
                "file_name": str(im["file_name"]),
                "rel_path": str(im["file_name"]),
                "frame_num": int(im.get("frame_num", 0)),
            })
        sequences.append({
            "id": int(seq_idx),
            "video_id": str(seq_id),
            "seq_id": str(seq_id),
            "location": str(ordered[0].get("location", "")) if ordered else "",
            "frames": frames,
        })

    out_ann_dir = os.path.join(subset_root, "annotations")
    ensure_dir(out_ann_dir)
    gt_out = os.path.join(out_ann_dir, "{}_{}.json".format(args.annotation_prefix, args.split))
    seq_out = os.path.join(out_ann_dir, "{}_{}_sequences.json".format(args.annotation_prefix, args.split))
    summary_out = os.path.join(out_ann_dir, "wcs_export_summary.json")

    save_json(gt_out, {
        "info": info,
        "licenses": licenses,
        "images": exported_images,
        "annotations": exported_annotations,
        "categories": categories,
    })
    save_json(seq_out, sequences)
    save_json(summary_out, {
        "subset_root": subset_root,
        "split": str(args.split),
        "num_images": len(exported_images),
        "num_annotations": len(exported_annotations),
        "num_sequences": len(sequences),
        "gt_json": gt_out,
        "seq_json": seq_out,
    })

    print("Saved:", gt_out)
    print("Saved:", seq_out)
    print("Saved:", summary_out)


if __name__ == "__main__":
    main()
