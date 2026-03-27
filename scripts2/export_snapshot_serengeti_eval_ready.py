#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import zipfile
from collections import defaultdict
from typing import Any, Dict, List


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def load_single_json_from_zip(path: str) -> Dict[str, Any]:
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".json")]
        if len(names) != 1:
            raise ValueError("Expected exactly one json in zip: {}".format(path))
        return json.loads(z.read(names[0]))


def save_json(path: str, obj: Any) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Snapshot Serengeti split into workspace COCO+sequence format.")
    parser.add_argument("--images-json-zip", required=True)
    parser.add_argument("--splits-json", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val"])
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--annotation-prefix", default="fbdsv24_vid")
    args = parser.parse_args()

    images_meta = load_single_json_from_zip(args.images_json_zip)
    splits_meta = load_json(args.splits_json)
    split_locations = {str(x) for x in splits_meta["splits"][args.split]}

    all_images: List[Dict[str, Any]] = list(images_meta.get("images", []))
    all_annotations: List[Dict[str, Any]] = list(images_meta.get("annotations", []))
    categories: List[Dict[str, Any]] = list(images_meta.get("categories", []))

    split_images_raw = [im for im in all_images if str(im.get("location")) in split_locations]
    split_images_raw.sort(key=lambda im: (str(im.get("seq_id", "")), int(im.get("frame_num", 0)), str(im.get("id", ""))))

    orig_to_new_img_id: Dict[str, int] = {}
    split_images: List[Dict[str, Any]] = []
    seq_to_frames: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    seq_to_video_id: Dict[str, str] = {}

    for idx, im in enumerate(split_images_raw, start=1):
        orig_id = str(im["id"])
        new_id = int(idx)
        orig_to_new_img_id[orig_id] = new_id
        rel_path = os.path.join(os.path.abspath(args.image_root), str(im["file_name"]))
        out_im = dict(im)
        out_im["id"] = new_id
        out_im["orig_image_id"] = orig_id
        out_im["abs_file_name"] = rel_path
        split_images.append(out_im)

        seq_id = str(im.get("seq_id", ""))
        seq_to_video_id.setdefault(seq_id, seq_id)
        seq_to_frames[seq_id].append({
            "image_id": new_id,
            "rel_path": rel_path,
            "frame_num": int(im.get("frame_num", 0)),
            "orig_image_id": orig_id,
        })

    split_seq_ids = set(seq_to_frames.keys())
    split_annotations: List[Dict[str, Any]] = []
    for ann in all_annotations:
        if not isinstance(ann, dict):
            continue
        ann_cat = ann.get("category_id", None)
        seq_id = ann.get("seq_id", None)
        image_id = ann.get("image_id", None)

        keep = False
        out_ann = dict(ann)
        if image_id is not None and str(image_id) in orig_to_new_img_id:
            out_ann["image_id"] = int(orig_to_new_img_id[str(image_id)])
            keep = True
        elif seq_id is not None and str(seq_id) in split_seq_ids:
            keep = True

        if keep:
            if ann_cat is not None:
                out_ann["category_id"] = int(ann_cat)
            split_annotations.append(out_ann)

    sequences: List[Dict[str, Any]] = []
    for seq_id, frames in sorted(seq_to_frames.items(), key=lambda kv: kv[0]):
        frames.sort(key=lambda x: (int(x.get("frame_num", 0)), int(x["image_id"])))
        sequences.append({
            "video_id": seq_to_video_id[seq_id],
            "seq_id": seq_id,
            "frames": [
                {
                    "image_id": int(fr["image_id"]),
                    "rel_path": str(fr["rel_path"]),
                    "frame_num": int(fr.get("frame_num", 0)),
                    "orig_image_id": str(fr.get("orig_image_id", "")),
                }
                for fr in frames
            ],
        })

    ann_root = os.path.join(os.path.abspath(args.out_root), "annotations")
    ensure_dir(ann_root)
    gt_path = os.path.join(ann_root, "{}_{}.json".format(args.annotation_prefix, args.split))
    seq_path = os.path.join(ann_root, "{}_{}_sequences.json".format(args.annotation_prefix, args.split))
    summary_path = os.path.join(ann_root, "snapshot_export_{}_summary.json".format(args.split))

    gt = {
        "info": images_meta.get("info", {}),
        "licenses": images_meta.get("licenses", []),
        "images": split_images,
        "annotations": split_annotations,
        "categories": categories,
    }
    summary = {
        "split": str(args.split),
        "num_locations": len(split_locations),
        "num_images": len(split_images),
        "num_sequences": len(sequences),
        "num_annotations": len(split_annotations),
        "image_root": os.path.abspath(args.image_root),
        "gt_json": gt_path,
        "seq_json": seq_path,
    }

    save_json(gt_path, gt)
    save_json(seq_path, sequences)
    save_json(summary_path, summary)
    print("Saved:", gt_path)
    print("Saved:", seq_path)
    print("Saved:", summary_path)


if __name__ == "__main__":
    main()
