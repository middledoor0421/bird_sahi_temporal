#!/usr/bin/env python3
import argparse
import json
import os
from collections import defaultdict


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, ensure_ascii=False)


def _normalize_filename(name):
    # Normalize slashes for consistent keying
    return name.replace("\\", "/")


def merge_coco_by_filename(coco_a, coco_b):
    """
    Merge two COCO datasets by image file_name, reindexing image_id and annotation_id.
    This avoids id collisions between train/val.
    """
    images_a = coco_a.get("images", [])
    images_b = coco_b.get("images", [])
    anns_a = coco_a.get("annotations", [])
    anns_b = coco_b.get("annotations", [])

    # Use categories and other top-level fields from A if present, else from B
    merged = {}
    for k in ["info", "licenses", "categories"]:
        if k in coco_a:
            merged[k] = coco_a[k]
        elif k in coco_b:
            merged[k] = coco_b[k]

    # Build filename -> image dict
    fn_to_img = {}
    fn_sources = defaultdict(list)

    for img in images_a:
        fn = _normalize_filename(img["file_name"])
        fn_to_img[fn] = dict(img)
        fn_sources[fn].append("A")

    for img in images_b:
        fn = _normalize_filename(img["file_name"])
        if fn not in fn_to_img:
            fn_to_img[fn] = dict(img)
        fn_sources[fn].append("B")

    # Assign new contiguous image ids
    fns_sorted = sorted(fn_to_img.keys())
    new_images = []
    old_to_new = {}

    for new_id, fn in enumerate(fns_sorted, start=1):
        img = fn_to_img[fn]
        old_ids = []
        # store old ids for mapping (could be two datasets)
        # We map by file_name, so old_to_new is not used directly for images, only for annotations remap
        img["id"] = new_id
        img["file_name"] = fn
        new_images.append(img)
        old_to_new[fn] = new_id

    # Remap annotations by matching image_id -> file_name
    # Build image_id -> file_name for each source
    id_to_fn_a = {img["id"]: _normalize_filename(img["file_name"]) for img in images_a}
    id_to_fn_b = {img["id"]: _normalize_filename(img["file_name"]) for img in images_b}

    new_anns = []
    ann_id = 1

    def _remap_ann(ann, id_to_fn):
        nonlocal ann_id
        img_fn = id_to_fn.get(ann["image_id"], None)
        if img_fn is None:
            return
        if img_fn not in old_to_new:
            return
        out = dict(ann)
        out["id"] = ann_id
        ann_id += 1
        out["image_id"] = old_to_new[img_fn]
        new_anns.append(out)

    for ann in anns_a:
        _remap_ann(ann, id_to_fn_a)
    for ann in anns_b:
        _remap_ann(ann, id_to_fn_b)

    merged["images"] = new_images
    merged["annotations"] = new_anns
    return merged


def merge_sequences(seq_a, seq_b, val_suffix="_val"):
    """
    Merge sequences list. If video_id collides, append suffix to seq_b video_id.
    Each sequence element is expected to contain 'video_id' and either
    ('frame_ids' and optionally 'frame_files') or 'frames'.
    """
    out = []
    used = set()

    def _get_vid(x):
        return x.get("video_id", x.get("video_name", None))

    for x in seq_a:
        vid = _get_vid(x)
        if vid is None:
            continue
        used.add(vid)
        out.append(x)

    for x in seq_b:
        vid = _get_vid(x)
        if vid is None:
            continue
        if vid in used:
            x = dict(x)
            x["video_id"] = str(vid) + str(val_suffix)
            vid = x["video_id"]
        used.add(vid)
        out.append(x)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_ann", required=True)
    ap.add_argument("--val_ann", required=True)
    ap.add_argument("--train_seq", required=True)
    ap.add_argument("--val_seq", required=True)
    ap.add_argument("--out_ann", required=True)
    ap.add_argument("--out_seq", required=True)
    ap.add_argument("--val_suffix", default="_val")
    args = ap.parse_args()

    coco_train = _load_json(args.train_ann)
    coco_val = _load_json(args.val_ann)

    merged_coco = merge_coco_by_filename(coco_train, coco_val)
    _save_json(merged_coco, args.out_ann)

    seq_train = _load_json(args.train_seq)
    seq_val = _load_json(args.val_seq)
    merged_seq = merge_sequences(seq_train, seq_val, val_suffix=args.val_suffix)
    _save_json(merged_seq, args.out_seq)

    print("Wrote:", args.out_ann)
    print("  images:", len(merged_coco.get("images", [])))
    print("  annotations:", len(merged_coco.get("annotations", [])))
    print("Wrote:", args.out_seq)
    print("  videos:", len(merged_seq))


if __name__ == "__main__":
    main()
