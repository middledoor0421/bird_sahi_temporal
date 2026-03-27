#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import gzip
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from scripts2.gt_utils import resolve_target_category_id


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def iter_frames(video_entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    frames = video_entry.get("frames", [])
    if isinstance(frames, list) and len(frames) > 0:
        for idx, fr in enumerate(frames):
            if isinstance(fr, dict):
                out.append({"frame_idx": idx, "image_id": fr.get("image_id", None)})
            else:
                out.append({"frame_idx": idx, "image_id": fr})
        return out

    frame_ids = video_entry.get("frame_ids", [])
    if isinstance(frame_ids, list) and len(frame_ids) > 0:
        for idx, image_id in enumerate(frame_ids):
            out.append({"frame_idx": idx, "image_id": image_id})
    return out


def build_gt_index(gt_path: str, target_cid: int) -> Dict[int, List[Dict[str, Any]]]:
    gt = load_json(gt_path)
    anns_by_img: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for ann in gt.get("annotations", []):
        if not isinstance(ann, dict):
            continue
        try:
            cid = int(ann.get("category_id", -1))
            image_id = int(ann.get("image_id"))
        except Exception:
            continue
        if cid != int(target_cid):
            continue
        bbox = ann.get("bbox", None)
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        anns_by_img[image_id].append({
            "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
            "area": float(ann.get("area", 0.0)),
            "category_id": cid,
        })
    return anns_by_img


def build_nonempty_index_any_annotation(gt_path: str) -> Tuple[Set[int], Set[str]]:
    gt = load_json(gt_path)
    categories = gt.get("categories", [])
    empty_ids: Set[int] = set()
    for cat in categories:
        try:
            cid = int(cat.get("id"))
        except Exception:
            continue
        name = str(cat.get("name", "")).strip().lower()
        if name == "empty":
            empty_ids.add(cid)

    nonempty_image_ids: Set[int] = set()
    nonempty_seq_ids: Set[str] = set()
    for ann in gt.get("annotations", []):
        if not isinstance(ann, dict):
            continue
        try:
            cid = int(ann.get("category_id", -1))
        except Exception:
            cid = -1
        if cid in empty_ids:
            continue

        image_id = ann.get("image_id", None)
        if image_id is not None:
            try:
                nonempty_image_ids.add(int(image_id))
            except Exception:
                pass

        seq_id = ann.get("seq_id", None)
        if seq_id is not None:
            nonempty_seq_ids.add(str(seq_id))
    return nonempty_image_ids, nonempty_seq_ids


def default_paths(data_root: str, split: str) -> Dict[str, str]:
    ann_dir = os.path.join(os.path.abspath(data_root), "annotations")
    return {
        "gt": os.path.join(ann_dir, "fbdsv24_vid_{}.json".format(split)),
        "seq": os.path.join(ann_dir, "fbdsv24_vid_{}_sequences.json".format(split)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build strict-empty frame mask from COCO GT + sequence json."
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--gt-json", type=str, default=None)
    parser.add_argument("--seq-json", type=str, default=None)
    parser.add_argument("--target-category-id", type=int, default=None)
    parser.add_argument("--target-category-name", type=str, default=None)
    parser.add_argument("--nonempty-mode", type=str, default="target_bbox",
                        choices=["target_bbox", "any_annotation"])
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--out-name", type=str, default="empty_mask_strict.jsonl.gz")
    args = parser.parse_args()

    defaults = default_paths(args.data_root, args.split)
    gt_path = args.gt_json if args.gt_json else defaults["gt"]
    seq_path = args.seq_json if args.seq_json else defaults["seq"]

    if not os.path.isfile(gt_path):
        raise FileNotFoundError("GT json not found: {}".format(gt_path))
    if not os.path.isfile(seq_path):
        raise FileNotFoundError("Sequence json not found: {}".format(seq_path))

    ensure_dir(args.out_dir)

    seq_list = load_json(seq_path)
    if not isinstance(seq_list, list):
        raise ValueError("Sequences json must be a list: {}".format(seq_path))

    target_cid: Optional[int] = None
    gt_by_img: Dict[int, List[Dict[str, Any]]] = {}
    nonempty_image_ids: Set[int] = set()
    nonempty_seq_ids: Set[str] = set()
    if args.nonempty_mode == "target_bbox":
        target_cid = resolve_target_category_id(
            gt_path,
            target_category_id=args.target_category_id,
            target_category_name=args.target_category_name,
        )
        gt_by_img = build_gt_index(gt_path, int(target_cid))
    else:
        nonempty_image_ids, nonempty_seq_ids = build_nonempty_index_any_annotation(gt_path)

    out_path = os.path.join(args.out_dir, args.out_name)
    summary_path = os.path.join(args.out_dir, "empty_mask_summary.json")

    total_frames = 0
    total_empty = 0
    total_non_empty = 0
    per_video: List[Dict[str, Any]] = []

    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        for video_entry in seq_list:
            if not isinstance(video_entry, dict):
                continue
            video_id = str(video_entry.get("video_id", video_entry.get("id", "")))
            frames = iter_frames(video_entry)
            video_total = 0
            video_empty = 0

            for fr in frames:
                image_id_raw = fr.get("image_id", None)
                if image_id_raw is None:
                    continue
                try:
                    image_id = int(image_id_raw)
                except Exception:
                    continue

                video_seq_id = str(video_entry.get("seq_id", video_id))
                gt_boxes = gt_by_img.get(image_id, [])
                if args.nonempty_mode == "target_bbox":
                    is_empty = len(gt_boxes) == 0
                else:
                    is_nonempty = (image_id in nonempty_image_ids) or (video_seq_id in nonempty_seq_ids)
                    is_empty = not is_nonempty

                rec = {
                    "dataset": str(args.dataset),
                    "split": str(args.split),
                    "video_id": video_id,
                    "frame_idx": int(fr.get("frame_idx", video_total)),
                    "image_id": image_id,
                    "is_empty": bool(is_empty),
                    "gt_num_boxes": int(len(gt_boxes)),
                    "target_category_id": int(target_cid) if target_cid is not None else None,
                    "nonempty_mode": str(args.nonempty_mode),
                }
                f.write(json.dumps(rec) + "\n")

                total_frames += 1
                video_total += 1
                if is_empty:
                    total_empty += 1
                    video_empty += 1
                else:
                    total_non_empty += 1

            if video_total > 0:
                per_video.append({
                    "dataset": str(args.dataset),
                    "split": str(args.split),
                    "video_id": video_id,
                    "num_frames": int(video_total),
                    "num_empty_frames": int(video_empty),
                    "num_non_empty_frames": int(video_total - video_empty),
                    "empty_ratio": float(video_empty) / float(video_total),
                })

    summary = {
        "dataset": str(args.dataset),
        "split": str(args.split),
        "gt_json": str(gt_path),
        "seq_json": str(seq_path),
        "target_category_id": int(target_cid) if target_cid is not None else None,
        "nonempty_mode": str(args.nonempty_mode),
        "num_frames": int(total_frames),
        "num_empty_frames": int(total_empty),
        "num_non_empty_frames": int(total_non_empty),
        "empty_ratio": float(total_empty) / float(total_frames) if total_frames > 0 else 0.0,
        "per_video": per_video,
        "out_jsonl_gz": os.path.basename(out_path),
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved:", out_path)
    print("Saved:", summary_path)


if __name__ == "__main__":
    main()
