#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import gzip
import json
import os
from typing import Any, Dict

from sahi_base.tiler import count_slices


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a dense SAHI-on per-frame cost log from eval-ready GT.")
    parser.add_argument("--gt-json", required=True)
    parser.add_argument("--seq-json", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--slice-height", type=int, required=True)
    parser.add_argument("--slice-width", type=int, required=True)
    parser.add_argument("--overlap-height-ratio", type=float, required=True)
    parser.add_argument("--overlap-width-ratio", type=float, required=True)
    parser.add_argument("--out-path", required=True)
    parser.add_argument("--latency-ms", type=float, default=0.0)
    args = parser.parse_args()

    gt = load_json(args.gt_json)
    seq_list = load_json(args.seq_json)
    img_meta: Dict[int, Dict[str, Any]] = {}
    for im in gt.get("images", []):
        try:
            img_meta[int(im["id"])] = im
        except Exception:
            continue

    ensure_dir(os.path.dirname(os.path.abspath(args.out_path)))
    num_rows = 0
    with gzip.open(args.out_path, "wt", encoding="utf-8") as f:
        for video_entry in seq_list:
            if not isinstance(video_entry, dict):
                continue
            video_id = str(video_entry.get("video_id", video_entry.get("id", "")))
            frames = video_entry.get("frames", [])
            for frame_idx, fr in enumerate(frames):
                if not isinstance(fr, dict):
                    continue
                image_id = fr.get("image_id", None)
                if image_id is None:
                    continue
                try:
                    iid = int(image_id)
                except Exception:
                    continue
                im = img_meta.get(iid, {})
                h = int(im.get("height", 0))
                w = int(im.get("width", 0))
                tiles = count_slices(
                    height=h,
                    width=w,
                    slice_height=int(args.slice_height),
                    slice_width=int(args.slice_width),
                    overlap_height_ratio=float(args.overlap_height_ratio),
                    overlap_width_ratio=float(args.overlap_width_ratio),
                )
                rec = {
                    "dataset": str(args.dataset),
                    "split": str(args.split),
                    "video_id": str(video_id),
                    "frame_idx": int(frame_idx),
                    "image_id": int(iid),
                    "sahi_on": 1,
                    "tiles_used": int(tiles),
                    "latency_ms": float(args.latency_ms),
                }
                f.write(json.dumps(rec) + "\n")
                num_rows += 1

    print("Saved:", args.out_path)
    print("Rows:", num_rows)


if __name__ == "__main__":
    main()
