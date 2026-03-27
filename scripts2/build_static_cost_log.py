#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import gzip
import json
import os
from typing import Any, Dict, List


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a synthetic per-frame cost log for no-SAHI baselines.")
    parser.add_argument("--seq-json", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--out-path", required=True)
    parser.add_argument("--latency-ms", type=float, default=0.0)
    args = parser.parse_args()

    seq_list = load_json(args.seq_json)
    ensure_dir(os.path.dirname(os.path.abspath(args.out_path)))
    num_rows = 0

    with gzip.open(args.out_path, "wt", encoding="utf-8") as f:
        for video_entry in seq_list:
            if not isinstance(video_entry, dict):
                continue
            video_id = str(video_entry.get("video_id", video_entry.get("id", "")))
            frames: List[Dict[str, Any]] = video_entry.get("frames", [])
            for frame_idx, fr in enumerate(frames):
                if not isinstance(fr, dict):
                    continue
                image_id = fr.get("image_id", None)
                if image_id is None:
                    continue
                rec = {
                    "dataset": str(args.dataset),
                    "split": str(args.split),
                    "video_id": str(video_id),
                    "frame_idx": int(frame_idx),
                    "image_id": int(image_id),
                    "sahi_on": 0,
                    "tiles_used": 0,
                    "latency_ms": float(args.latency_ms),
                }
                f.write(json.dumps(rec) + "\n")
                num_rows += 1

    print("Saved:", args.out_path)
    print("Rows:", num_rows)


if __name__ == "__main__":
    main()
