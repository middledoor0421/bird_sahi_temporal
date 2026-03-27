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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build missing image path list from COCO manifest.")
    parser.add_argument("--gt-json", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--path-key", default="file_name", choices=["file_name", "abs_file_name"])
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    gt = load_json(args.gt_json)
    images: List[Dict[str, Any]] = list(gt.get("images", []))

    missing_rel: List[str] = []
    total = 0
    existing = 0
    for im in images:
        total += 1
        rel = str(im.get("file_name", ""))
        if not rel:
            continue
        abs_path = str(im.get("abs_file_name", "")) if args.path_key == "abs_file_name" else os.path.join(args.image_root, rel)
        if os.path.isfile(abs_path):
            existing += 1
        else:
            missing_rel.append(rel)

    summary = {
        "gt_json": os.path.abspath(args.gt_json),
        "image_root": os.path.abspath(args.image_root),
        "num_images_total": int(total),
        "num_images_existing": int(existing),
        "num_images_missing": int(len(missing_rel)),
        "existing_ratio": float(existing) / float(total) if total > 0 else 0.0,
    }

    txt_path = os.path.join(args.out_dir, "missing_files.txt")
    summary_path = os.path.join(args.out_dir, "missing_files_summary.json")
    with open(txt_path, "w") as f:
        for rel in missing_rel:
            f.write(rel + "\n")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved:", txt_path)
    print("Saved:", summary_path)
    print("Missing:", len(missing_rel))


if __name__ == "__main__":
    main()
