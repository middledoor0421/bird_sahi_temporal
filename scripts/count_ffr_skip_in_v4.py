#!/usr/bin/env python
# -*- coding: utf-8 -*-

# scripts/count_ffr_skip_in_v4.py
# Python 3.9 compatible. Comments in English only.

import argparse
import json
from typing import Any, Dict, List


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Count skip rate on FFR frames using v4 logs.")
    parser.add_argument("--ffr-list", type=str, required=True, help="JSON list of image_ids (FFR frames).")
    parser.add_argument("--v4-log", type=str, required=True, help="v4 log json (list).")

    args = parser.parse_args()

    ffr_ids = load_json(args.ffr_list)
    logs = load_json(args.v4_log)

    if not isinstance(ffr_ids, list):
        raise ValueError("--ffr-list must be a JSON list.")
    if not isinstance(logs, list):
        raise ValueError("--v4-log must be a JSON list of dicts.")

    ffr_set = set(int(x) for x in ffr_ids)

    log_by_img: Dict[int, Dict] = {}
    for row in logs:
        img_id = int(row.get("image_id"))
        log_by_img[img_id] = row

    total = 0
    cnt_skip = 0
    cnt_subset = 0
    cnt_full = 0
    cnt_missing_log = 0

    cnt_escape = 0
    cnt_escape_skip = 0
    cnt_tiles0 = 0

    for img_id in ffr_set:
        if img_id not in log_by_img:
            cnt_missing_log += 1
            continue

        row = log_by_img[img_id]
        total += 1

        mode = str(row.get("mode", ""))
        if mode == "skip":
            cnt_skip += 1
        elif mode == "subset":
            cnt_subset += 1
        elif mode == "full":
            cnt_full += 1

        esc = bool(row.get("escape_hatch", False))
        if esc:
            cnt_escape += 1
            if mode == "skip":
                cnt_escape_skip += 1

        if bool(row.get("do_sahi", False)) and int(row.get("tiles_selected", 0)) == 0:
            cnt_tiles0 += 1

    def ratio(a: int, b: int) -> float:
        return float(a) / float(b) if b > 0 else 0.0

    out = {
        "ffr_total_in_list": int(len(ffr_set)),
        "ffr_total_with_log": int(total),
        "missing_log_count": int(cnt_missing_log),
        "mode_counts": {
            "skip": int(cnt_skip),
            "subset": int(cnt_subset),
            "full": int(cnt_full),
        },
        "mode_ratios": {
            "skip": ratio(cnt_skip, total),
            "subset": ratio(cnt_subset, total),
            "full": ratio(cnt_full, total),
        },
        "escape_hatch": {
            "count": int(cnt_escape),
            "escape_skip_count": int(cnt_escape_skip),
        },
        "tiles_selected_zero_when_do_sahi": int(cnt_tiles0),
    }

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
