#!/usr/bin/env python
# -*- coding: utf-8 -*-

# scripts/summarize_v4_log.py
# Summarize v4 per-frame logs: low_motion/strong_skip/escape_soft/escape_hard and gm_z stats.
# Python 3.9 compatible. Comments in English only.

import argparse
import json
from typing import Any, Dict, List

import numpy as np


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def pct(a: int, b: int) -> float:
    return float(a) / float(b) if b > 0 else 0.0


def quantiles(vals: List[float], qs: List[float]) -> Dict[str, float]:
    if len(vals) == 0:
        return {str(q): 0.0 for q in qs}
    arr = np.asarray(vals, dtype=np.float32)
    return {str(q): float(np.quantile(arr, q)) for q in qs}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Temporal SAHI v4 logs.")
    parser.add_argument("--log", type=str, required=True)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    logs = load_json(args.log)
    if not isinstance(logs, list):
        raise ValueError("log must be a JSON list")

    n = len(logs)

    mode_counts = {"skip": 0, "subset": 0, "full": 0}
    do_sahi = 0

    tiles0_when_do = 0

    # flags
    low_motion_cnt = 0
    empty_change_cnt = 0
    strong_skip_cnt = 0
    escape_soft_cnt = 0
    escape_hard_cnt = 0
    escape_any_cnt = 0

    # escape but skip (should be 0 ideally)
    escape_any_skip_cnt = 0

    # distributions
    gm_raw_vals = []
    gm_vals = []
    gm_z_vals = []
    gm_mu_vals = []
    tiles_sel_vals = []

    low_mode = {"skip": 0, "subset": 0, "full": 0}

    for r in logs:
        mode = str(r.get("mode", "skip"))
        if mode not in mode_counts:
            mode = "skip"
        mode_counts[mode] += 1

        ds = bool(r.get("do_sahi", False))
        if ds:
            do_sahi += 1
            tiles_sel_vals.append(float(r.get("tiles_selected", 0)))
            if int(r.get("tiles_selected", 0)) == 0:
                tiles0_when_do += 1

        low = bool(r.get("low_motion", False))
        if low:
            low_motion_cnt += 1
            low_mode[mode] += 1

        if bool(r.get("empty_change", False)):
            empty_change_cnt += 1

        if bool(r.get("strong_skip", False)):
            strong_skip_cnt += 1

        esc_soft = bool(r.get("escape_soft", False))
        esc_hard = bool(r.get("escape_hard", False))
        esc_any = esc_soft or esc_hard
        if esc_soft:
            escape_soft_cnt += 1
        if esc_hard:
            escape_hard_cnt += 1
        if esc_any:
            escape_any_cnt += 1
            if mode == "skip":
                escape_any_skip_cnt += 1

        gm_raw_vals.append(float(r.get("global_motion_raw", 0.0)))
        gm_vals.append(float(r.get("global_motion", 0.0)))
        gm_mu_vals.append(float(r.get("gm_mu", 0.0)))
        gm_z_vals.append(float(r.get("gm_z", 0.0)))

    qs = [0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]

    out = {
        "num_frames": int(n),
        "do_sahi_ratio": pct(do_sahi, n),
        "mode_counts": mode_counts,
        "mode_ratios": {k: pct(v, n) for k, v in mode_counts.items()},
        "tiles_selected_zero_when_do_sahi": int(tiles0_when_do),

        "low_motion_count": int(low_motion_cnt),
        "low_motion_ratio": pct(low_motion_cnt, n),
        "low_motion_mode_counts": low_mode,

        "empty_change_count": int(empty_change_cnt),
        "empty_change_ratio": pct(empty_change_cnt, n),

        "strong_skip_count": int(strong_skip_cnt),
        "strong_skip_ratio": pct(strong_skip_cnt, n),

        "escape_soft_count": int(escape_soft_cnt),
        "escape_soft_ratio": pct(escape_soft_cnt, n),
        "escape_hard_count": int(escape_hard_cnt),
        "escape_hard_ratio": pct(escape_hard_cnt, n),
        "escape_any_count": int(escape_any_cnt),
        "escape_any_ratio": pct(escape_any_cnt, n),
        "escape_any_skip_count": int(escape_any_skip_cnt),

        "tiles_selected_when_do_sahi_quantiles": quantiles(tiles_sel_vals, qs),

        "global_motion_raw_quantiles": quantiles(gm_raw_vals, qs),
        "global_motion_quantiles": quantiles(gm_vals, qs),
        "gm_mu_quantiles": quantiles(gm_mu_vals, qs),
        "gm_z_quantiles": quantiles(gm_z_vals, qs),
    }

    print(json.dumps(out, indent=2))

    if str(args.out).strip() != "":
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print("Saved summary to:", args.out)


if __name__ == "__main__":
    main()
