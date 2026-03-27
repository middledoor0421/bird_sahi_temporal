#!/usr/bin/env python
# -*- coding: utf-8 -*-

# scripts/summarize_v3_log.py
# Summarize v3 per-frame logs.
# Python 3.9 compatible. Comments in English only.

import argparse
import json
from typing import Any, Dict, List, Optional

import numpy as np


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: str, obj: Any) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def safe_get(d: Dict, keys: List[str], default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def pct(a: int, b: int) -> float:
    return float(a) / float(b) if b > 0 else 0.0


def summarize(logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(logs)
    do_sahi = sum(1 for x in logs if bool(x.get("do_sahi", False)))
    mode_full = sum(1 for x in logs if str(x.get("mode", "")) == "full")
    mode_subset = sum(1 for x in logs if str(x.get("mode", "")) == "subset")
    mode_skip = n - mode_full - mode_subset

    tiles_sel = [int(x.get("num_tiles_selected", 0)) for x in logs if bool(x.get("do_sahi", False))]
    risk_scores = [float(x.get("risk_score", 0.0)) for x in logs]

    def q(arr: List[float], p: float) -> float:
        if len(arr) == 0:
            return 0.0
        return float(np.quantile(np.asarray(arr, dtype=np.float32), p))

    # Flags counts
    flag_names = ["low_motion", "blur_susp", "uncertain", "tracking_fail"]
    flag_counts: Dict[str, int] = {}
    for fn in flag_names:
        c = 0
        for x in logs:
            v = safe_get(x, ["flags", fn], False)
            if bool(v):
                c += 1
        flag_counts[fn] = int(c)

    out: Dict[str, Any] = {
        "num_frames": int(n),
        "do_sahi_frames": int(do_sahi),
        "do_sahi_ratio": float(pct(do_sahi, n)),
        "mode_counts": {
            "full": int(mode_full),
            "subset": int(mode_subset),
            "skip": int(mode_skip),
        },
        "mode_ratios": {
            "full": float(pct(mode_full, n)),
            "subset": float(pct(mode_subset, n)),
            "skip": float(pct(mode_skip, n)),
        },
        "tiles_selected_when_sahi": {
            "mean": float(np.mean(tiles_sel)) if len(tiles_sel) > 0 else 0.0,
            "p50": q([float(x) for x in tiles_sel], 0.50),
            "p95": q([float(x) for x in tiles_sel], 0.95),
            "max": int(max(tiles_sel)) if len(tiles_sel) > 0 else 0,
        },
        "risk_score": {
            "mean": float(np.mean(risk_scores)) if len(risk_scores) > 0 else 0.0,
            "p50": q(risk_scores, 0.50),
            "p95": q(risk_scores, 0.95),
            "max": float(max(risk_scores)) if len(risk_scores) > 0 else 0.0,
        },
        "flag_counts": flag_counts,
        "flag_ratios": {k: float(pct(v, n)) for k, v in flag_counts.items()},
    }

    # A couple useful conditional summaries
    # Risk stats for do_sahi vs skip
    rs_sahi = [float(x.get("risk_score", 0.0)) for x in logs if bool(x.get("do_sahi", False))]
    rs_skip = [float(x.get("risk_score", 0.0)) for x in logs if not bool(x.get("do_sahi", False))]
    out["risk_score_by_action"] = {
        "do_sahi_mean": float(np.mean(rs_sahi)) if len(rs_sahi) > 0 else 0.0,
        "skip_mean": float(np.mean(rs_skip)) if len(rs_skip) > 0 else 0.0,
        "do_sahi_p95": q(rs_sahi, 0.95),
        "skip_p95": q(rs_skip, 0.95),
    }

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Temporal SAHI v3 logs.")
    parser.add_argument("--log", type=str, required=True, help="Path to v3 log json (list).")
    parser.add_argument("--out", type=str, default="", help="Optional output summary json path.")
    args = parser.parse_args()

    logs = load_json(args.log)
    if not isinstance(logs, list):
        raise ValueError("Log file must be a JSON list of per-frame dicts.")

    summ = summarize(logs)

    print(json.dumps(summ, indent=2))
    if str(args.out).strip() != "":
        save_json(args.out, summ)
        print("Saved summary to:", args.out)


if __name__ == "__main__":
    main()
