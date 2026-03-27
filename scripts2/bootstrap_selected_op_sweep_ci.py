#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

BASELINE = {
    "tag": "baseline_ours_selected",
    "fbd_per_video": "output/thesis2/phase4_candidates/experiments/2026-03-11_v7_noconfirm_fbdsv_val/kpi/per_video.csv",
    "icct_per_video": "output/thesis2/phase4_candidates/experiments/2026-03-11_v7_noconfirm_icct_dev/kpi/per_video.csv",
    "fbd_summary": "output/thesis2/phase4_candidates/experiments/2026-03-11_v7_noconfirm_fbdsv_val/kpi/summary.json",
    "icct_summary": "output/thesis2/phase4_candidates/experiments/2026-03-11_v7_noconfirm_icct_dev/kpi/summary.json",
    "readiness": "output/thesis2/phase4_candidates/experiments/readiness_v7_noconfirm_fbdsv_val.json",
}

SWEEPS = [
    "2026-03-16_op_lowbudget",
    "2026-03-16_op_midplus",
    "2026-03-16_op_highbudget",
]


def fmt(x: Optional[float], digits: int = 4) -> str:
    if x is None:
        return "--"
    return f"{x:.{digits}f}"


def mean(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def quantile(xs: List[float], q: float) -> Optional[float]:
    if not xs:
        return None
    arr = np.asarray(xs, dtype=np.float32)
    return float(np.quantile(arr, q))


def ci_from_samples(xs: List[float]) -> Dict[str, Optional[float]]:
    return {
        "mean": mean(xs),
        "ci_low": quantile(xs, 0.025),
        "ci_high": quantile(xs, 0.975),
    }


def load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def bootstrap_video_metric(rows: List[Dict[str, Any]], column: str, n_boot: int, seed: int) -> Dict[str, Optional[float]]:
    vals: List[float] = []
    for r in rows:
        v = r.get(column, "")
        if v in ("", None):
            continue
        vals.append(float(v))
    if not vals:
        return {"mean": None, "ci_low": None, "ci_high": None}
    rng = np.random.default_rng(seed)
    idx = np.arange(len(vals))
    arr = np.asarray(vals, dtype=np.float32)
    boot: List[float] = []
    for _ in range(n_boot):
        sample = rng.choice(idx, size=len(idx), replace=True)
        boot.append(float(np.mean(arr[sample])))
    return ci_from_samples(boot)


def get_small_recall(summary: Dict[str, Any], tau: str = "0.02") -> Optional[float]:
    val = (summary.get("small_tau", {}) or {}).get(tau, {})
    if isinstance(val, dict):
        x = val.get("Recall", None)
        return None if x is None else float(x)
    return None


def get_episode_capture(readiness: Dict[str, Any]) -> Optional[float]:
    x = (readiness.get("kpi", {}) or {}).get("overall", {}).get("track_init_recall", None)
    return None if x is None else float(x)


def get_stable_delay_p50(readiness: Dict[str, Any]) -> Optional[float]:
    x = (readiness.get("kpi", {}) or {}).get("overall", {}).get("stable_ttd_p50", None)
    return None if x is None else float(x)


def get_icct_summary_cost(icct: Dict[str, Any], key: str) -> Optional[float]:
    cost = (icct.get("cost", {}) or {})
    x = cost.get(key, None)
    if isinstance(x, dict):
        x = x.get("mean", None)
    return None if x is None else float(x)


def row_from_paths(tag: str, fbd_per_video: Path, icct_per_video: Path, fbd_summary: Path, icct_summary: Path, readiness: Path, n_boot: int, seed: int) -> Dict[str, Any]:
    fbd_rows = read_csv_rows(fbd_per_video)
    icct_rows = read_csv_rows(icct_per_video)
    fbd = load_json(fbd_summary)
    icct = load_json(icct_summary)
    ready = load_json(readiness)

    on_col = "verifier_on_ratio" if "verifier_on_ratio" in icct_rows[0] else "sahi_on_ratio"
    tiles_col = "avg_verifier_regions" if "avg_verifier_regions" in icct_rows[0] else "avg_tiles"

    return {
        "tag": tag,
        "summary": {
            "recall_002": get_small_recall(fbd, "0.02"),
            "episode_capture": get_episode_capture(ready),
            "stable_delay_p50": get_stable_delay_p50(ready),
            "effr_07": float((((icct.get("safety", {}) or {}).get("EFFR", {}) or {}).get("0.7", 0.0) or 0.0)),
            "on_ratio": get_icct_summary_cost(icct, "sahi_on_ratio"),
            "avg_tiles": get_icct_summary_cost(icct, "avg_tiles_per_frame"),
            "fps": get_icct_summary_cost(icct, "fps"),
        },
        "bootstrap": {
            "fbd_video_smallhit_tau002": bootstrap_video_metric(fbd_rows, "has_small_tau_0_02", n_boot, seed + 1),
            "fbd_delay_p50": bootstrap_video_metric(fbd_rows, "delay_p50", n_boot, seed + 2),
            "icct_on_ratio": bootstrap_video_metric(icct_rows, on_col, n_boot, seed + 3),
            "icct_avg_tiles": bootstrap_video_metric(icct_rows, tiles_col, n_boot, seed + 4),
            "icct_empty_fp_rate": bootstrap_video_metric(icct_rows, "empty_fp_rate", n_boot, seed + 5),
        },
    }


def render_md(rows: List[Dict[str, Any]]) -> str:
    lines: List[str] = ["# Selected Operating Point Bootstrap CI", ""]
    lines.append("This table strengthens the local-knee interpretation around the selected operating point using only existing sweep outputs.")
    lines.append("")
    lines.append("## Raw Summary Metrics")
    lines.append("")
    lines.append("| Tag | Recall@0.02 | Episode capture | Stable delay p50 | EFFR@0.7 | Verifier on-ratio | Avg. tiles/frame | FPS |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        s = r["summary"]
        lines.append(
            f"| {r['tag']} | {fmt(s['recall_002'], 6)} | {fmt(s['episode_capture'], 6)} | {fmt(s['stable_delay_p50'], 2)} | {fmt(s['effr_07'], 6)} | {fmt(s['on_ratio'], 6)} | {fmt(s['avg_tiles'], 6)} | {fmt(s['fps'], 3)} |"
        )
    lines.append("")
    lines.append("## Video-Level Bootstrap Support")
    lines.append("")
    lines.append("| Tag | Video small-hit@0.02 mean | 95% CI | Delay p50 mean | 95% CI | On-ratio mean | 95% CI | Empty FP rate mean | 95% CI |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        b = r["bootstrap"]
        lines.append(
            "| {tag} | {a} | [{al}, {ah}] | {b0} | [{bl}, {bh}] | {c} | [{cl}, {ch}] | {d} | [{dl}, {dh}] |".format(
                tag=r["tag"],
                a=fmt(b["fbd_video_smallhit_tau002"]["mean"], 4),
                al=fmt(b["fbd_video_smallhit_tau002"]["ci_low"], 4),
                ah=fmt(b["fbd_video_smallhit_tau002"]["ci_high"], 4),
                b0=fmt(b["fbd_delay_p50"]["mean"], 2),
                bl=fmt(b["fbd_delay_p50"]["ci_low"], 2),
                bh=fmt(b["fbd_delay_p50"]["ci_high"], 2),
                c=fmt(b["icct_on_ratio"]["mean"], 4),
                cl=fmt(b["icct_on_ratio"]["ci_low"], 4),
                ch=fmt(b["icct_on_ratio"]["ci_high"], 4),
                d=fmt(b["icct_empty_fp_rate"]["mean"], 4),
                dl=fmt(b["icct_empty_fp_rate"]["ci_low"], 4),
                dh=fmt(b["icct_empty_fp_rate"]["ci_high"], 4),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap local robustness around the selected operating point.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-boot", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260323)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    rows.append(row_from_paths(
        BASELINE["tag"],
        repo / BASELINE["fbd_per_video"],
        repo / BASELINE["icct_per_video"],
        repo / BASELINE["fbd_summary"],
        repo / BASELINE["icct_summary"],
        repo / BASELINE["readiness"],
        args.n_boot,
        args.seed,
    ))

    root = repo / 'output/thesis2/selected_point_sweeps/experiments'
    for i, tag in enumerate(SWEEPS, start=1):
        rows.append(row_from_paths(
            tag,
            root / f"{tag}_fbdsv_val/kpi/per_video.csv",
            root / f"{tag}_icct_dev/kpi/per_video.csv",
            root / f"{tag}_fbdsv_val/kpi/summary.json",
            root / f"{tag}_icct_dev/kpi/summary.json",
            root / f"readiness_{tag}_fbdsv_val.json",
            args.n_boot,
            args.seed + i * 100,
        ))

    (out_dir / 'selected_op_bootstrap.json').write_text(json.dumps(rows, indent=2))
    (out_dir / 'selected_op_bootstrap.md').write_text(render_md(rows))
    print(f"Saved: {out_dir / 'selected_op_bootstrap.json'}")
    print(f"Saved: {out_dir / 'selected_op_bootstrap.md'}")


if __name__ == '__main__':
    main()
