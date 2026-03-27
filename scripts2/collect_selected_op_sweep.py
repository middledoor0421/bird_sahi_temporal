#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


BASELINE = {
    "tag": "baseline_ours_selected",
    "fbd_summary": "output/thesis2/phase4_candidates/experiments/2026-03-11_v7_noconfirm_fbdsv_val/kpi/summary.json",
    "icct_summary": "output/thesis2/phase4_candidates/experiments/2026-03-11_v7_noconfirm_icct_dev/kpi/summary.json",
    "readiness": "output/thesis2/phase4_candidates/experiments/readiness_v7_noconfirm_fbdsv_val.json",
}

SWEEPS = [
    "2026-03-16_op_lowbudget",
    "2026-03-16_op_midplus",
    "2026-03-16_op_highbudget",
]


def load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def load_json_optional(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r") as f:
        return json.load(f)


def get_small_recall(summary: Dict[str, Any], tau: str = "0.02") -> float:
    val = summary.get("small_tau", {}).get(tau, {})
    if isinstance(val, dict):
        return float(val.get("Recall", 0.0) or 0.0)
    return 0.0


def get_episode_capture(readiness: Dict[str, Any]) -> float:
    return float(readiness.get("kpi", {}).get("overall", {}).get("track_init_recall", 0.0) or 0.0)


def get_stable_delay_p50(readiness: Dict[str, Any]) -> float:
    val = readiness.get("kpi", {}).get("overall", {}).get("stable_ttd_p50", None)
    if val is None:
        return 0.0
    return float(val)


def fmt_optional(x: Any, digits: int = 6) -> str:
    if x is None:
        return "--"
    if isinstance(x, (int, float)):
        return f"{x:.{digits}f}"
    return str(x)


def row_from_paths(tag: str, fbd_path: Path, icct_path: Path, readiness_path: Path) -> Dict[str, Any]:
    fbd = load_json_optional(fbd_path)
    icct = load_json_optional(icct_path)
    readiness = load_json_optional(readiness_path)
    cost = (icct or {}).get("cost", {}) or {}
    avg_tiles = cost.get("avg_tiles_per_frame", 0.0)
    if isinstance(avg_tiles, dict):
        avg_tiles = avg_tiles.get("mean", 0.0)
    return {
        "tag": tag,
        "recall_002": get_small_recall(fbd, "0.02") if fbd is not None else None,
        "effr_07": float((icct or {}).get("safety", {}).get("EFFR", {}).get("0.7", 0.0) or 0.0) if icct is not None else None,
        "sahi_on_ratio": float(cost.get("sahi_on_ratio", 0.0) or 0.0) if icct is not None else None,
        "avg_tiles": float(avg_tiles or 0.0) if icct is not None else None,
        "fps": float(cost.get("fps", 0.0) or 0.0) if icct is not None else None,
        "episode_capture": get_episode_capture(readiness) if readiness is not None else None,
        "stable_delay_p50": get_stable_delay_p50(readiness) if readiness is not None else None,
    }


def render_markdown(rows: List[Dict[str, Any]]) -> str:
    lines = [
        "# Selected Operating Point Small Sweep",
        "",
        "| Tag | Recall@0.02 | Episode capture | Stable delay p50 | Bird-EFFR@0.7 | SAHI on-ratio | Avg. tiles/frame | FPS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| {tag} | {recall} | {cap} | {delay} | {effr} | {on} | {tiles} | {fps} |".format(
                tag=r["tag"],
                recall=fmt_optional(r["recall_002"], 6),
                cap=fmt_optional(r["episode_capture"], 6),
                delay=fmt_optional(r["stable_delay_p50"], 2),
                effr=fmt_optional(r["effr_07"], 6),
                on=fmt_optional(r["sahi_on_ratio"], 6),
                tiles=fmt_optional(r["avg_tiles"], 6),
                fps=fmt_optional(r["fps"], 3),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect selected-point sweep results.")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    rows.append(
        row_from_paths(
            BASELINE["tag"],
            repo / BASELINE["fbd_summary"],
            repo / BASELINE["icct_summary"],
            repo / BASELINE["readiness"],
        )
    )

    root = repo / "output/thesis2/selected_point_sweeps/experiments"
    for tag in SWEEPS:
        rows.append(
            row_from_paths(
                tag,
                root / f"{tag}_fbdsv_val/kpi/summary.json",
                root / f"{tag}_icct_dev/kpi/summary.json",
                root / f"readiness_{tag}_fbdsv_val.json",
            )
        )

    (out_dir / "selected_op_sweep.json").write_text(json.dumps(rows, indent=2))
    (out_dir / "selected_op_sweep.md").write_text(render_markdown(rows))
    print(f"Saved: {out_dir / 'selected_op_sweep.json'}")
    print(f"Saved: {out_dir / 'selected_op_sweep.md'}")


if __name__ == "__main__":
    main()
