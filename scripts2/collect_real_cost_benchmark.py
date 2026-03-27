#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import gzip
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_RUNS = {
    "icct_dev": {
        "YOLO": "output/thesis2/experiments/2026-03-08_yolo_icct_dev",
        "AlwaysVerify": "output/thesis2/aligned_verify_baselines/experiments/2026-03-14_yolo_sahi_always_verify_icct_dev",
        "KeyframeVerify": "output/thesis2/aligned_verify_baselines/experiments/2026-03-14_yolo_sahi_keyframe_verify_icct_dev",
        "Ours (selected)": "output/thesis2/phase4_candidates/experiments/2026-03-11_v7_noconfirm_icct_dev",
    },
    "fbdsv_val": {
        "YOLO": "output/thesis2/experiments/2026-03-08_yolo_fbdsv_val",
        "AlwaysVerify": "output/thesis2/aligned_verify_baselines/experiments/2026-03-14_yolo_sahi_always_verify_fbdsv_val",
        "KeyframeVerify": "output/thesis2/aligned_verify_baselines/experiments/2026-03-14_yolo_sahi_keyframe_verify_fbdsv_val",
        "Ours (selected)": "output/thesis2/phase4_candidates/experiments/2026-03-11_v7_noconfirm_fbdsv_val",
    },
}


def load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def find_first(patterns: List[Path]) -> Optional[Path]:
    for path in patterns:
        if path.exists():
            return path
    return None


def pct(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    rank = q * (len(values) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(values[lo])
    frac = rank - lo
    return float(values[lo] * (1.0 - frac) + values[hi] * frac)


def read_cost_stats(cost_path: Optional[Path]) -> Dict[str, Optional[float]]:
    if cost_path is None or not cost_path.exists():
        return {
            "latency_ms_mean": None,
            "latency_ms_p50": None,
            "latency_ms_p90": None,
            "latency_ms_p95": None,
        }
    values: List[float] = []
    with gzip.open(cost_path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            try:
                values.append(float(rec.get("latency_ms", 0.0)))
            except Exception:
                continue
    values.sort()
    if not values:
        return {
            "latency_ms_mean": None,
            "latency_ms_p50": None,
            "latency_ms_p90": None,
            "latency_ms_p95": None,
        }
    return {
        "latency_ms_mean": float(sum(values) / len(values)),
        "latency_ms_p50": pct(values, 0.50),
        "latency_ms_p90": pct(values, 0.90),
        "latency_ms_p95": pct(values, 0.95),
    }


def format_num(x: Optional[float], digits: int = 3) -> str:
    if x is None:
        return "--"
    return f"{x:.{digits}f}"


def collect_run(repo: Path, run_root: str) -> Dict[str, Any]:
    run_dir = repo / run_root
    infer_dir = run_dir / "infer"
    kpi_dir = run_dir / "kpi"

    stats_path = find_first(sorted(infer_dir.glob("stats_*.json")))
    summary_path = kpi_dir / "summary.json"
    if stats_path is None:
        raise FileNotFoundError(f"Missing stats JSON under {infer_dir}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary JSON: {summary_path}")

    stats = load_json(stats_path)
    summary = load_json(summary_path)

    num_frames = int(stats.get("num_frames", 0) or 0)
    elapsed_sec = float(stats.get("elapsed_sec", 0.0) or 0.0)
    fps_stats = stats.get("fps", None)
    fps_stats = float(fps_stats) if fps_stats is not None else None

    summary_cost = summary.get("cost", {}) if isinstance(summary.get("cost", {}), dict) else {}
    cost_path_str = ""
    if isinstance(summary.get("run_meta", {}), dict):
        maybe = summary["run_meta"].get("cost_per_frame", "")
        if isinstance(maybe, str):
            cost_path_str = maybe
    if not cost_path_str:
        maybe = stats.get("cost_per_frame_path", "")
        if isinstance(maybe, str):
            cost_path_str = maybe
    cost_path = (repo / cost_path_str) if cost_path_str else None
    cost_latency = read_cost_stats(cost_path)

    end_to_end_latency_ms = None
    if num_frames > 0 and elapsed_sec > 0.0:
        end_to_end_latency_ms = (elapsed_sec / float(num_frames)) * 1000.0

    fps = fps_stats
    if fps is None and end_to_end_latency_ms is not None and end_to_end_latency_ms > 0.0:
        fps = 1000.0 / end_to_end_latency_ms

    sahi_on_ratio = summary_cost.get("sahi_on_ratio", None)
    if sahi_on_ratio is None:
        sahi_on_ratio = summary_cost.get("verifier_on_ratio", None)
    if sahi_on_ratio is not None:
        sahi_on_ratio = float(sahi_on_ratio)
    avg_tiles = summary_cost.get("avg_tiles_per_frame", None)
    if isinstance(avg_tiles, dict):
        avg_tiles = avg_tiles.get("mean", None)
    if avg_tiles is None:
        avg_tiles = stats.get("avg_tiles_selected", None)
    if avg_tiles is not None:
        avg_tiles = float(avg_tiles)

    return {
        "run_dir": str(run_dir),
        "stats_path": str(stats_path),
        "summary_path": str(summary_path),
        "cost_path": str(cost_path) if cost_path is not None else None,
        "num_frames": num_frames,
        "elapsed_sec": elapsed_sec,
        "fps": fps,
        "latency_ms_end_to_end_mean": end_to_end_latency_ms,
        "latency_ms_process_mean": cost_latency["latency_ms_mean"],
        "latency_ms_process_p50": cost_latency["latency_ms_p50"],
        "latency_ms_process_p90": cost_latency["latency_ms_p90"],
        "latency_ms_process_p95": cost_latency["latency_ms_p95"],
        "sahi_on_ratio": sahi_on_ratio,
        "avg_tiles_per_frame": avg_tiles,
    }


def render_markdown(results: Dict[str, Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("# Real Cost Benchmark")
    lines.append("")
    lines.append("Measured from existing end-to-end inference runs. Mean latency and FPS are real wall-clock values computed from total elapsed time. Process-level p50/p90/p95 latency are reported separately when per-frame cost logs are available.")
    lines.append("")
    for split_name, rows in results.items():
        lines.append(f"## {split_name}")
        lines.append("")
        lines.append("| Method | Frames | Elapsed (s) | End-to-end mean latency (ms) | Process P50 (ms) | Process P90 (ms) | Process P95 (ms) | FPS | Verifier on-ratio | Avg. tiles/frame |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for method, rec in rows.items():
            lines.append(
                "| {method} | {frames} | {elapsed} | {mean} | {p50} | {p90} | {p95} | {fps} | {sahi} | {tiles} |".format(
                    method=method,
                    frames=rec["num_frames"],
                    elapsed=format_num(rec["elapsed_sec"], 3),
                    mean=format_num(rec["latency_ms_end_to_end_mean"], 2),
                    p50=format_num(rec["latency_ms_process_p50"], 2),
                    p90=format_num(rec["latency_ms_process_p90"], 2),
                    p95=format_num(rec["latency_ms_process_p95"], 2),
                    fps=format_num(rec["fps"], 2),
                    sahi=format_num(rec["sahi_on_ratio"], 4),
                    tiles=format_num(rec["avg_tiles_per_frame"], 4),
                )
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect real runtime/cost benchmark tables from existing run artifacts.")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[str, Any]] = {}
    for split_name, split_runs in DEFAULT_RUNS.items():
        rows: Dict[str, Any] = {}
        for method, run_root in split_runs.items():
            rows[method] = collect_run(repo, run_root)
        results[split_name] = rows

    json_path = out_dir / "real_cost_benchmark.json"
    md_path = out_dir / "real_cost_benchmark.md"
    with json_path.open("w") as f:
        json.dump(results, f, indent=2)
    md_path.write_text(render_markdown(results))

    print(f"Saved JSON: {json_path}")
    print(f"Saved MD: {md_path}")


if __name__ == "__main__":
    main()
