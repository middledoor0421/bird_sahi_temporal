#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


PAPER_LABELS = {
    "yolo": "YOLO",
    "full_sahi": "Naive-SAHI",
    "keyframe_sahi": "Periodic-SAHI",
    "yolo_sahi_always_verify": "YOLO+AlwaysVerify-SAHI",
    "yolo_sahi_keyframe_verify": "YOLO+KeyframeVerify-SAHI",
    "ours_reference": "Ours (reference)",
    "ours_selected": "Ours (selected)",
    "ours_selected_cropverify": "Ours (selected) + CropVerifier",
}


METHOD_SPECS = [
    {
        "method_id": "yolo",
        "fbd_summary": "output/thesis2/experiments/2026-03-08_yolo_fbdsv_val/kpi/summary.json",
        "icct_summary": "output/thesis2/experiments/2026-03-08_yolo_icct_dev/kpi/summary.json",
        "readiness": "output/thesis2/phase4_candidates/experiments/readiness_yolo_fbdsv_val.json",
    },
    {
        "method_id": "full_sahi",
        "fbd_summary": "output/thesis2/experiments/2026-03-08_fullsahi_fbdsv_val/kpi/summary.json",
        "icct_summary": "output/thesis2/experiments/2026-03-08_fullsahi_icct_dev/kpi/summary.json",
        "readiness": "output/thesis2/phase4_candidates/experiments/readiness_fullsahi_fbdsv_val.json",
    },
    {
        "method_id": "keyframe_sahi",
        "fbd_summary": "output/thesis2/experiments/2026-03-08_keyframe4_fbdsv_val/kpi/summary.json",
        "icct_summary": "output/thesis2/experiments/2026-03-08_keyframe4_icct_dev/kpi/summary.json",
        "readiness": "output/thesis2/phase4_candidates/experiments/readiness_keyframe4_fbdsv_val.json",
    },
    {
        "method_id": "ours_reference",
        "fbd_summary": "output/thesis2/experiments/2026-03-08_v7exp0_fbdsv_val/kpi/summary.json",
        "icct_summary": "output/thesis2/experiments/2026-03-08_v7exp0_icct_dev/kpi/summary.json",
        "readiness": "output/thesis2/phase4_candidates/experiments/readiness_v7base_fbdsv_val.json",
    },
    {
        "method_id": "ours_selected",
        "fbd_summary": "output/thesis2/phase4_candidates/experiments/2026-03-11_v7_noconfirm_fbdsv_val/kpi/summary.json",
        "icct_summary": "output/thesis2/phase4_candidates/experiments/2026-03-11_v7_noconfirm_icct_dev/kpi/summary.json",
        "readiness": "output/thesis2/phase4_candidates/experiments/readiness_v7_noconfirm_fbdsv_val.json",
    },
    {
        "method_id": "ours_selected_cropverify",
        "fbd_summary": "output/thesis2/phase4_candidates/experiments/2026-03-12_v7_cropverify_fbdsv_val/kpi/summary.json",
        "icct_summary": "output/thesis2/phase4_candidates/experiments/2026-03-12_v7_cropverify_icct_dev/kpi/summary.json",
        "readiness": "output/thesis2/phase4_candidates/experiments/readiness_v7_cropverify_fbdsv_val.json",
    },
]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _verify_ratio(cost: Dict[str, Any]) -> Any:
    if "verifier_on_ratio" in cost:
        return cost.get("verifier_on_ratio")
    return cost.get("sahi_on_ratio")


def _avg_verify_regions(cost: Dict[str, Any]) -> Any:
    if isinstance(cost.get("avg_verifier_regions"), dict):
        return cost.get("avg_verifier_regions", {}).get("mean")
    return cost.get("avg_tiles_per_frame", {}).get("mean")


def _empty_verify_ratio(cost: Dict[str, Any]) -> Any:
    ewc = cost.get("empty_wasted_compute", {})
    if "empty_verifier_on_ratio" in ewc:
        return ewc.get("empty_verifier_on_ratio")
    return ewc.get("empty_sahi_on_ratio")


def read_fbd_metrics(path: Path, tau: str) -> Dict[str, Any]:
    data = load_json(path)
    tau_metrics = data.get("small_tau", {}).get(tau, {})
    delay = tau_metrics.get("delay", {})
    cost = data.get("cost", {})
    return {
        "recall_tau002": tau_metrics.get("Recall"),
        "ap50_tau002": tau_metrics.get("AP50"),
        "delay_mean": delay.get("mean"),
        "delay_p50": delay.get("p50"),
        "verify_on_ratio_fbd": _verify_ratio(cost),
        "avg_verify_regions_fbd": _avg_verify_regions(cost),
    }


def read_icct_metrics(path: Path) -> Dict[str, Any]:
    data = load_json(path)
    safety = data.get("safety", {})
    cost = data.get("cost", {})
    effr = safety.get("EFFR", {})
    return {
        "bird_effr_05": effr.get("0.5"),
        "bird_effr_07": effr.get("0.7"),
        "verify_on_ratio_icct": _verify_ratio(cost),
        "avg_verify_regions_icct": _avg_verify_regions(cost),
        "empty_verify_on_ratio": _empty_verify_ratio(cost),
    }


def read_readiness_metrics(path: Path) -> Dict[str, Any]:
    data = load_json(path)
    overall = data.get("kpi", {}).get("overall", {})
    return {
        "episode_capture_rate": overall.get("track_init_recall"),
        "stable_ttd_p50": overall.get("stable_ttd_p50"),
        "stable_ttd_p90": overall.get("stable_ttd_p90"),
        "sustained_lock_rate": overall.get("sustained_lock_rate"),
        "deterrence_ready_rate": overall.get("deterrence_ready_rate"),
    }


def build_rows(repo_root: Path, tau: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for spec in METHOD_SPECS:
        method_id = str(spec["method_id"])
        row: Dict[str, Any] = {"method": PAPER_LABELS.get(method_id, method_id)}
        row.update(read_fbd_metrics(repo_root / spec["fbd_summary"], tau))
        row.update(read_readiness_metrics(repo_root / spec["readiness"]))
        row.update(read_icct_metrics(repo_root / spec["icct_summary"]))
        rows.append(row)
    return rows


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    columns = [
        "method",
        "recall_tau002",
        "ap50_tau002",
        "delay_mean",
        "delay_p50",
        "episode_capture_rate",
        "stable_ttd_p50",
        "stable_ttd_p90",
        "sustained_lock_rate",
        "deterrence_ready_rate",
        "bird_effr_05",
        "bird_effr_07",
        "verify_on_ratio_fbd",
        "avg_verify_regions_fbd",
        "verify_on_ratio_icct",
        "avg_verify_regions_icct",
        "empty_verify_on_ratio",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col) for col in columns})


def make_table(headers: List[str], rows: List[List[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def write_markdown(rows: List[Dict[str, Any]], path: Path) -> None:
    discovery_rows = [
        [
            row["method"],
            fmt(row["recall_tau002"]),
            fmt(row["ap50_tau002"]),
            fmt(row["delay_mean"], 2),
            fmt(row["delay_p50"], 1),
            fmt(row["episode_capture_rate"]),
        ]
        for row in rows
    ]
    readiness_rows = [
        [
            row["method"],
            fmt(row["stable_ttd_p50"], 2),
            fmt(row["stable_ttd_p90"], 2),
            fmt(row["sustained_lock_rate"]),
            fmt(row["deterrence_ready_rate"]),
        ]
        for row in rows
    ]
    risk_rows = [
        [
            row["method"],
            fmt(row["bird_effr_07"]),
            fmt(row["bird_effr_05"]),
            fmt(row["verify_on_ratio_icct"]),
            fmt(row["avg_verify_regions_icct"]),
            fmt(row["empty_verify_on_ratio"]),
        ]
        for row in rows
    ]

    body = [
        "# Phase 4-B Initial Tables",
        "",
        "Generated from the frozen comparison runs on 2026-03-11.",
        "",
        "## Discovery",
        "",
        make_table(
            ["Method", "Recall@tau0.02", "AP50@tau0.02", "Delay mean", "Delay p50", "Episode capture"],
            discovery_rows,
        ),
        "",
        "## Readiness",
        "",
        make_table(
            ["Method", "Stable delay p50", "Stable delay p90", "Sustained lock", "Deterrence-ready"],
            readiness_rows,
        ),
        "",
        "## Risk / Cost",
        "",
        make_table(
            ["Method", "Bird-EFFR@0.7", "Bird-EFFR@0.5", "Verify on ratio", "Avg verify regions/frame", "Empty verify on ratio"],
            risk_rows,
        ),
        "",
        "## Note",
        "",
        "- `Naive-SAHI` and `Periodic-SAHI` are sliced-inference reference schedules, not architecture-aligned YOLO+Verify baselines.",
        "- `Ours (selected)` is mapped to `v7_noconfirm`.",
        "- Readiness metrics are from `eval_video_kpi2.py` on the FBD-SV far-small image-id subset.",
        "- `Empty verify on ratio` is a cost-side diagnostic from ICCT summary cost fields.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect initial Phase 4-B tables for the frozen 5-method set.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/thesis2/phase4b/2026-03-11_initial"),
        help="Output directory.",
    )
    parser.add_argument("--tau", type=str, default="0.02", help="small_tau key for discovery metrics.")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    out_dir = (repo_root / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir
    rows = build_rows(repo_root, args.tau)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phase4b_initial_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_csv(rows, out_dir / "phase4b_initial_results.csv")
    write_markdown(rows, out_dir / "phase4b_initial_tables.md")
    print(json.dumps({"out_dir": str(out_dir), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
