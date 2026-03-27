#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def pick_method(rows: List[Dict[str, str]], tau: str, method: str) -> float:
    for row in rows:
        if row["tau"] == tau and row["method"] == method:
            return float(row["prevalence_on_empty"])
    raise KeyError((tau, method))


def stats_ratio(path: Path) -> Dict[str, float]:
    data = load_json(path)
    num_frames = float(data.get("num_frames", 0) or 0)
    do_sahi = float(data.get("do_sahi_frames", 0) or 0)
    return {
        "fps": float(data.get("fps", 0.0) or 0.0),
        "sahi_on_ratio": (do_sahi / num_frames) if num_frames > 0 else 0.0,
        "avg_tiles": float(data.get("avg_tiles_selected", data.get("avg_tiles", 0.0)) or 0.0),
    }


def make_table(headers: List[str], rows: List[List[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect final support-set Ours-final results for Phase 4-B.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/thesis2/phase4b_support_final/2026-03-11"),
        help="Output directory.",
    )
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    out_dir = (repo / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cal_rows = load_csv(repo / "output/thesis2/phase4b_support_final/target_effr/caltech_bird/step4/conditional_prevalence_fbdsv_eccv_val_target_only.csv")
    coco_rows = load_csv(repo / "output/thesis2/phase4b_support_final/target_effr/coco_bird_absent/step4/conditional_prevalence_fbdsv_val2017_target_only.csv")

    wcs_splits = ["target05", "target10", "target15", "target20"]
    wcs = []
    for split in wcs_splits:
        rows = load_csv(repo / f"output/thesis2/phase4b_support_final/target_effr/wcs_{split}/step4/conditional_prevalence_wcs_subset_{split}_target_only.csv")
        stats = stats_ratio(repo / f"output/thesis2/phase4b_support_final/experiments/2026-03-11_v7_noconfirm_wcs_{split}/infer/stats_temporal_sahi_v7_exp0_{split}.json")
        wcs.append({
            "split": split,
            "effr_07": pick_method(rows, "0.7", "temporal_sahi_v7"),
            "effr_05": pick_method(rows, "0.5", "temporal_sahi_v7"),
            "sahi_on_ratio": stats["sahi_on_ratio"],
            "avg_tiles": stats["avg_tiles"],
            "fps": stats["fps"],
        })

    result = {
        "caltech": {
            "bird_effr_07": pick_method(cal_rows, "0.7", "temporal_sahi_v7"),
            "bird_effr_05": pick_method(cal_rows, "0.5", "temporal_sahi_v7"),
            "stats": stats_ratio(repo / "output/thesis2/phase4b_support_final/experiments/2026-03-11_v7_noconfirm_caltech_eccv_val/infer/stats_temporal_sahi_v7_exp0_eccv_val.json"),
        },
        "coco_bird_absent": {
            "bird_effr_07": pick_method(coco_rows, "0.7", "temporal_sahi_v7"),
            "bird_effr_05": pick_method(coco_rows, "0.5", "temporal_sahi_v7"),
            "stats": stats_ratio(repo / "output/thesis2/phase4b_support_final/experiments/2026-03-11_v7_noconfirm_coco_bird_absent_val2017/infer/stats_temporal_sahi_v7_exp0_val2017.json"),
        },
        "wcs": wcs,
    }

    md = [
        "# Phase 4-B Support Final Results",
        "",
        "These are direct `Ours-final (= v7_noconfirm)` support-set reruns.",
        "",
        "## Caltech ECCV Val",
        "",
        make_table(
            ["Metric", "Ours-final"],
            [
                ["Bird-EFFR@0.7", fmt(result["caltech"]["bird_effr_07"])],
                ["Bird-EFFR@0.5", fmt(result["caltech"]["bird_effr_05"])],
                ["SAHI on ratio", fmt(result["caltech"]["stats"]["sahi_on_ratio"])],
                ["Avg tiles", fmt(result["caltech"]["stats"]["avg_tiles"])],
                ["FPS", fmt(result["caltech"]["stats"]["fps"], 3)],
            ],
        ),
        "",
        "## COCO Bird-Absent Val2017",
        "",
        make_table(
            ["Metric", "Ours-final"],
            [
                ["Bird-EFFR@0.7", fmt(result["coco_bird_absent"]["bird_effr_07"])],
                ["Bird-EFFR@0.5", fmt(result["coco_bird_absent"]["bird_effr_05"])],
                ["SAHI on ratio", fmt(result["coco_bird_absent"]["stats"]["sahi_on_ratio"])],
                ["Avg tiles", fmt(result["coco_bird_absent"]["stats"]["avg_tiles"])],
                ["FPS", fmt(result["coco_bird_absent"]["stats"]["fps"], 3)],
            ],
        ),
        "",
        "## WCS Target-Ratio Subsets",
        "",
        make_table(
            ["Split", "Bird-EFFR@0.7", "Bird-EFFR@0.5", "SAHI on ratio", "Avg tiles", "FPS"],
            [
                [row["split"], fmt(row["effr_07"]), fmt(row["effr_05"]), fmt(row["sahi_on_ratio"]), fmt(row["avg_tiles"]), fmt(row["fps"], 3)]
                for row in result["wcs"]
            ],
        ),
    ]
    (out_dir / "phase4b_support_final_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out_dir / "phase4b_support_final_tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
