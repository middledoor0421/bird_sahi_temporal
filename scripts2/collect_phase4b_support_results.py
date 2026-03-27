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


def effr_lookup(rows: List[Dict[str, str]], tau: str) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for row in rows:
        if row["tau"] != tau:
            continue
        out[row["method"]] = {
            "prevalence_on_empty": float(row["prevalence_on_empty"]),
            "prevalence_on_empty_given_sahi_on": float(row["prevalence_on_empty_given_sahi_on"]),
            "prevalence_on_empty_given_sahi_off": float(row["prevalence_on_empty_given_sahi_off"]),
        }
    return out


def stats_summary(path: Path) -> Dict[str, Any]:
    d = load_json(path)
    num_frames = float(d.get("num_frames", 0) or 0)
    do_sahi = float(d.get("do_sahi_frames", 0) or 0)
    sahi_ratio = (do_sahi / num_frames) if num_frames > 0 else 0.0
    avg_tiles = d.get("avg_tiles_selected", d.get("avg_tiles"))
    return {
        "fps": d.get("fps"),
        "sahi_on_ratio": sahi_ratio,
        "avg_tiles": avg_tiles,
    }


def build_rows(repo_root: Path) -> Dict[str, Any]:
    caltech = effr_lookup(
        load_csv(repo_root / "output/thesis2/target_effr/caltech_bird/step4/conditional_prevalence_fbdsv_eccv_val_target_only.csv"),
        "0.7",
    )
    coco = effr_lookup(
        load_csv(repo_root / "output/thesis2/target_effr/coco_bird_absent/step4/conditional_prevalence_fbdsv_val2017_target_only.csv"),
        "0.7",
    )

    wcs_rows: List[Dict[str, Any]] = []
    for split in ["target05", "target10", "target15", "target20"]:
        effr = effr_lookup(
            load_csv(repo_root / f"output/thesis2/target_effr/wcs_{split}/step4/conditional_prevalence_wcs_subset_{split}_target_only.csv"),
            "0.7",
        )
        stats_v7 = stats_summary(
            repo_root / f"output/thesis2/wcs_effr/experiments/2026-03-10_v7_wcs_{split}/infer/stats_temporal_sahi_v7_exp0_{split}.json"
        )
        stats_yolo = stats_summary(
            repo_root / f"output/thesis2/wcs_effr/experiments/2026-03-10_yolo_wcs_{split}/infer/stats_yolo_{split}.json"
        )
        wcs_rows.append(
            {
                "split": split,
                "yolo_effr_07": effr["yolo"]["prevalence_on_empty"],
                "v7_effr_07": effr["temporal_sahi_v7"]["prevalence_on_empty"],
                "v7_sahi_on_ratio": stats_v7["sahi_on_ratio"],
                "v7_avg_tiles": stats_v7["avg_tiles"],
                "v7_fps": stats_v7["fps"],
                "yolo_fps": stats_yolo["fps"],
            }
        )

    return {
        "caltech_tau07": caltech,
        "coco_tau07": coco,
        "wcs_tau07": wcs_rows,
        "notes": {
            "method_scope": "Support-set v7 runs are existing temporal_sahi_v7 reference runs, not rerun as v7_noconfirm.",
            "metric_scope": "Support-set source-of-truth is target-only Bird-EFFR step4.",
        },
    }


def make_table(headers: List[str], rows: List[List[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def write_markdown(data: Dict[str, Any], path: Path) -> None:
    cal = data["caltech_tau07"]
    coco = data["coco_tau07"]
    wcs = data["wcs_tau07"]
    lines = [
        "# Phase 4-B Support Evidence",
        "",
        "Support-set source-of-truth uses target-only Bird-EFFR Step 4 outputs.",
        "",
        "## Caltech ECCV Val",
        "",
        make_table(
            ["Method", "Bird-EFFR@0.7"],
            [
                ["YOLO", fmt(cal["yolo"]["prevalence_on_empty"])],
                ["Ours(v7 ref)", fmt(cal["temporal_sahi_v7"]["prevalence_on_empty"])],
                ["Full-SAHI", fmt(cal["full_sahi"]["prevalence_on_empty"])],
            ],
        ),
        "",
        "Read: target-only 기준으로는 세 방법 모두 `0.0`이다.",
        "",
        "## COCO Bird-Absent Val2017",
        "",
        make_table(
            ["Method", "Bird-EFFR@0.7"],
            [
                ["YOLO", fmt(coco["yolo"]["prevalence_on_empty"])],
                ["Ours(v7 ref)", fmt(coco["temporal_sahi_v7"]["prevalence_on_empty"])],
                ["Full-SAHI", fmt(coco["full_sahi"]["prevalence_on_empty"])],
            ],
        ),
        "",
        "Read: `YOLO`와 `Ours(v7 ref)`는 동일하고, `Full-SAHI`가 더 높다.",
        "",
        "## WCS Target-Ratio Subsets",
        "",
        make_table(
            ["Split", "YOLO EFFR@0.7", "Ours(v7 ref) EFFR@0.7", "Ours SAHI on ratio", "Ours avg tiles", "YOLO fps", "Ours fps"],
            [
                [
                    row["split"],
                    fmt(row["yolo_effr_07"]),
                    fmt(row["v7_effr_07"]),
                    fmt(row["v7_sahi_on_ratio"]),
                    fmt(row["v7_avg_tiles"]),
                    fmt(row["yolo_fps"], 3),
                    fmt(row["v7_fps"], 3),
                ]
                for row in wcs
            ],
        ),
        "",
        "Read: 현재 target-only `tau=0.7` 기준 WCS subsets에서는 `YOLO`와 `Ours(v7 ref)`가 모두 `0.0`이다.",
        "",
        "## Scope Note",
        "",
        "- support set의 `Ours`는 기존 `temporal_sahi_v7` reference run이다.",
        "- 즉, support set은 `Ours-final=v7_noconfirm` 직접 재실행 evidence가 아니라 architecture-level support evidence로 해석하는 것이 안전하다.",
        "- 본문 main claim은 여전히 `FBD-SV + ICCT` frozen 5-method 결과에 둔다.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect support-set evidence for Phase 4-B.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/thesis2/phase4b_support/2026-03-11_initial"),
        help="Output directory.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    out_dir = (repo_root / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir
    data = build_rows(repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phase4b_support_results.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    write_markdown(data, out_dir / "phase4b_support_tables.md")
    print(json.dumps({"out_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
