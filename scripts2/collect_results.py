#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

def _safe_get(d: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _find_summary_jsons(root: Path) -> List[Path]:
    # Typical layout: <run_dir>/results/summary.json
    # We also accept summary.json anywhere under root.
    cands = list(root.rglob("results/summary.json"))
    if cands:
        return sorted(cands)
    return sorted(root.rglob("summary.json"))

def _load_json(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def _flatten_summary(summary: Dict[str, Any], tau_main: str, safety_tau: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {}

    # Basic identifiers
    row["run_dir"] = str(summary.get("run_dir", ""))
    row["method"] = str(summary.get("method", summary.get("method_name", "")))
    row["split"] = str(summary.get("split", ""))

    # Quality (main)
    quality = summary.get("quality", {})
    row["mAP"] = quality.get("mAP", None)
    row["AP50"] = quality.get("AP50", None)
    row["Recall"] = quality.get("Recall", None)

    # Empty FP (legacy)
    row["empty_fp_rate"] = quality.get("empty_fp_rate", summary.get("empty_fp_rate", None))
    row["num_tp_frames"] = quality.get("tp_frames", summary.get("num_tp_frames", None))
    row["num_fp_frames"] = quality.get("fp_frames", summary.get("num_fp_frames", None))
    row["num_fn_frames"] = quality.get("fn_frames", summary.get("num_fn_frames", None))

    # Cost
    cost = summary.get("cost", {})
    row["num_pred_kept"] = cost.get("num_pred_kept", None)
    row["sahi_on_ratio"] = cost.get("sahi_on_ratio", None)
    row["avg_tiles_mean"] = cost.get("avg_tiles_mean", None)
    row["avg_tiles_per_sahi_frame"] = cost.get("avg_tiles_per_sahi_frame", None)
    row["latency_ms_mean"] = _safe_get(cost, ["latency_ms", "mean"], None)
    row["fps"] = cost.get("fps", None)

    # Sanity
    sanity = summary.get("sanity", {})
    row["bbox_oob_rate"] = sanity.get("bbox_oob_rate", None)
    row["num_pred_raw"] = sanity.get("num_pred_raw", None)

    # Safety (EFFR/EFPD) for a specific tau
    safety = summary.get("safety", {})
    # safety["EFFR"] and ["EFPD"] may be dict keyed by tau string or float-like
    effr = safety.get("EFFR", {})
    efpd = safety.get("EFPD", {})
    # normalize key
    tau_key_candidates = [safety_tau, str(float(safety_tau))]
    effr_val = None
    efpd_val = None
    if isinstance(effr, dict):
        for k in tau_key_candidates:
            if k in effr:
                effr_val = effr[k]
                break
    if isinstance(efpd, dict):
        for k in tau_key_candidates:
            if k in efpd:
                efpd_val = efpd[k]
                break
    row[f"EFFR@{safety_tau}"] = effr_val
    row[f"EFPD@{safety_tau}"] = efpd_val

    # Confirmation (Step2)
    conf = summary.get("confirm", {})
    row["confirm_frames"] = conf.get("confirm_frames", None)
    row["unconfirmed_frames"] = conf.get("unconfirmed_frames", None)
    row["newly_confirmed_total"] = conf.get("newly_confirmed_total", None)
    row["ttc_mean"] = conf.get("ttc_mean", None)
    row["ttc_p90"] = conf.get("ttc_p90", None)

    # Budget (Step3)
    budget = summary.get("budget", {})
    row["pos_budget_explore"] = budget.get("pos_budget_explore", None)
    row["empty_budget_skip"] = budget.get("empty_budget_skip", None)
    row["verify_burst"] = budget.get("verify_burst", None)
    row["frames_with_pos_keys"] = budget.get("frames_with_pos_keys", None)

    # Ablation (Step5)
    ab = summary.get("ablation", {})
    row["use_confirmation"] = ab.get("use_confirmation", None)
    row["use_dual_budget"] = ab.get("use_dual_budget", None)
    row["use_targeted_verify"] = ab.get("use_targeted_verify", None)

    # Small object tau-specific quality if present
    # Some pipelines nest quality by small_tau: summary["quality"]["0.01"]["mAP"] etc.
    if isinstance(quality, dict) and tau_main in quality and isinstance(quality[tau_main], dict):
        q = quality[tau_main]
        row[f"mAP@tau{tau_main}"] = q.get("mAP", None)
        row[f"AP50@tau{tau_main}"] = q.get("AP50", None)
        row[f"Recall@tau{tau_main}"] = q.get("Recall", None)

    return row

def _write_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    # Determine columns: stable first, then extras
    base_cols = [
        "method","split","mAP","AP50","Recall",
        "empty_fp_rate","num_tp_frames","num_fp_frames","num_fn_frames",
        "num_pred_kept","sahi_on_ratio","avg_tiles_mean","avg_tiles_per_sahi_frame",
        "latency_ms_mean","fps",
        "bbox_oob_rate","num_pred_raw",
    ]
    extra_cols = []
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    for k in sorted(all_keys):
        if k not in base_cols and k not in ("run_dir",):
            extra_cols.append(k)
    cols = ["run_dir"] + base_cols + extra_cols

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, None) for c in cols})

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Root directory that contains run folders")
    ap.add_argument("--out-dir", type=str, required=True, help="Output directory for tables")
    ap.add_argument("--tau-main", type=str, default="0.01", help="Small object tau key used in summary.json (if nested)")
    ap.add_argument("--safety-tau", type=str, default="0.5", help="Tau value for EFFR/EFPD to export")
    ap.add_argument("--out-name", type=str, default="results_table.csv")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_csv = out_dir / args.out_name

    summary_paths = _find_summary_jsons(root)
    rows: List[Dict[str, Any]] = []
    for sp in summary_paths:
        try:
            s = _load_json(sp)
        except Exception:
            continue
        # attach run_dir/method if missing
        if "run_dir" not in s:
            s["run_dir"] = str(sp.parent.parent if sp.name == "summary.json" and sp.parent.name == "results" else sp.parent)
        rows.append(_flatten_summary(s, tau_main=args.tau_main, safety_tau=args.safety_tau))

    _write_csv(rows, out_csv)
    print(f"Wrote {len(rows)} rows to {out_csv}")

if __name__ == "__main__":
    main()
