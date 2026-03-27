#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from scripts2.eval_video_kpi2 import (
    load_json,
    _load_image_ids,
    load_pred_any,
    group_by_video,
    build_pseudo_tracks,
    filter_pred,
    bbox_area_xywh,
    match_pred_to_gt_frame,
    first_run_start,
)


DEFAULT_METHODS = {
    "YOLO": {
        "fbd_per_video": "output/thesis2/experiments/2026-03-08_yolo_fbdsv_val/kpi/per_video.csv",
        "icct_per_video": "output/thesis2/experiments/2026-03-08_yolo_icct_dev/kpi/per_video.csv",
        "pred": "output/thesis2/experiments/2026-03-08_yolo_fbdsv_val/infer/pred_slim_yolo_val.jsonl.gz",
    },
    "AlwaysVerify": {
        "fbd_per_video": "output/thesis2/aligned_verify_baselines/experiments/2026-03-14_yolo_sahi_always_verify_fbdsv_val/kpi/per_video.csv",
        "icct_per_video": "output/thesis2/aligned_verify_baselines/experiments/2026-03-14_yolo_sahi_always_verify_icct_dev/kpi/per_video.csv",
        "pred": "output/thesis2/aligned_verify_baselines/experiments/2026-03-14_yolo_sahi_always_verify_fbdsv_val/infer/pred_slim_yolo_sahi_always_verify_exp0_val.jsonl.gz",
    },
    "KeyframeVerify": {
        "fbd_per_video": "output/thesis2/aligned_verify_baselines/experiments/2026-03-14_yolo_sahi_keyframe_verify_fbdsv_val/kpi/per_video.csv",
        "icct_per_video": "output/thesis2/aligned_verify_baselines/experiments/2026-03-14_yolo_sahi_keyframe_verify_icct_dev/kpi/per_video.csv",
        "pred": "output/thesis2/aligned_verify_baselines/experiments/2026-03-14_yolo_sahi_keyframe_verify_fbdsv_val/infer/pred_slim_yolo_sahi_keyframe_verify_exp0_n4_val.jsonl.gz",
    },
    "Ours (selected)": {
        "fbd_per_video": "output/thesis2/phase4_candidates/experiments/2026-03-11_v7_noconfirm_fbdsv_val/kpi/per_video.csv",
        "icct_per_video": "output/thesis2/phase4_candidates/experiments/2026-03-11_v7_noconfirm_icct_dev/kpi/per_video.csv",
        "pred": "output/thesis2/phase4_candidates/experiments/2026-03-11_v7_noconfirm_fbdsv_val/infer/pred_slim_temporal_sahi_v7_exp0_val.jsonl.gz",
    },
}


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
    boot: List[float] = []
    for _ in range(n_boot):
        sample = rng.choice(idx, size=len(idx), replace=True)
        boot.append(float(np.mean(np.asarray(vals)[sample])))
    return ci_from_samples(boot)


def build_small_tracks(gt_path: Path, pred_path: Path, image_ids_json: Path) -> Dict[str, Any]:
    gt = load_json(str(gt_path))
    pred = load_pred_any(str(pred_path))
    allowed_ids = set(_load_image_ids(str(image_ids_json)))

    images = [im for im in gt.get("images", []) if int(im.get("id", -1)) in allowed_ids]
    ann = [a for a in gt.get("annotations", []) if int(a.get("image_id", -1)) in allowed_ids]
    pred = [p for p in pred if int(p.get("image_id", -1)) in allowed_ids]

    by_video = group_by_video(images)
    cat_id = 0
    try:
        cats = gt.get("categories", [])
        if cats:
            cat_id = int(cats[0]["id"])
    except Exception:
        cat_id = 0
    if cat_id == 0:
        # bird target class in this project
        cat_id = 1

    gt_by_img: Dict[int, List[List[float]]] = {}
    for a in ann:
        if int(a.get("category_id", -1)) != cat_id:
            continue
        gt_by_img.setdefault(int(a["image_id"]), []).append(
            [float(a["bbox"][0]), float(a["bbox"][1]), float(a["bbox"][2]), float(a["bbox"][3])]
        )

    tracks = build_pseudo_tracks(
        by_video_frames=by_video,
        gt_by_img=gt_by_img,
        link_iou_thr=0.3,
        link_center_px=0.0,
        link_center_frac=0.5,
        use_center_link=0,
    )
    pred_by_img = filter_pred(pred, category_id=cat_id, score_thr=0.25)

    small_tracks = []
    for tr in tracks:
        small_frames = sum(1 for b in tr.gt_boxes if bbox_area_xywh(b) <= 1024.0)
        frac = float(small_frames) / float(len(tr.frames)) if tr.frames else 0.0
        if frac >= 0.5:
            small_tracks.append(tr)

    return {"tracks": small_tracks, "pred_by_img": pred_by_img}


def eval_small_track_sample(track_sample: List[Any], pred_by_img: Dict[int, List[List[float]]]) -> Dict[str, Optional[float]]:
    if not track_sample:
        return {"track_init_recall": None, "stable_ttd_p50": None}

    init_ok = 0
    stable_starts: List[int] = []
    for tr in track_sample:
        hit_seq: List[int] = []
        for idx, img_id in enumerate(tr.frames):
            gt_box = tr.gt_boxes[idx]
            preds = pred_by_img.get(img_id, [])
            hit = 1 if match_pred_to_gt_frame(
                gt_box=gt_box,
                pred_boxes=preds,
                match_iou_thr=0.5,
                center_match=0,
                center_px=0.0,
                center_frac=0.5,
            ) else 0
            hit_seq.append(hit)
        if any(hit_seq):
            init_ok += 1
        stable_start = first_run_start(hit_seq, n=5)
        if stable_start is not None:
            stable_starts.append(int(stable_start))

    n = len(track_sample)
    return {
        "track_init_recall": float(init_ok) / float(n),
        "stable_ttd_p50": quantile([float(x) for x in stable_starts], 0.50),
    }


def bootstrap_track_metrics(track_info: Dict[str, Any], n_boot: int, seed: int) -> Dict[str, Dict[str, Optional[float]]]:
    tracks = track_info["tracks"]
    pred_by_img = track_info["pred_by_img"]
    if not tracks:
        return {
            "track_init_recall": {"mean": None, "ci_low": None, "ci_high": None},
            "stable_ttd_p50": {"mean": None, "ci_low": None, "ci_high": None},
        }

    rng = np.random.default_rng(seed)
    idx = np.arange(len(tracks))
    init_vals: List[float] = []
    stable_vals: List[float] = []
    for _ in range(n_boot):
        sample_idx = rng.choice(idx, size=len(idx), replace=True)
        sample_tracks = [tracks[int(i)] for i in sample_idx]
        out = eval_small_track_sample(sample_tracks, pred_by_img)
        if out["track_init_recall"] is not None:
            init_vals.append(float(out["track_init_recall"]))
        if out["stable_ttd_p50"] is not None:
            stable_vals.append(float(out["stable_ttd_p50"]))

    return {
        "track_init_recall": ci_from_samples(init_vals),
        "stable_ttd_p50": ci_from_samples(stable_vals),
    }


def render_md(results: Dict[str, Any]) -> str:
    lines: List[str] = ["# Main Comparison Bootstrap CI", ""]
    lines.append("## FBD-SV Readiness Bootstrap")
    lines.append("")
    lines.append("| Method | Track-init recall mean | 95% CI | Stable TTD p50 mean | 95% CI |")
    lines.append("|---|---:|---:|---:|---:|")
    for method, rec in results["fbd_bootstrap"].items():
        tir = rec["track_init_recall"]
        sttd = rec["stable_ttd_p50"]
        lines.append(
            "| {m} | {a} | [{al}, {ah}] | {b} | [{bl}, {bh}] |".format(
                m=method,
                a=fmt(tir["mean"], 4),
                al=fmt(tir["ci_low"], 4),
                ah=fmt(tir["ci_high"], 4),
                b=fmt(sttd["mean"], 2),
                bl=fmt(sttd["ci_low"], 2),
                bh=fmt(sttd["ci_high"], 2),
            )
        )
    lines.append("")
    lines.append("## Per-Video Cost Bootstrap")
    lines.append("")
    lines.append("| Split | Method | Verifier on-ratio mean | 95% CI | Avg. tiles mean | 95% CI | Avg. latency mean | 95% CI | Empty FP rate mean | 95% CI |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for split in ("fbd", "icct"):
        for method, rec in results[f"{split}_video_bootstrap"].items():
            on = rec["sahi_on_ratio"]
            tiles = rec["avg_tiles"]
            lat = rec["avg_latency_ms"]
            fp = rec["empty_fp_rate"]
            lines.append(
                "| {split} | {m} | {a} | [{al}, {ah}] | {b} | [{bl}, {bh}] | {c} | [{cl}, {ch}] | {d} | [{dl}, {dh}] |".format(
                    split=split,
                    m=method,
                    a=fmt(on["mean"], 4),
                    al=fmt(on["ci_low"], 4),
                    ah=fmt(on["ci_high"], 4),
                    b=fmt(tiles["mean"], 4),
                    bl=fmt(tiles["ci_low"], 4),
                    bh=fmt(tiles["ci_high"], 4),
                    c=fmt(lat["mean"], 2),
                    cl=fmt(lat["ci_low"], 2),
                    ch=fmt(lat["ci_high"], 2),
                    d=fmt(fp["mean"], 4),
                    dl=fmt(fp["ci_low"], 4),
                    dh=fmt(fp["ci_high"], 4),
                )
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap CI for the main comparison.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gt-fbd", default="/mnt/ssd960/jm/sahi/data/FBD-SV-2024/annotations/fbdsv24_vid_val.json")
    parser.add_argument("--image-ids-json", default="output/thesis2/analysis/2026-03-11_far_small_fbdsv_val_tau002.json")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_fbd = Path(str(args.gt_fbd).strip())
    image_ids_json = repo / args.image_ids_json

    fbd_bootstrap: Dict[str, Any] = {}
    fbd_video_bootstrap: Dict[str, Any] = {}
    icct_video_bootstrap: Dict[str, Any] = {}

    for method, paths in DEFAULT_METHODS.items():
        track_info = build_small_tracks(gt_fbd, repo / paths["pred"], image_ids_json)
        fbd_bootstrap[method] = bootstrap_track_metrics(track_info, args.n_boot, args.seed)

        fbd_rows = read_csv_rows(repo / paths["fbd_per_video"])
        fbd_video_bootstrap[method] = {
            "sahi_on_ratio": bootstrap_video_metric(fbd_rows, "sahi_on_ratio", args.n_boot, args.seed),
            "avg_tiles": bootstrap_video_metric(fbd_rows, "avg_tiles", args.n_boot, args.seed),
            "avg_latency_ms": bootstrap_video_metric(fbd_rows, "avg_latency_ms", args.n_boot, args.seed),
            "empty_fp_rate": bootstrap_video_metric(fbd_rows, "empty_fp_rate", args.n_boot, args.seed),
        }

        icct_rows = read_csv_rows(repo / paths["icct_per_video"])
        icct_video_bootstrap[method] = {
            "sahi_on_ratio": bootstrap_video_metric(icct_rows, "sahi_on_ratio", args.n_boot, args.seed),
            "avg_tiles": bootstrap_video_metric(icct_rows, "avg_tiles", args.n_boot, args.seed),
            "avg_latency_ms": bootstrap_video_metric(icct_rows, "avg_latency_ms", args.n_boot, args.seed),
            "empty_fp_rate": bootstrap_video_metric(icct_rows, "empty_fp_rate", args.n_boot, args.seed),
        }

    out = {
        "settings": {
            "n_boot": int(args.n_boot),
            "seed": int(args.seed),
            "gt_fbd": str(gt_fbd),
            "image_ids_json": str(image_ids_json),
        },
        "fbd_bootstrap": fbd_bootstrap,
        "fbd_video_bootstrap": fbd_video_bootstrap,
        "icct_video_bootstrap": icct_video_bootstrap,
    }

    (out_dir / "main_comparison_bootstrap.json").write_text(json.dumps(out, indent=2))
    (out_dir / "main_comparison_bootstrap.md").write_text(render_md(out))
    print(f"Saved: {out_dir / 'main_comparison_bootstrap.json'}")
    print(f"Saved: {out_dir / 'main_comparison_bootstrap.md'}")


if __name__ == "__main__":
    main()
