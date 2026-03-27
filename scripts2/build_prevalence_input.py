#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import gzip
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

from scripts2.gt_utils import resolve_target_category_id
from scripts2.pred_slim import load_pred_any


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def load_jsonl_gz(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def discover_single_file(root: str, prefix: str, suffix: str) -> Optional[str]:
    if not os.path.isdir(root):
        return None
    hits = []
    for name in os.listdir(root):
        if name.startswith(prefix) and name.endswith(suffix):
            hits.append(os.path.join(root, name))
    hits.sort()
    if not hits:
        return None
    if len(hits) > 1:
        # Keep the deterministic first hit; current run folders are expected to hold one.
        return hits[0]
    return hits[0]


def discover_cost_path(run_dir: str) -> Optional[str]:
    candidates = [
        os.path.join(run_dir, "infer", "cost_per_frame.jsonl.gz"),
        os.path.join(run_dir, "kpi", "cost_per_frame.jsonl.gz"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def build_pred_index(pred_path: str, target_category_id: Optional[int] = None) -> Dict[int, List[Dict[str, Any]]]:
    pred_by_img: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for p in load_pred_any(pred_path):
        if not isinstance(p, dict):
            continue
        if target_category_id is not None:
            try:
                cid = int(p.get("category_id", -1))
            except Exception:
                continue
            if cid != int(target_category_id):
                continue
        try:
            image_id = int(p.get("image_id"))
        except Exception:
            continue
        pred_by_img[image_id].append({
            "category_id": p.get("category_id"),
            "bbox": p.get("bbox"),
            "score": float(p.get("score", 0.0)),
        })
    for image_id in pred_by_img:
        pred_by_img[image_id].sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return pred_by_img


def build_cost_index(cost_path: str) -> Dict[int, Dict[str, Any]]:
    cost_by_img: Dict[int, Dict[str, Any]] = {}
    for row in load_jsonl_gz(cost_path):
        try:
            image_id = int(row.get("image_id"))
        except Exception:
            continue
        cost_by_img[image_id] = row
    return cost_by_img


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join strict-empty mask with predictions and per-frame achieved cost for EFFR Step 2."
    )
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--empty-mask", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--pred-path", type=str, default=None)
    parser.add_argument("--cost-path", type=str, default=None)
    parser.add_argument("--category-mode", type=str, default="all_category",
                        choices=["all_category", "target_only"])
    parser.add_argument("--target-category-id", type=int, default=None)
    parser.add_argument("--target-category-name", type=str, default=None)
    parser.add_argument("--gt-json", type=str, default=None)
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    infer_dir = os.path.join(run_dir, "infer")
    run_config_path = os.path.join(infer_dir, "run_config.json")
    if not os.path.isfile(run_config_path):
        raise FileNotFoundError("run_config.json not found: {}".format(run_config_path))

    run_config = load_json(run_config_path)
    run_meta = run_config.get("run_meta", {})
    args_meta = run_config.get("args", {})
    method = str(run_meta.get("method", args_meta.get("method", "")))
    dataset = str(run_meta.get("dataset", args_meta.get("datasets", "")))
    split = str(run_meta.get("split", args_meta.get("split", "")))
    target_category_id = None
    if str(args.category_mode) == "target_only":
        if args.target_category_id is not None:
            target_category_id = int(args.target_category_id)
        elif run_meta.get("target_category_id", None) is not None:
            try:
                target_category_id = int(run_meta.get("target_category_id"))
            except Exception:
                target_category_id = None
        else:
            gt_json = args.gt_json
            if not gt_json:
                data_root = str(run_meta.get("data_root", ""))
                split_name = str(run_meta.get("split", split))
                maybe_gt = os.path.join(data_root, "annotations", "fbdsv24_vid_{}.json".format(split_name))
                if os.path.isfile(maybe_gt):
                    gt_json = maybe_gt
            if gt_json:
                target_category_id = resolve_target_category_id(
                    gt_json,
                    target_category_id=args.target_category_id,
                    target_category_name=args.target_category_name,
                )
        if target_category_id is None:
            raise ValueError(
                "target_only mode requires a target category. "
                "Provide --target-category-id/--target-category-name or ensure run_config has run_meta.target_category_id."
            )

    pred_path = args.pred_path
    if not pred_path:
        pred_path = discover_single_file(infer_dir, "pred_slim_", ".jsonl.gz")
    if not pred_path or not os.path.isfile(pred_path):
        raise FileNotFoundError("Could not resolve pred_slim jsonl.gz under: {}".format(infer_dir))

    cost_path = args.cost_path
    if not cost_path:
        cost_path = discover_cost_path(run_dir)
    if not cost_path or not os.path.isfile(cost_path):
        raise FileNotFoundError("Could not resolve cost_per_frame.jsonl.gz under: {}".format(run_dir))

    ensure_dir(args.out_dir)
    out_jsonl = os.path.join(args.out_dir, "prevalence_input.jsonl.gz")
    out_summary = os.path.join(args.out_dir, "prevalence_input_summary.json")

    mask_rows = load_jsonl_gz(args.empty_mask)
    pred_by_img = build_pred_index(pred_path, target_category_id=target_category_id)
    cost_by_img = build_cost_index(cost_path)

    num_rows = 0
    num_empty = 0
    num_sahi_on = 0
    num_empty_and_sahi_on = 0
    pred_count_all = 0
    pred_count_empty = 0

    with gzip.open(out_jsonl, "wt", encoding="utf-8") as f:
        for row in mask_rows:
            try:
                image_id = int(row.get("image_id"))
            except Exception:
                continue

            preds = pred_by_img.get(image_id, [])
            cost = cost_by_img.get(image_id, {})
            sahi_on = int(cost.get("sahi_on", 0))
            tiles_used = float(cost.get("tiles_used", 0.0))
            latency_ms = float(cost.get("latency_ms", 0.0))
            pred_count = int(len(preds))
            top_score = float(preds[0]["score"]) if preds else 0.0

            rec = {
                "dataset": str(dataset),
                "split": str(split),
                "method": str(method),
                "category_mode": str(args.category_mode),
                "target_category_id": int(target_category_id) if target_category_id is not None else None,
                "video_id": row.get("video_id"),
                "frame_idx": int(row.get("frame_idx", 0)),
                "image_id": image_id,
                "is_empty": bool(row.get("is_empty", False)),
                "gt_num_boxes": int(row.get("gt_num_boxes", 0)),
                "sahi_executed": int(sahi_on),
                "tiles_used": float(tiles_used),
                "latency_ms": float(latency_ms),
                "pred_count": int(pred_count),
                "top_score": float(top_score),
                "preds": preds,
            }
            f.write(json.dumps(rec) + "\n")

            num_rows += 1
            pred_count_all += pred_count
            if rec["is_empty"]:
                num_empty += 1
                pred_count_empty += pred_count
            if sahi_on == 1:
                num_sahi_on += 1
                if rec["is_empty"]:
                    num_empty_and_sahi_on += 1

    summary = {
        "dataset": str(dataset),
        "split": str(split),
        "method": str(method),
        "category_mode": str(args.category_mode),
        "target_category_id": int(target_category_id) if target_category_id is not None else None,
        "run_dir": run_dir,
        "empty_mask": os.path.abspath(args.empty_mask),
        "pred_path": os.path.abspath(pred_path),
        "cost_path": os.path.abspath(cost_path),
        "num_rows": int(num_rows),
        "num_empty_rows": int(num_empty),
        "num_sahi_on_rows": int(num_sahi_on),
        "num_empty_and_sahi_on_rows": int(num_empty_and_sahi_on),
        "avg_pred_count_per_frame": float(pred_count_all) / float(num_rows) if num_rows > 0 else 0.0,
        "avg_pred_count_per_empty_frame": float(pred_count_empty) / float(num_empty) if num_empty > 0 else 0.0,
        "out_jsonl_gz": os.path.basename(out_jsonl),
    }

    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved:", out_jsonl)
    print("Saved:", out_summary)


if __name__ == "__main__":
    main()
