#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import inspect
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from datasets.fbdsv import FbdsvDataset
from datasets.icct_sparse20 import IcctSparse20Dataset
from datasets.wcs_subset import WcsSubsetDataset

from methods.yolo_runner import YoloRunner
from methods.full_sahi_runner import SahiFullRunner
from methods.keyframe_sahi_runner import KeyframeSahiRunner
from methods.temporal_sahi_v6_runner import SahiTemporalV6Runner
from methods.temporal_sahi_v7_runner import SahiTemporalV7Runner
from scripts2.gt_utils import resolve_target_category_id
from scripts2.pred_slim import slim_predictions, save_jsonl_gz

METHOD_REGISTRY = {
    "yolo": YoloRunner,
    "full_sahi": SahiFullRunner,
    "keyframe_sahi": KeyframeSahiRunner,
    "temporal_sahi_v6": SahiTemporalV6Runner,
    "temporal_sahi_v7": SahiTemporalV7Runner,
    "yolo_sahi_always_verify": SahiTemporalV7Runner,
    "yolo_sahi_keyframe_verify": SahiTemporalV7Runner,
}

V7_FAMILY_METHODS = {
    "temporal_sahi_v7",
    "yolo_sahi_always_verify",
    "yolo_sahi_keyframe_verify",
}

KEYFRAME_METHODS = {
    "keyframe_sahi",
    "yolo_sahi_keyframe_verify",
}


def is_v7_family_method(method: str) -> bool:
    return str(method).lower() in V7_FAMILY_METHODS


def is_keyframe_method(method: str) -> bool:
    return str(method).lower() in KEYFRAME_METHODS


def resolve_verify_schedule(method: str, requested_mode: str) -> str:
    m = str(method).lower()
    if m == "yolo_sahi_always_verify":
        return "always"
    if m == "yolo_sahi_keyframe_verify":
        return "keyframe"
    return str(requested_mode).lower().strip()


def resolve_method_architecture(method: str) -> Optional[str]:
    m = str(method).lower()
    if m == "yolo":
        return "detector_only"
    if m == "full_sahi":
        return "sliced_inference"
    if m == "keyframe_sahi":
        return "periodic_sliced_inference"
    if m == "temporal_sahi_v6":
        return "temporal_sahi_policy"
    if is_v7_family_method(m):
        return "yolo_plus_local_verifier"
    return None


def build_dataset(dataset_name: str, data_root: str, split: str):
    dataset_name = dataset_name.lower()
    if dataset_name == "fbdsv":
        return FbdsvDataset(data_root=data_root, split=split)
    if dataset_name in ["icct", "icct_sparse20", "icct_sparse20_frac025"]:
        return IcctSparse20Dataset(data_root=data_root, split=split)
    if dataset_name in ["wcs", "wcs_subset"]:
        return WcsSubsetDataset(data_root=data_root, split=split)
    raise ValueError("Unsupported dataset: {}".format(dataset_name))


class LimitedDataset:
    # A thin wrapper to stop dataset iteration early without modifying the underlying dataset.
    def __init__(self, ds, max_frames: int):
        self._ds = ds
        self._max_frames = int(max_frames)

    def __len__(self):
        try:
            return min(len(self._ds), self._max_frames)
        except Exception:
            return self._max_frames

    def __getattr__(self, name):
        return getattr(self._ds, name)

    def iter_frames(self):
        n = 0
        for item in self._ds.iter_frames():
            yield item
            n += 1
            if self._max_frames > 0 and n >= self._max_frames:
                break


def load_class_mapping(path: Optional[str]) -> Optional[Dict[int, int]]:
    if path is None:
        return None
    with open(path, "r") as f:
        raw = json.load(f)
    return {int(k): int(v) for k, v in raw.items()}


def filter_kwargs_for_class(cls: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(cls.__init__)
    allowed = set(sig.parameters.keys())
    allowed.discard("self")
    return {k: v for k, v in kwargs.items() if k in allowed}


def default_output_paths(out_dir: str, method: str, split: str, exp_id: Optional[int], key_interval: Optional[int]) -> \
        Tuple[str, str, str]:
    os.makedirs(out_dir, exist_ok=True)
    tag = method
    if exp_id is not None:
        tag += "_exp{}".format(int(exp_id))
    if key_interval is not None and is_keyframe_method(method):
        tag += "_n{}".format(int(key_interval))
    out_pred_slim = os.path.join(out_dir, "pred_slim_{}_{}.jsonl.gz".format(tag, split))
    out_pred_raw = os.path.join(out_dir, "pred_raw_{}_{}.jsonl.gz".format(tag, split))
    out_stats = os.path.join(out_dir, "stats_{}_{}.json".format(tag, split))
    return out_pred_slim, out_pred_raw, out_stats


def save_stats(path: str, stats: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)


def available_splits_in_annotations(ann_dir: str) -> List[str]:
    """
    Discover available split names by scanning fbdsv24_vid_*.json in annotations dir.
    Excludes *_sequences.json, *_fixed.json, *.bak.
    """
    if not os.path.isdir(ann_dir):
        return []
    out = set()
    pat = re.compile(r"^fbdsv24_vid_(.+)\.json$")
    for fn in os.listdir(ann_dir):
        m = pat.match(fn)
        if not m:
            continue
        name = m.group(1)
        if name.endswith("_sequences"):
            continue
        if name.endswith("_fixed"):
            continue
        if fn.endswith(".bak"):
            continue
        out.add(name)
    return sorted(out)


def resolve_gt_and_seq_paths(data_root: str, split: str, dataset_name: str = "fbdsv") -> Tuple[str, str]:
    """
    Resolve GT/sequence file paths for a given split under data_root/annotations.
    This keeps dataset folder structure unchanged while making errors actionable.
    """
    ann_dir = os.path.join(data_root, "annotations")
    gt = os.path.join(ann_dir, "fbdsv24_vid_{}.json".format(split))
    seq = os.path.join(ann_dir, "fbdsv24_vid_{}_sequences.json".format(split))

    if os.path.exists(gt) and os.path.exists(seq):
        return gt, seq

    avail = available_splits_in_annotations(ann_dir)
    msg = [
        "Split files not found under: {}".format(ann_dir),
        "Requested split='{}' -> expected:".format(split),
        "  - {}".format(gt),
        "  - {}".format(seq),
        "Available splits: {}".format(avail if avail else "[]"),
        "Hint: FBD-SV commonly uses train/val/total, while ICCT-style sparse sets commonly use dev/holdout.",
        "Resolved dataset name: {}".format(dataset_name),
    ]
    raise FileNotFoundError("\n".join(msg))


def build_runner_from_args(method: str, args: argparse.Namespace, class_mapping: Optional[Dict[int, int]],
                           v6_target_ids: str):
    method = method.lower()
    if method not in METHOD_REGISTRY:
        raise ValueError("Unsupported method: {}".format(method))

    cls = METHOD_REGISTRY[method]

    common_kwargs: Dict[str, Any] = {
        "weights": args.weights,
        "device": args.device,
        "img_size": args.img_size,
        "conf_thres": args.conf_thres,
        "class_mapping": class_mapping,
        "cost_per_frame_path": args.cost_per_frame,
    }

    sahi_kwargs: Dict[str, Any] = {
        "slice_height": args.slice_height,
        "slice_width": args.slice_width,
        "overlap_height_ratio": args.overlap_height_ratio,
        "overlap_width_ratio": args.overlap_width_ratio,
        "nms_iou_threshold": args.nms_iou_threshold,
    }

    keyframe_kwargs: Dict[str, Any] = {
        "key_interval": args.key_interval,
    }

    v6_log_path = args.v6_log_path
    if v6_log_path is None or str(v6_log_path).strip() == "":
        v6_log_path = "/dev/null"

    v6_kwargs: Dict[str, Any] = {
        "merge_mode": args.v6_merge_mode,
        "log_path": v6_log_path,
        "v6_target_ids": v6_target_ids,
        "v6_K_mid": args.v6_k_mid,
        "v6_full_ttl": args.v6_full_ttl,
        "exp_id": int(args.v6_exp),
        "cost_per_frame_path": args.v6_cost_per_frame,
    }
    v7_log_path = args.v7_log_path
    if v7_log_path is None or str(v7_log_path).strip() == "":
        v7_log_path = "/dev/null"

    v7_cost_per_frame = args.v7_cost_per_frame
    if v7_cost_per_frame is None or str(v7_cost_per_frame).strip() == "":
        v7_cost_per_frame = "/dev/null"

    v7_kwargs: Dict[str, Any] = {
        "log_path": v7_log_path,
        "cost_per_frame_path": v7_cost_per_frame,
        "exp_id": int(args.v7_exp),
        "v7_confirm_iou": float(args.v7_confirm_iou),
        "v7_confirm_len": int(args.v7_confirm_len),
        "v7_confirm_max_age": int(args.v7_confirm_max_age),
        "v7_pp_do_clamp": bool(int(args.v7_pp_do_clamp)),
        "v7_pp_do_nms": bool(int(args.v7_pp_do_nms)),
        "v7_pp_nms_iou": float(args.v7_pp_nms_iou),
        "v7_pp_class_agnostic_nms": bool(int(args.v7_pp_class_agnostic_nms)),
        "v7_pp_max_det": int(args.v7_pp_max_det),
        "v7_pp_score_thr": float(args.v7_pp_score_thr),
        "use_confirmation": bool(int(args.use_confirmation)),
        "use_dual_budget": bool(int(args.use_dual_budget)),
        "use_targeted_verify": bool(int(args.use_targeted_verify)),
        "verifier_backend": str(args.verifier_backend),
        "crop_margin_ratio": float(args.crop_margin_ratio),
        "crop_input_size": int(args.crop_input_size),
        "empty_explore_rate": float(args.empty_explore_rate),
        "empty_explore_cap": float(args.empty_explore_cap),
        "empty_explore_cost": float(args.empty_explore_cost),
        "pos_explore_rate": float(args.pos_explore_rate),
        "pos_explore_cap": float(args.pos_explore_cap),
        "pos_explore_cost": float(args.pos_explore_cost),
        "pos_motion_local_peak_thr": float(args.pos_motion_local_peak_thr),
        "pos_diff_ratio_thr": float(args.pos_diff_ratio_thr),
        "tile_score_mode": str(args.tile_score_mode),
        "tile_apply_row_quota": bool(int(args.tile_apply_row_quota)),
        "tile_quota_mode": str(args.tile_quota_mode),
        "tile_use_memory": bool(int(args.tile_use_memory)),
        "tile_memory_ttl": int(args.tile_memory_ttl),
        "tile_cand_extra": int(args.tile_cand_extra),
        "force_verify_mode": str(args.force_verify_mode),
        "force_verify_key_interval": int(args.force_verify_key_interval),
    }

    merged: Dict[str, Any] = dict(common_kwargs)

    if method in ["full_sahi", "keyframe_sahi", "temporal_sahi_v6"]:
        merged.update(sahi_kwargs)

    if method == "keyframe_sahi":
        merged.update(keyframe_kwargs)

    if method == "temporal_sahi_v6":
        merged.update(v6_kwargs)
    if is_v7_family_method(method):
        merged.update(sahi_kwargs)
        merged.update(v7_kwargs)
        if method == "yolo_sahi_always_verify":
            merged["force_verify_mode"] = "always"
        elif method == "yolo_sahi_keyframe_verify":
            merged["force_verify_mode"] = "keyframe"
            merged["force_verify_key_interval"] = int(args.key_interval)

    filtered = filter_kwargs_for_class(cls, merged)
    return cls(**filtered)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="scripts2: compact run_detector with split-aware validation and safe outputs.")

    parser.add_argument("--datasets", type=str, default="fbdsv",
                        choices=["fbdsv", "icct", "icct_sparse20", "icct_sparse20_frac025", "wcs", "wcs_subset"])
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--method", type=str, required=True, choices=list(METHOD_REGISTRY.keys()))

    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--score-threshold", type=float, default=None,
                        help="Optional post-merge score threshold passed into SAHI runners (predict_frame). If omitted, defaults to conf-thres.")

    parser.add_argument("--slice-height", type=int, default=512)
    parser.add_argument("--slice-width", type=int, default=512)
    parser.add_argument("--overlap-height-ratio", type=float, default=0.2)
    parser.add_argument("--overlap-width-ratio", type=float, default=0.2)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.5)

    parser.add_argument("--key-interval", type=int, default=4)

    parser.add_argument("--v6-merge-mode", type=str, default="vanilla", choices=["vanilla", "diou"])
    parser.add_argument("--v6-exp", type=int, default=0)
    parser.add_argument("--v6-k-mid", type=int, default=3)
    parser.add_argument("--v6-full-ttl", type=int, default=3)
    parser.add_argument("--v6-log-path", type=str, default="/dev/null")
    parser.add_argument("--v6-cost-per-frame", type=str, default="/dev/null")

    parser.add_argument("--gt-json", type=str, default=None)
    parser.add_argument("--target-category-id", type=int, default=None)
    parser.add_argument("--target-category-name", type=str, default=None)
    parser.add_argument("--keep-all-categories", type=int, default=0, choices=[0, 1])

    parser.add_argument("--class-mapping-json", type=str, default=None)

    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--max-frames", type=int, default=0,
                        help="If >0, process only the first N frames and exit early.")
    parser.add_argument("--debug-out-sample", type=int, default=0,
                        help="Print first K raw preds (after runner.run) for debugging.")
    parser.add_argument("--debug-out-file", type=str, default="",
                        help="If set, dump first K raw preds to this JSON file.")
    parser.add_argument("--force-one-class", type=int, default=0, choices=[0, 1],
                        help="If 1, remap all prediction category_id to target_category_id before slim saving.")
    parser.add_argument("--save-pred-slim", type=int, default=0, choices=[0, 1])
    parser.add_argument("--save-pred-raw", type=int, default=0, choices=[0, 1],
                        help="If 1, save runner.run outputs before slim_predictions as pred_raw_*.jsonl.gz")
    parser.add_argument("--pred-max-det", type=int, default=100)

    # V7 confirmation options
    parser.add_argument("--v7-confirm-iou", type=float, default=0.4)
    parser.add_argument("--v7-confirm-len", type=int, default=2)
    parser.add_argument("--v7-confirm-max-age", type=int, default=2)

    # V7 engine-level postprocess options (keep conservative defaults)
    parser.add_argument("--v7-pp-do-clamp", type=int, default=1)  # 1/0
    parser.add_argument("--v7-pp-do-nms", type=int, default=0)
    parser.add_argument("--v7-pp-class-agnostic-nms", type=int, default=0)
    parser.add_argument("--v7-pp-max-det", type=int, default=0)
    parser.add_argument("--v7-pp-score-thr", type=float, default=0.0)
    parser.add_argument("--v7-exp", type=int, default=0)
    parser.add_argument("--v7-log-path", type=str, default="/dev/null")
    parser.add_argument("--v7-cost-per-frame", type=str, default="/dev/null")
    parser.add_argument("--cost-per-frame", type=str, default="/dev/null")
    parser.add_argument("--v7-pp-nms-iou", type=float, default=0.35)
    # Step5: ablation switches (controller components)
    parser.add_argument("--use-confirmation", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use-dual-budget", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use-targeted-verify", type=int, default=1, choices=[0, 1])
    parser.add_argument("--verifier-backend", type=str, default="sahi", choices=["sahi", "crop_recheck"])
    parser.add_argument("--crop-margin-ratio", type=float, default=0.2)
    parser.add_argument("--crop-input-size", type=int, default=960)
    parser.add_argument("--empty-explore-rate", type=float, default=0.10)
    parser.add_argument("--empty-explore-cap", type=float, default=2.0)
    parser.add_argument("--empty-explore-cost", type=float, default=1.0)
    parser.add_argument("--pos-explore-rate", type=float, default=0.15)
    parser.add_argument("--pos-explore-cap", type=float, default=2.0)
    parser.add_argument("--pos-explore-cost", type=float, default=1.0)
    parser.add_argument("--pos-motion-local-peak-thr", type=float, default=1e9)
    parser.add_argument("--pos-diff-ratio-thr", type=float, default=1.5)
    parser.add_argument("--tile-score-mode", type=str, default="mean", choices=["mean", "p95", "p99", "top1_gap"])
    parser.add_argument("--tile-apply-row-quota", type=int, default=0, choices=[0, 1])
    parser.add_argument("--tile-quota-mode", type=str, default="top1", choices=["top1", "top1_bottom1"])
    parser.add_argument("--tile-use-memory", type=int, default=0, choices=[0, 1])
    parser.add_argument("--tile-memory-ttl", type=int, default=0)
    parser.add_argument("--tile-cand-extra", type=int, default=6)
    parser.add_argument("--force-verify-mode", type=str, default="policy", choices=["policy", "always", "keyframe"])
    parser.add_argument("--force-verify-key-interval", type=int, default=4)

    args = parser.parse_args()

    # Step6: Save run configuration for reproducibility/fairness (always-on).

    # This should be written as early as possible after parsing args.

    try:

        os.makedirs(args.out_dir, exist_ok=True)

        run_cfg_path = os.path.join(args.out_dir, 'run_config.json')

        with open(run_cfg_path, 'w') as f:

            json.dump({'args': vars(args)}, f, indent=2)

        print('Saved run_config:', run_cfg_path)

    except Exception as e:

        print('Failed to save run_config.json:', repr(e))

    # Validate expected split files early (actionable error message).
    gt_auto, _ = resolve_gt_and_seq_paths(args.data_root, args.split, args.datasets)

    gt_json = args.gt_json
    if gt_json is None:
        gt_json = gt_auto

    target_cid = None
    v6_target_ids = None
    if int(args.keep_all_categories) != 1:
        target_cid = resolve_target_category_id(gt_json, args.target_category_id, args.target_category_name)
        v6_target_ids = str(target_cid)

    class_mapping = load_class_mapping(args.class_mapping_json)
    dataset = build_dataset(args.datasets, args.data_root, args.split)
    if int(args.max_frames) > 0:
        dataset = LimitedDataset(dataset, int(args.max_frames))
    runner = build_runner_from_args(args.method, args, class_mapping, v6_target_ids)

    # Optional post-merge threshold (used by SAHI-style runners). If not set, reuse conf-thres.
    score_thr = args.score_threshold if args.score_threshold is not None else float(args.conf_thres)

    # Call runner.run with only supported kwargs
    run_sig = None
    try:
        run_sig = inspect.signature(runner.run)
    except Exception:
        run_sig = None

    run_kwargs = {}
    if run_sig is not None and 'score_threshold' in run_sig.parameters:
        run_kwargs['score_threshold'] = float(score_thr)

    preds, stats = runner.run(dataset, **run_kwargs)

    # Debug: inspect raw preds right after runner output
    if int(args.debug_out_sample) > 0:
        k = int(args.debug_out_sample)
        for i, p in enumerate(preds[:k]):
            try:
                print('[RAW_PRED]', i, 'image_id', p.get('image_id'), 'category_id', p.get('category_id'), 'bbox',
                      p.get('bbox'), 'score', p.get('score'))
            except Exception:
                print('[RAW_PRED]', i, p)
        if args.debug_out_file is not None and str(args.debug_out_file).strip() != '':
            try:
                d = os.path.dirname(str(args.debug_out_file))
                if d:
                    os.makedirs(d, exist_ok=True)
                with open(str(args.debug_out_file), 'w') as f:
                    json.dump(preds[:k], f, indent=2)
                print('Saved debug raw preds to:', args.debug_out_file)
            except Exception as e:
                print('Failed to save debug raw preds:', repr(e))

    verify_schedule = resolve_verify_schedule(args.method, args.force_verify_mode)
    architecture = resolve_method_architecture(args.method)

    out_pred_slim, out_pred_raw, out_stats = default_output_paths(
        args.out_dir,
        args.method,
        args.split,
        exp_id=(
            int(args.v6_exp) if args.method == "temporal_sahi_v6"
            else int(args.v7_exp) if is_v7_family_method(args.method)
            else None
        ),
        key_interval=int(args.key_interval) if is_keyframe_method(args.method) else None,
    )

    if isinstance(stats, dict):
        stats["run_meta"] = {
            "dataset": args.datasets,
            "data_root": args.data_root,
            "split": args.split,
            "method": args.method,
            "architecture": architecture,
            "verify_schedule": verify_schedule if is_v7_family_method(args.method) else None,
            "target_category_id": int(target_cid) if target_cid is not None else None,
            "keep_all_categories": int(args.keep_all_categories),
            "conf_thres": float(args.conf_thres),
            "score_threshold": float(score_thr),
            "cost_per_frame": str(args.cost_per_frame),
            "force_one_class": int(args.force_one_class),
            "img_size": int(args.img_size),
            "slice_h": int(args.slice_height),
            "slice_w": int(args.slice_width),
            "overlap_h": float(args.overlap_height_ratio),
            "overlap_w": float(args.overlap_width_ratio),
            "merge_nms_iou": float(args.nms_iou_threshold),
            "key_interval": int(args.key_interval) if is_keyframe_method(args.method) else None,
            "v6": {
                "exp": int(args.v6_exp),
                "merge_mode": str(args.v6_merge_mode),
                "k_mid": int(args.v6_k_mid),
                "full_ttl": int(args.v6_full_ttl),
                "target_ids": v6_target_ids,
                "log_path": str(args.v6_log_path),
            } if args.method == "temporal_sahi_v6" else None,
            "local_verifier_policy": {
                "architecture": "yolo_plus_local_verifier",
                "verify_schedule": verify_schedule,
                "exp": int(args.v7_exp),
                "log_path": str(args.v7_log_path),
                "cost_per_frame": str(args.v7_cost_per_frame),
                "confirm_iou": float(args.v7_confirm_iou),
                "confirm_len": int(args.v7_confirm_len),
                "confirm_max_age": int(args.v7_confirm_max_age),
                "pp_do_clamp": int(args.v7_pp_do_clamp),
                "pp_do_nms": int(args.v7_pp_do_nms),
                "pp_nms_iou": float(args.v7_pp_nms_iou),
                "pp_class_agnostic_nms": int(args.v7_pp_class_agnostic_nms),
                "pp_max_det": int(args.v7_pp_max_det),
                "pp_score_thr": float(args.v7_pp_score_thr),
                "use_confirmation": int(args.use_confirmation),
                "use_dual_budget": int(args.use_dual_budget),
                "use_targeted_verify": int(args.use_targeted_verify),
                "verifier_backend": str(args.verifier_backend),
                "crop_margin_ratio": float(args.crop_margin_ratio),
                "crop_input_size": int(args.crop_input_size),
                "empty_explore_rate": float(args.empty_explore_rate),
                "empty_explore_cap": float(args.empty_explore_cap),
                "empty_explore_cost": float(args.empty_explore_cost),
                "pos_explore_rate": float(args.pos_explore_rate),
                "pos_explore_cap": float(args.pos_explore_cap),
                "pos_explore_cost": float(args.pos_explore_cost),
                "pos_motion_local_peak_thr": float(args.pos_motion_local_peak_thr),
                "pos_diff_ratio_thr": float(args.pos_diff_ratio_thr),
                "tile_score_mode": str(args.tile_score_mode),
                "tile_apply_row_quota": int(args.tile_apply_row_quota),
                "tile_quota_mode": str(args.tile_quota_mode),
                "tile_use_memory": int(args.tile_use_memory),
                "tile_memory_ttl": int(args.tile_memory_ttl),
                "tile_cand_extra": int(args.tile_cand_extra),
                "force_verify_mode": verify_schedule,
                "force_verify_key_interval": int(args.key_interval) if args.method == "yolo_sahi_keyframe_verify"
                else int(args.force_verify_key_interval),
            } if is_v7_family_method(args.method) else None,
        }

    # Optionally save pred_raw right after runner output (before slim_predictions).
    if isinstance(stats, dict):
        stats["pred_raw_meta"] = {
            "pred_kind": "raw",
            "saved_before_slim": True,
            "score_threshold_used": float(score_thr) if ('score_threshold' in run_kwargs) else None,
            "remap_category_id": int(target_cid) if int(args.force_one_class) == 1 else None,
        }

    save_stats(out_stats, stats if isinstance(stats, dict) else {"stats": stats})
    if int(args.save_pred_raw) == 1:
        raw_list = preds if isinstance(preds, list) else []
        # Keep runner outputs as-is. If force_one_class=1, remap category_id for fair eval_run2 filtering.
        if int(args.force_one_class) == 1:
            remapped: List[Dict[str, Any]] = []
            for p in raw_list:
                if not isinstance(p, dict):
                    continue
                q = dict(p)
                q["category_id"] = int(target_cid)
                remapped.append(q)
            raw_to_save = remapped
        else:
            raw_to_save = [p for p in raw_list if isinstance(p, dict)]

        save_jsonl_gz(out_pred_raw, raw_to_save)
        print("Saved pred_raw:", out_pred_raw)
    else:
        print("Raw pred saving is OFF (save-pred-raw=0).")

    if int(args.save_pred_slim) == 1:
        slim, meta = slim_predictions(
            preds if isinstance(preds, list) else [],
            # IMPORTANT: when force_one_class=0, DO NOT filter by target_category_id (keep multi-class preds)
            target_category_id=int(target_cid) if int(args.force_one_class) == 1 else None,
            # IMPORTANT: use the same score threshold that was passed into runner.run (post-merge threshold)
            score_thr=float(score_thr),
            max_det_per_image=int(args.pred_max_det),
            # If force_one_class=1, remap all category_id to target_cid(=1-class). Otherwise keep original classes.
            remap_category_id=int(target_cid) if int(args.force_one_class) == 1 else None,
        )
        if isinstance(stats, dict):
            stats["pred_slim_meta"] = meta
            save_stats(out_stats, stats)
        save_jsonl_gz(out_pred_slim, slim)
        print("Saved pred_slim:", out_pred_slim)
    else:
        print("Pred saving is OFF (save-pred-slim=0).")

    # Update run_config.json with resolved meta (e.g., auto GT path, target id, etc.).

    try:

        run_cfg_path = os.path.join(args.out_dir, 'run_config.json')

        if os.path.exists(run_cfg_path) and isinstance(stats, dict) and isinstance(stats.get('run_meta'), dict):
            with open(run_cfg_path, 'r') as f:
                cfg = json.load(f)

            cfg['run_meta'] = stats['run_meta']

            with open(run_cfg_path, 'w') as f:
                json.dump(cfg, f, indent=2)

    except Exception as e:

        print('Failed to update run_config.json:', repr(e))

    print("Saved stats:", out_stats)


if __name__ == "__main__":
    main()
