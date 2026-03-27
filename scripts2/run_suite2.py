#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import subprocess
from typing import Dict, List, Optional


def run(cmd: List[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def read_simple_yaml(path: str) -> Dict[str, str]:
    """
    Minimal YAML reader for key: value pairs.
    Supports comments (#) and blank lines.
    Values are returned as raw strings; casting is handled by get_* helpers.
    """
    out: Dict[str, str] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "#" in line:
                line = line.split("#", 1)[0].strip()
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k:
                out[k] = v
    return out


def get_int(cfg: Dict[str, str], key: str, default: int) -> int:
    if key not in cfg:
        return default
    try:
        return int(float(cfg[key]))
    except Exception:
        return default


def get_float(cfg: Dict[str, str], key: str, default: float) -> float:
    if key not in cfg:
        return default
    try:
        return float(cfg[key])
    except Exception:
        return default


def resolve_config_path(setting: Optional[str], config: Optional[str]) -> Optional[str]:
    if config is not None and str(config).strip():
        return config
    if setting is None or not str(setting).strip():
        return None
    setting = str(setting).strip().lower()
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(here, "configs", "{}.yaml".format(setting))
    if os.path.exists(cand):
        return cand
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="scripts2: run suite (infer -> kpi) with YAML setting support.")

    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--datasets", type=str, default="fbdsv",
                        choices=["fbdsv", "icct", "icct_sparse20", "icct_sparse20_frac025", "wcs", "wcs_subset"])
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--out-root", type=str, required=True)

    # Setting
    parser.add_argument("--setting", type=str, default=None, choices=[None, "standard", "fast"])
    parser.add_argument("--config", type=str, default=None, help="Path to YAML. If given, overrides --setting.")

    # Methods
    parser.add_argument("--method", type=str, required=True, choices=["keyframe_sahi", "temporal_sahi_v6"])
    parser.add_argument("--v6-exps", type=str, default="60,61,62")
    parser.add_argument("--v6-log-path", type=str, default="/dev/null")

    # Optional overrides (if provided, override YAML)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--conf-thres", type=float, default=None)
    parser.add_argument("--slice-height", type=int, default=None)
    parser.add_argument("--slice-width", type=int, default=None)
    parser.add_argument("--overlap-height-ratio", type=float, default=None)
    parser.add_argument("--overlap-width-ratio", type=float, default=None)
    parser.add_argument("--nms-iou-threshold", type=float, default=None)

    parser.add_argument("--key-interval", type=int, default=None)
    parser.add_argument("--pred-max-det", type=int, default=None)

    # v6 params
    parser.add_argument("--v6-merge-mode", type=str, default="diou")
    parser.add_argument("--v6-k-mid", type=int, default=3)
    parser.add_argument("--v6-full-ttl", type=int, default=3)

    # KPI options
    parser.add_argument("--save-per-frame", type=int, default=0, choices=[0, 1])

    args = parser.parse_args()
    os.makedirs(args.out_root, exist_ok=True)

    # Resolve config
    cfg_path = resolve_config_path(args.setting, args.config)
    cfg: Dict[str, str] = {}
    if cfg_path is not None:
        cfg = read_simple_yaml(cfg_path)

    # Defaults (standard baseline)
    img_size = get_int(cfg, "img_size", 640)
    conf_thres = get_float(cfg, "conf_thres", 0.25)
    slice_height = get_int(cfg, "slice_height", 512)
    slice_width = get_int(cfg, "slice_width", 512)
    overlap_height_ratio = get_float(cfg, "overlap_height_ratio", 0.2)
    overlap_width_ratio = get_float(cfg, "overlap_width_ratio", 0.2)
    nms_iou_threshold = get_float(cfg, "nms_iou_threshold", 0.5)
    key_interval = get_int(cfg, "key_interval", 4)
    pred_max_det = get_int(cfg, "pred_max_det", 100)

    # Apply CLI overrides if provided
    if args.img_size is not None:
        img_size = int(args.img_size)
    if args.conf_thres is not None:
        conf_thres = float(args.conf_thres)
    if args.slice_height is not None:
        slice_height = int(args.slice_height)
    if args.slice_width is not None:
        slice_width = int(args.slice_width)
    if args.overlap_height_ratio is not None:
        overlap_height_ratio = float(args.overlap_height_ratio)
    if args.overlap_width_ratio is not None:
        overlap_width_ratio = float(args.overlap_width_ratio)
    if args.nms_iou_threshold is not None:
        nms_iou_threshold = float(args.nms_iou_threshold)
    if args.key_interval is not None:
        key_interval = int(args.key_interval)
    if args.pred_max_det is not None:
        pred_max_det = int(args.pred_max_det)

    # Paths
    gt = os.path.join(args.data_root, "annotations", "fbdsv24_vid_{}.json".format(args.split))
    seq = os.path.join(args.data_root, "annotations", "fbdsv24_vid_{}_sequences.json".format(args.split))

    infer_out = os.path.join(args.out_root, "infer")
    kpi_root = os.path.join(args.out_root, "kpi")
    os.makedirs(infer_out, exist_ok=True)
    os.makedirs(kpi_root, exist_ok=True)

    # Keyframe
    if args.method == "keyframe_sahi":
        run([
            "python", "scripts2/run_detector2.py",
            "--datasets", args.datasets,
            "--data-root", args.data_root,
            "--split", args.split,
            "--method", "keyframe_sahi",
            "--weights", args.weights,
            "--device", args.device,
            "--img-size", str(img_size),
            "--conf-thres", str(conf_thres),
            "--slice-height", str(slice_height),
            "--slice-width", str(slice_width),
            "--overlap-height-ratio", str(overlap_height_ratio),
            "--overlap-width-ratio", str(overlap_width_ratio),
            "--nms-iou-threshold", str(nms_iou_threshold),
            "--key-interval", str(key_interval),
            "--out-dir", infer_out,
            "--save-pred-slim", "1",
            "--pred-max-det", str(pred_max_det),
        ])

        pred = os.path.join(infer_out, "pred_slim_keyframe_sahi_n{}_{}.jsonl.gz".format(key_interval, args.split))
        stats = os.path.join(infer_out, "stats_keyframe_sahi_n{}_{}.json".format(key_interval, args.split))
        out_dir = os.path.join(kpi_root, "keyframe_n{}_{}".format(key_interval, args.split))
        os.makedirs(out_dir, exist_ok=True)

        run([
            "python", "scripts2/eval_run2.py",
            "--gt", gt,
            "--seq", seq,
            "--pred", pred,
            "--stats", stats,
            "--method", "keyframe_sahi",
            "--dataset", args.datasets,
            "--split", args.split,
            "--setting", (args.setting if args.setting else (os.path.basename(cfg_path).replace(".yaml", "") if cfg_path else "")),
            "--seed", "42",
            "--img-size", str(img_size),
            "--conf-thr", str(conf_thres),
            "--slice-height", str(slice_height),
            "--slice-width", str(slice_width),
            "--overlap-height-ratio", str(overlap_height_ratio),
            "--overlap-width-ratio", str(overlap_width_ratio),
            "--merge-nms-iou", str(nms_iou_threshold),
            "--key-interval", str(key_interval),
            "--out-dir", out_dir,
            "--save-per-frame", str(args.save_per_frame),
        ])
        return

    # v6 burst
    exps = [x.strip() for x in args.v6_exps.split(",") if x.strip()]
    for e in exps:
        run([
            "python", "scripts2/run_detector2.py",
            "--datasets", args.datasets,
            "--data-root", args.data_root,
            "--split", args.split,
            "--method", "temporal_sahi_v6",
            "--weights", args.weights,
            "--device", args.device,
            "--img-size", str(img_size),
            "--conf-thres", str(conf_thres),
            "--slice-height", str(slice_height),
            "--slice-width", str(slice_width),
            "--overlap-height-ratio", str(overlap_height_ratio),
            "--overlap-width-ratio", str(overlap_width_ratio),
            "--nms-iou-threshold", str(nms_iou_threshold),
            "--v6-exp", str(e),
            "--v6-merge-mode", str(args.v6_merge_mode),
            "--v6-k-mid", str(args.v6_k_mid),
            "--v6-full-ttl", str(args.v6_full_ttl),
            "--v6-log-path", str(args.v6_log_path),
            "--out-dir", infer_out,
            "--save-pred-slim", "1",
            "--pred-max-det", str(pred_max_det),
        ])

        pred = os.path.join(infer_out, "pred_slim_temporal_sahi_v6_exp{}_{}.jsonl.gz".format(e, args.split))
        stats = os.path.join(infer_out, "stats_temporal_sahi_v6_exp{}_{}.json".format(e, args.split))
        out_dir = os.path.join(kpi_root, "v6_exp{}_{}".format(e, args.split))
        os.makedirs(out_dir, exist_ok=True)

        run([
            "python", "scripts2/eval_run2.py",
            "--gt", gt,
            "--seq", seq,
            "--pred", pred,
            "--stats", stats,
            "--method", "temporal_sahi_v6",
            "--dataset", args.datasets,
            "--split", args.split,
            "--exp-id", str(e),
            "--setting", (args.setting if args.setting else (os.path.basename(cfg_path).replace(".yaml", "") if cfg_path else "")),
            "--seed", "42",
            "--img-size", str(img_size),
            "--conf-thr", str(conf_thres),
            "--slice-height", str(slice_height),
            "--slice-width", str(slice_width),
            "--overlap-height-ratio", str(overlap_height_ratio),
            "--overlap-width-ratio", str(overlap_width_ratio),
            "--merge-nms-iou", str(nms_iou_threshold),
            "--v6-merge-mode", str(args.v6_merge_mode),
            "--v6-k-mid", str(args.v6_k_mid),
            "--v6-full-ttl", str(args.v6_full_ttl),
            "--out-dir", out_dir,
            "--save-per-frame", str(args.save_per_frame),
        ])


if __name__ == "__main__":
    main()
