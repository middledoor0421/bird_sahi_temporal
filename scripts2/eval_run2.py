# scripts2_v2/eval_run2.py
# !/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import gzip
import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from scripts2.gt_utils import resolve_target_category_id
from scripts2.pred_slim import load_pred_any
import gzip
import json


V7_FAMILY_METHODS = {
    "temporal_sahi_v7",
    "yolo_sahi_always_verify",
    "yolo_sahi_keyframe_verify",
}

KEYFRAME_METHODS = {
    "keyframe_sahi",
    "yolo_sahi_keyframe_verify",
}


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def mean(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    return float(sum(xs)) / float(len(xs))


def percentile(xs: List[float], q: float) -> Optional[float]:
    if not xs:
        return None
    ys = sorted(xs)
    if q <= 0.0:
        return float(ys[0])
    if q >= 100.0:
        return float(ys[-1])
    k = (len(ys) - 1) * (q / 100.0)
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return float(ys[f])
    d0 = ys[f] * (c - k)
    d1 = ys[c] * (k - f)
    return float(d0 + d1)


def summarize(xs: List[float]) -> Dict[str, Any]:
    return {"mean": mean(xs), "p50": percentile(xs, 50.0), "p90": percentile(xs, 90.0), "count": int(len(xs))}


def iou_xywh(a: List[float], b: List[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    union = (aw * ah) + (bw * bh) - inter
    if union <= 0.0:
        return 0.0
    return float(inter) / float(union)


def greedy_frame_match(gt_boxes: List[List[float]], pred_boxes: List[List[float]], pred_scores: List[float],
                       iou_thr: float) -> Tuple[int, int, int, float]:
    max_conf = float(max(pred_scores)) if pred_scores else 0.0
    if not gt_boxes:
        return 0, int(len(pred_boxes)), 0, max_conf
    if not pred_boxes:
        return 0, 0, 1, 0.0

    order = list(range(len(pred_boxes)))
    order.sort(key=lambda i: float(pred_scores[i]), reverse=True)

    matched_gt = [False] * len(gt_boxes)
    tp = 0
    fp = 0
    for i in order:
        pb = pred_boxes[i]
        best_j = -1
        best_iou = 0.0
        for j, gb in enumerate(gt_boxes):
            if matched_gt[j]:
                continue
            v = iou_xywh(pb, gb)
            if v > best_iou:
                best_iou = v
                best_j = j
        if best_j >= 0 and best_iou >= float(iou_thr):
            matched_gt[best_j] = True
            tp = 1
        else:
            fp += 1
    fn = 0 if tp == 1 else 1
    return int(tp), int(fp), int(fn), max_conf


def load_sequences(seq_path: str) -> List[Dict[str, Any]]:
    obj = json.load(open(seq_path, "r"))
    if not isinstance(obj, list):
        raise ValueError("Sequences json must be a list.")
    return obj


def iter_frames(video_entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    frames = video_entry.get("frames", [])
    if isinstance(frames, list) and len(frames) > 0:
        for idx, fr in enumerate(frames):
            if isinstance(fr, dict):
                out.append({"frame_idx": idx, "image_id": fr.get("image_id", None)})
            else:
                out.append({"frame_idx": idx, "image_id": fr})
        return out

    frame_ids = video_entry.get("frame_ids", [])
    if isinstance(frame_ids, list) and len(frame_ids) > 0:
        for idx, image_id in enumerate(frame_ids):
            out.append({"frame_idx": idx, "image_id": image_id})
    return out


def estimate_full_tiles(w: int, h: int, slice_w: int, slice_h: int, ov_w: float, ov_h: float) -> int:
    sw = max(1, int(slice_w))
    sh = max(1, int(slice_h))
    stride_w = max(1.0, float(sw) * (1.0 - float(ov_w)))
    stride_h = max(1.0, float(sh) * (1.0 - float(ov_h)))

    if w <= sw:
        nx = 1
    else:
        nx = int(math.ceil((float(w) - float(sw)) / stride_w)) + 1
    if h <= sh:
        ny = 1
    else:
        ny = int(math.ceil((float(h) - float(sh)) / stride_h)) + 1
    return int(nx * ny)


def compute_segments(flags: List[bool]) -> List[Tuple[int, int]]:
    segs: List[Tuple[int, int]] = []
    s = None
    for i, f in enumerate(flags):
        if f and s is None:
            s = i
        if (not f) and s is not None:
            segs.append((s, i - 1))
            s = None
    if s is not None:
        segs.append((s, len(flags) - 1))
    return segs


def clamp_bbox_xywh(b, W, H):
    # Clamp COCO bbox [x,y,w,h] into image bounds; return (clamped_bbox, is_oob)
    x, y, w, h = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    oob = False

    if w < 0.0 or h < 0.0:
        oob = True
        w = max(0.0, w)
        h = max(0.0, h)

    if x < 0.0:
        oob = True
        x = 0.0
    if y < 0.0:
        oob = True
        y = 0.0

    if x > float(W):
        oob = True
        x = float(W)
        w = 0.0
    if y > float(H):
        oob = True
        y = float(H)
        h = 0.0

    if x + w > float(W):
        oob = True
        w = max(0.0, float(W) - x)
    if y + h > float(H):
        oob = True
        h = max(0.0, float(H) - y)

    return [x, y, w, h], oob


def build_gt_index(gt_path: str, target_cid: int, small_taus: List[float]) -> Tuple[
    Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]], Dict[str, Any]]:
    gt = json.load(open(gt_path, "r"))
    images_map = {int(im["id"]): im for im in gt.get("images", [])}

    anns_by_img: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for a in gt.get("annotations", []):
        try:
            cid = int(a.get("category_id", -1))
        except Exception:
            continue
        if cid != int(target_cid):
            continue
        iid = int(a["image_id"])
        anns_by_img[iid].append(a)

    gt_info: Dict[int, Dict[str, Any]] = {}
    for iid, im in images_map.items():
        w = int(im.get("width", 0))
        h = int(im.get("height", 0))
        img_area = float(max(1, w * h))

        gt_boxes: List[List[float]] = []
        has_small = {tau: False for tau in small_taus}
        for a in anns_by_img.get(iid, []):
            bbox = a.get("bbox", None)
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            bx, by, bw, bh = [float(x) for x in bbox]
            gt_boxes.append([bx, by, bw, bh])
            b_area = max(0.0, bw) * max(0.0, bh)
            for tau in small_taus:
                if (b_area / img_area) <= float(tau):
                    has_small[tau] = True

        gt_info[iid] = {
            "w": w,
            "h": h,
            "gt_boxes": gt_boxes,
            "is_pos": True if gt_boxes else False,
            "is_empty": True if not gt_boxes else False,
            "has_small": has_small,
        }

    return gt_info, images_map, gt


def build_pred_index(pred_path: str, target_cid: int) -> Dict[int, List[Dict[str, Any]]]:
    preds = load_pred_any(pred_path)
    by_img: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for p in preds:
        if not isinstance(p, dict):
            continue
        try:
            cid = int(p.get("category_id", -1))
        except Exception:
            continue
        if cid != int(target_cid):
            continue
        try:
            iid = int(p.get("image_id"))
        except Exception:
            continue
        by_img[iid].append(p)
    return by_img


def coco_eval_basic(gt_json_path: str, preds_list: List[Dict[str, Any]], tmp_dir: str) -> Dict[str, Any]:
    # Patch GT to include fields required by COCOeval
    ensure_dir(tmp_dir)
    if len(preds_list) == 0:
        return {"mAP": 0.0, "AP50": 0.0, "Recall": 0.0}

    gt = json.load(open(gt_json_path, "r"))
    if "info" not in gt:
        gt["info"] = {}
    if "licenses" not in gt:
        gt["licenses"] = []
    for a in gt.get("annotations", []):
        # COCOeval expects iscrowd
        if "iscrowd" not in a:
            a["iscrowd"] = 0
        # area is commonly expected; if missing, compute from bbox
        if "area" not in a:
            bbox = a.get("bbox", None)
            if isinstance(bbox, list) and len(bbox) == 4:
                bw = float(bbox[2])
                bh = float(bbox[3])
                a["area"] = max(0.0, bw) * max(0.0, bh)
            else:
                a["area"] = 0.0

    tmp_gt = os.path.join(tmp_dir, "__tmp_gt_for_coco.json")
    with open(tmp_gt, "w") as f:
        json.dump(gt, f)

    coco_gt = COCO(tmp_gt)

    tmp_pred = os.path.join(tmp_dir, "__tmp_pred_for_coco.json")
    with open(tmp_pred, "w") as f:
        json.dump(preds_list, f)

    coco_dt = coco_gt.loadRes(tmp_pred)
    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    out = {"mAP": float(ev.stats[0]), "AP50": float(ev.stats[1]), "Recall": float(ev.stats[8])}
    return out


def build_small_gt_subset(full_gt: Dict[str, Any], images_map: Dict[int, Dict[str, Any]], target_cid: int,
                          tau: float) -> Tuple[Dict[str, Any], List[int]]:
    kept_anns: List[Dict[str, Any]] = []
    kept_img_ids = set()

    for a in full_gt.get("annotations", []):
        try:
            cid = int(a.get("category_id", -1))
        except Exception:
            continue
        if cid != int(target_cid):
            continue
        iid = int(a["image_id"])
        im = images_map.get(iid, None)
        if im is None:
            continue
        w = float(max(1, int(im.get("width", 1))))
        h = float(max(1, int(im.get("height", 1))))
        img_area = w * h

        bbox = a.get("bbox", None)
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        bw = float(bbox[2])
        bh = float(bbox[3])
        b_area = max(0.0, bw) * max(0.0, bh)
        if (b_area / img_area) <= float(tau):
            kept_anns.append(a)
            kept_img_ids.add(iid)

    subset = {
        "images": [images_map[iid] for iid in sorted(list(kept_img_ids))],
        "annotations": kept_anns,
        "categories": [c for c in full_gt.get("categories", []) if int(c.get("id", -1)) == int(target_cid)],
        "info": full_gt.get("info", {}),
        "licenses": full_gt.get("licenses", []),
    }
    return subset, sorted(list(kept_img_ids))


def write_per_video_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        with open(path, "w") as f:
            f.write("")
        return
    cols = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def load_cost_map(path: str) -> Optional[Dict[int, Dict[str, Any]]]:
    p = str(path).strip()
    if p == "" or p == "/dev/null":
        return None
    m: Dict[int, Dict[str, Any]] = {}
    import gzip, json
    with gzip.open(p, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            iid = int(r.get("image_id", -1))
            if iid >= 0:
                m[iid] = r
    return m


def resolve_cost_map_path(
    explicit_path: str,
    method: str,
    stats_path: Optional[str],
    stats_run_meta: Dict[str, Any],
    out_dir: str,
) -> str:
    candidates: List[str] = []
    p = str(explicit_path).strip() if isinstance(explicit_path, str) else ""
    if p != "":
        candidates.append(p)

    if isinstance(stats_run_meta, dict):
        root_cost = str(stats_run_meta.get("cost_per_frame", "")).strip()
        if root_cost != "":
            candidates.append(root_cost)
        branch_meta = (
            stats_run_meta.get("local_verifier_policy", stats_run_meta.get("v7"))
            if method in V7_FAMILY_METHODS
            else stats_run_meta.get("v6")
        )
        if isinstance(branch_meta, dict):
            branch_cost = str(branch_meta.get("cost_per_frame", branch_meta.get("cost_per_frame_path", ""))).strip()
            if branch_cost != "":
                candidates.append(branch_cost)

    if isinstance(stats_path, str) and stats_path.strip() != "":
        stats_dir = os.path.dirname(stats_path)
        candidates.append(os.path.join(stats_dir, "cost_per_frame.jsonl.gz"))

    candidates.append(os.path.join(out_dir, "cost_per_frame.jsonl.gz"))

    seen = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        if c != "" and c != "/dev/null" and os.path.exists(c):
            return c
    return p


def main() -> None:
    parser = argparse.ArgumentParser(description="scripts2_v2: schema-aligned KPI runner")
    parser.add_argument("--gt", type=str, required=True)
    parser.add_argument("--seq", type=str, required=True)
    parser.add_argument("--pred", type=str, required=True)
    parser.add_argument("--stats", type=str, default=None)

    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--split", type=str, default="")
    parser.add_argument("--method", type=str, required=True,
                        choices=[
                            "yolo",
                            "full_sahi",
                            "keyframe_sahi",
                            "temporal_sahi_v6",
                            "temporal_sahi_v7",
                            "yolo_sahi_always_verify",
                            "yolo_sahi_keyframe_verify",
                        ])
    parser.add_argument("--exp-id", type=str, default="")
    parser.add_argument("--setting", type=str, default="standard")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--target-category-id", type=int, default=None)
    parser.add_argument("--target-category-name", type=str, default=None)
    parser.add_argument("--iou-thr", type=float, default=0.5)
    parser.add_argument("--conf-thr", type=float, default=0.25)

    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--slice-height", type=int, default=512)
    parser.add_argument("--slice-width", type=int, default=512)
    parser.add_argument("--overlap-height-ratio", type=float, default=0.2)
    parser.add_argument("--overlap-width-ratio", type=float, default=0.2)
    parser.add_argument("--merge-nms-iou", type=float, default=0.5)

    parser.add_argument("--key-interval", type=int, default=4)

    parser.add_argument("--v6-budget", type=str, default="")
    parser.add_argument("--v6-merge-mode", type=str, default="")
    parser.add_argument("--v6-ttl", type=str, default="")
    parser.add_argument("--v6-cooldown", type=str, default="")
    parser.add_argument("--v6-k-mid", type=str, default="")
    parser.add_argument("--v6-k-max", type=str, default="")

    parser.add_argument("--small-taus", type=str, default="0.01,0.02,0.03")
    parser.add_argument("--save-per-frame", type=int, default=0, choices=[0, 1])
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--cost-per-frame", type=str, default="")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    tmp_dir = os.path.join(args.out_dir, "_tmp")
    ensure_dir(tmp_dir)

    # Baseline cost-per-frame caching:
    # For yolo/full_sahi/keyframe_sahi we can deterministically derive per-frame SAHI usage and tile counts,
    # so we optionally materialize a cost_per_frame.jsonl.gz for reuse across later analyses.
    auto_save_cost = False
    cost_out_path = str(args.cost_per_frame).strip() if isinstance(args.cost_per_frame, str) else ""
    if cost_out_path == "" or cost_out_path == "/dev/null":
        if str(args.method) in ["yolo", "full_sahi", "keyframe_sahi"]:
            cost_out_path = os.path.join(args.out_dir, "cost_per_frame.jsonl.gz")
            auto_save_cost = True
    cost_writer = None
    if auto_save_cost:
        try:
            cost_writer = gzip.open(cost_out_path, "wt", encoding="utf-8")
            print("Auto-saving baseline cost_per_frame:", cost_out_path)
        except Exception as e:
            print("Failed to open cost_per_frame writer:", repr(e))
            cost_writer = None

    small_taus = [float(x.strip()) for x in args.small_taus.split(",") if x.strip()]
    safety_taus = [0.3, 0.5, 0.7]
    target_cid = resolve_target_category_id(args.gt, args.target_category_id, args.target_category_name)

    gt_info, images_map, full_gt = build_gt_index(args.gt, target_cid, small_taus)
    pred_by_img = build_pred_index(args.pred, target_cid)
    preds_list = load_pred_any(args.pred)
    seq_list = load_sequences(args.seq)

    stats = json.load(open(args.stats, "r")) if args.stats else None
    stats_run_meta = (stats.get("run_meta") if isinstance(stats, dict) else None) or {}
    pred_slim_meta = (stats.get("pred_slim_meta") if isinstance(stats, dict) else None) or {}

    v6_avg_tiles = 0.0
    v6_lat = 0.0
    if stats:
        if "avg_tiles_selected" in stats:
            v6_avg_tiles = float(stats.get("avg_tiles_selected", 0.0))
        for k in ["avg_latency_ms", "avg_total_ms", "mean_latency_ms"]:
            if k in stats:
                v6_lat = float(stats[k])
                break

    pf = None
    pf_path = os.path.join(args.out_dir, "per_frame.jsonl.gz")
    if int(args.save_per_frame) == 1:
        pf = gzip.open(pf_path, "wt", encoding="utf-8")

    per_video_rows: List[Dict[str, Any]] = []

    delay_all: List[float] = []
    miss_all: List[float] = []
    sahi_on_all: List[float] = []
    tiles_all: List[float] = []
    lat_all: List[float] = []

    empty_fp_frames_total = 0
    empty_frames_total = 0
    empty_sahi_on_total = 0
    empty_tiles_sum = 0.0

    # Debug / extra aggregates
    bbox_oob_cnt = 0
    bbox_cnt = 0
    tp_frames_total = 0
    fp_frames_total = 0
    fn_frames_total = 0
    num_sahi_frames_all = 0
    total_tiles_all = 0.0
    total_tiles_sahi_all = 0.0
    num_pred_boxes_all = 0

    small_delay: Dict[float, List[float]] = {tau: [] for tau in small_taus}
    small_miss: Dict[float, List[float]] = {tau: [] for tau in small_taus}

    cost_out_path = resolve_cost_map_path(
        explicit_path=cost_out_path,
        method=str(args.method),
        stats_path=args.stats,
        stats_run_meta=stats_run_meta,
        out_dir=args.out_dir,
    )
    cost_map = load_cost_map(cost_out_path)
    use_cost_map = cost_map is not None
    if str(args.method) in [
        "temporal_sahi_v6",
        "temporal_sahi_v7",
        "yolo_sahi_always_verify",
        "yolo_sahi_keyframe_verify",
    ] and not use_cost_map:
        raise FileNotFoundError(
            "Per-frame achieved cost log is required for {} evaluation. "
            "Checked --cost-per-frame, stats metadata, sibling stats dir, and out-dir.".format(args.method)
        )

    for v in seq_list:
        vid = str(v.get("video_id", ""))
        frames = iter_frames(v)

        gt_pos: List[bool] = []
        gt_empty: List[bool] = []
        gt_small: Dict[float, List[bool]] = {tau: [] for tau in small_taus}
        pred_has_at: Dict[float, List[bool]] = {tau: [] for tau in safety_taus}
        pred_count_at: Dict[float, List[int]] = {tau: [] for tau in safety_taus}

        tp_flags: List[int] = []
        fn_flags: List[int] = []
        fp_counts: List[int] = []

        sahi_on_v: List[int] = []
        tiles_v: List[float] = []
        lat_v: List[float] = []

        for fr in frames:
            iid_raw = fr.get("image_id", None)
            if iid_raw is None:
                continue
            iid = int(iid_raw)

            gi = gt_info.get(iid, {"w": 0, "h": 0, "gt_boxes": [], "is_pos": False, "is_empty": True,
                                   "has_small": {tau: False for tau in small_taus}})
            w = int(gi["w"])
            h = int(gi["h"])
            gt_boxes = gi["gt_boxes"]

            gt_pos.append(bool(gi["is_pos"]))
            gt_empty.append(bool(gi["is_empty"]))
            for tau in small_taus:
                gt_small[tau].append(bool(gi["has_small"].get(tau, False)))

            preds = pred_by_img.get(iid, [])
            cnt_at = {tau: 0 for tau in safety_taus}
            pboxes: List[List[float]] = []
            pscores: List[float] = []
            for p in preds:
                bbox = p.get("bbox", None)
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                score = float(p.get("score", 0.0))
                for tau in safety_taus:
                    if score >= float(tau):
                        cnt_at[tau] += 1
                if score < float(args.conf_thr):
                    continue
                bb, oob = clamp_bbox_xywh(bbox, w, h)
                if oob:
                    bbox_oob_cnt += 1
                bbox_cnt += 1
                pboxes.append([float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])])
                pscores.append(score)

            for tau in safety_taus:
                pred_count_at[tau].append(int(cnt_at[tau]))
                pred_has_at[tau].append(bool(cnt_at[tau] > 0))

            tp, fp_cnt, fn, max_conf = greedy_frame_match(gt_boxes, pboxes, pscores, float(args.iou_thr))
            tp_flags.append(int(tp))
            tp_frames_total += int(tp)
            fp_frames_total += 1 if int(fp_cnt) > 0 else 0
            fn_frames_total += int(fn)
            num_pred_boxes_all += int(len(pboxes))
            fp_counts.append(int(fp_cnt))
            fn_flags.append(int(fn))

            full_tiles = estimate_full_tiles(w, h, args.slice_width, args.slice_height, args.overlap_width_ratio,
                                             args.overlap_height_ratio)

            if args.method == "yolo":
                sahi_on, tiles_used, lat = 0, 0.0, 0.0
            elif args.method == "full_sahi":
                sahi_on, tiles_used, lat = 1, float(full_tiles), 0.0
            elif args.method == "keyframe_sahi":
                sahi_on = 1 if (int(fr["frame_idx"]) % int(args.key_interval) == 0) else 0
                tiles_used = float(full_tiles) if sahi_on == 1 else 0.0
                lat = 0.0
            else:
                if use_cost_map and (iid in cost_map):
                    r = cost_map[iid]
                    sahi_on = int(r.get("sahi_on", 0))
                    tiles_used = float(r.get("tiles_used", 0))
                    lat = float(r.get("latency_ms", 0.0))
                else:
                    tiles_used = float(v6_avg_tiles)
                    sahi_on = 1 if tiles_used > 0.0 else 0
                    lat = float(v6_lat)

            # Optionally write per-frame cost record (jsonl.gz)
            if cost_writer is not None:
                try:
                    rec_cost = {
                        "video_id": vid,
                        "t": int(fr["frame_idx"]),
                        "image_id": int(iid),
                        "sahi_on": int(sahi_on),
                        "tiles_used": int(tiles_used) if float(tiles_used).is_integer() else float(tiles_used),
                        "latency_ms": float(lat),
                    }
                    cost_writer.write(json.dumps(rec_cost) + "\n")
                except Exception:
                    pass
            sahi_on_v.append(int(sahi_on))
            total_tiles_all += float(tiles_used)
            if int(sahi_on) == 1:
                num_sahi_frames_all += 1
                total_tiles_sahi_all += float(tiles_used)
            tiles_v.append(float(tiles_used))
            lat_v.append(float(lat))

            if gi["is_empty"]:
                empty_frames_total += 1
                if sahi_on == 1:
                    empty_sahi_on_total += 1
                empty_tiles_sum += float(tiles_used)
                if fp_cnt > 0:
                    empty_fp_frames_total += 1

            if pf is not None:
                rec = {
                    "video_id": vid,
                    "t": int(fr["frame_idx"]),
                    "image_id": iid,
                    "gt_is_empty": bool(gi["is_empty"]),
                    "gt_has_obj": bool(gi["is_pos"]),
                    "gt_num_boxes": int(len(gt_boxes)),
                    "pred_num_final": int(len(pboxes)),
                    "tp": int(tp),
                    "fp": int(fp_cnt),
                    "fn": int(fn),
                    "max_conf": float(max_conf),
                    "mode": str(args.method),
                    "sahi_on": int(sahi_on),
                    "tiles_used": float(tiles_used),
                    "key_interval": int(args.key_interval) if args.method in KEYFRAME_METHODS else None,
                    "v6_budget": str(args.v6_budget) if args.method == "temporal_sahi_v6" else None,
                    "latency_ms": float(lat),
                }
                for tau in small_taus:
                    rec["gt_has_small_tau_{:.2f}".format(tau).replace(".", "_")] = bool(gi["has_small"].get(tau, False))
                pf.write(json.dumps(rec) + "\n")

        n = len(gt_pos)
        if n == 0:
            continue

        num_pos_frames = int(sum(1 for x in gt_pos if x))
        num_empty_frames = int(sum(1 for x in gt_empty if x))
        presence_ratio = float(num_pos_frames) / float(n) if n > 0 else 0.0

        segs = compute_segments(gt_pos)
        delays: List[float] = []
        misses: List[float] = []
        for s, e in segs:
            first_tp = None
            for t in range(s, e + 1):
                if tp_flags[t] == 1:
                    first_tp = t
                    break
            delays.append(float(e - s + 1) if first_tp is None else float(first_tp - s))

            best = 0
            cur = 0
            for t in range(s, e + 1):
                if fn_flags[t] == 1:
                    cur += 1
                else:
                    best = max(best, cur)
                    cur = 0
            best = max(best, cur)
            misses.append(float(best))

        delay_all.extend(delays)
        miss_all.extend(misses)
        sahi_on_all.extend([float(x) for x in sahi_on_v])
        tiles_all.extend(tiles_v)
        lat_all.extend(lat_v)

        empty_total_v = num_empty_frames
        empty_fp_v = int(sum(1 for t in range(n) if gt_empty[t] and fp_counts[t] > 0))
        empty_on_total_v = int(sum(1 for t in range(n) if gt_empty[t] and sahi_on_v[t] == 1))
        empty_off_total_v = int(sum(1 for t in range(n) if gt_empty[t] and sahi_on_v[t] == 0))
        empty_fp_on_v = int(sum(1 for t in range(n) if gt_empty[t] and sahi_on_v[t] == 1 and fp_counts[t] > 0))
        empty_fp_off_v = int(sum(1 for t in range(n) if gt_empty[t] and sahi_on_v[t] == 0 and fp_counts[t] > 0))
        empty_fp_rate_v = (float(empty_fp_v) / float(empty_total_v)) if empty_total_v > 0 else 0.0
        effr: Dict[float, float] = {}
        efpd: Dict[float, float] = {}
        effr_on: Dict[float, float] = {}
        effr_off: Dict[float, float] = {}
        efpd_on: Dict[float, float] = {}
        efpd_off: Dict[float, float] = {}
        for tau in safety_taus:
            empty_has = sum(1 for t in range(n) if gt_empty[t] and pred_has_at[tau][t])
            empty_cnt = sum(int(pred_count_at[tau][t]) for t in range(n) if gt_empty[t])
            empty_has_on = sum(1 for t in range(n) if gt_empty[t] and sahi_on_v[t] == 1 and pred_has_at[tau][t])
            empty_has_off = sum(1 for t in range(n) if gt_empty[t] and sahi_on_v[t] == 0 and pred_has_at[tau][t])
            empty_cnt_on = sum(int(pred_count_at[tau][t]) for t in range(n) if gt_empty[t] and sahi_on_v[t] == 1)
            empty_cnt_off = sum(int(pred_count_at[tau][t]) for t in range(n) if gt_empty[t] and sahi_on_v[t] == 0)
            effr[tau] = (float(empty_has) / float(empty_total_v)) if empty_total_v > 0 else 0.0
            efpd[tau] = (float(empty_cnt) / float(empty_total_v)) if empty_total_v > 0 else 0.0
            effr_on[tau] = (float(empty_has_on) / float(empty_on_total_v)) if empty_on_total_v > 0 else 0.0
            effr_off[tau] = (float(empty_has_off) / float(empty_off_total_v)) if empty_off_total_v > 0 else 0.0
            efpd_on[tau] = (float(empty_cnt_on) / float(empty_on_total_v)) if empty_on_total_v > 0 else 0.0
            efpd_off[tau] = (float(empty_cnt_off) / float(empty_off_total_v)) if empty_off_total_v > 0 else 0.0
        empty_sahi_on_ratio_v = (float(sum(1 for t in range(n) if gt_empty[t] and sahi_on_v[t] == 1)) / float(
            empty_total_v)) if empty_total_v > 0 else 0.0
        empty_avg_tiles_v = mean([tiles_v[t] for t in range(n) if gt_empty[t]])
        pos_total_v = int(n - empty_total_v)
        pos_sahi_on_ratio_v = (float(sum(1 for t in range(n) if (not gt_empty[t]) and sahi_on_v[t] == 1)) / float(
            pos_total_v)) if pos_total_v > 0 else 0.0
        pos_avg_tiles_v = mean([tiles_v[t] for t in range(n) if (not gt_empty[t])])

        for tau in small_taus:
            segs_s = compute_segments(gt_small[tau])
            for s, e in segs_s:
                first_tp = None
                for t in range(s, e + 1):
                    if tp_flags[t] == 1:
                        first_tp = t
                        break
                small_delay[tau].append(float(e - s + 1) if first_tp is None else float(first_tp - s))

                best = 0
                cur = 0
                for t in range(s, e + 1):
                    if fn_flags[t] == 1:
                        cur += 1
                    else:
                        best = max(best, cur)
                        cur = 0
                best = max(best, cur)
                small_miss[tau].append(float(best))

        row = {
            "video_id": vid,
            "num_frames": int(n),
            "num_pos_frames": int(num_pos_frames),
            "num_empty_frames": int(num_empty_frames),
            "presence_ratio": float(presence_ratio),
            "delay_mean": mean(delays),
            "delay_p50": percentile(delays, 50.0),
            "delay_p90": percentile(delays, 90.0),
            "miss_mean": mean(misses),
            "miss_p50": percentile(misses, 50.0),
            "miss_p90": percentile(misses, 90.0),
            "sahi_on_ratio": mean([float(x) for x in sahi_on_v]),
            "avg_tiles": mean(tiles_v),
            "avg_latency_ms": mean(lat_v),
            "fps": (1000.0 / mean(lat_v)) if (mean(lat_v) and mean(lat_v) > 0.0) else None,
            "empty_sahi_on_ratio": float(empty_sahi_on_ratio_v),
            "empty_avg_tiles": empty_avg_tiles_v,
            "pos_sahi_on_ratio": float(pos_sahi_on_ratio_v),
            "pos_avg_tiles": pos_avg_tiles_v,
            "empty_fp_rate": float(empty_fp_rate_v),
            "empty_fp_rate_on": (float(empty_fp_on_v) / float(empty_on_total_v)) if empty_on_total_v > 0 else 0.0,
            "empty_fp_rate_off": (float(empty_fp_off_v) / float(empty_off_total_v)) if empty_off_total_v > 0 else 0.0,
            "empty_frames_sahi_on": int(empty_on_total_v),
            "empty_frames_sahi_off": int(empty_off_total_v),
        }
        for tau in safety_taus:
            row["conditional_effr_on_tau_{:.1f}".format(tau).replace(".", "_")] = float(effr_on[tau])
            row["conditional_effr_off_tau_{:.1f}".format(tau).replace(".", "_")] = float(effr_off[tau])
        for tau in small_taus:
            row["has_small_tau_{:.2f}".format(tau).replace(".", "_")] = 1 if any(gt_small[tau]) else 0

        per_video_rows.append(row)

    if pf is not None:
        pf.close()

    if cost_writer is not None:
        try:
            cost_writer.close()
        except Exception:
            pass

    per_video_csv = os.path.join(args.out_dir, "per_video.csv")
    write_per_video_csv(per_video_csv, per_video_rows)

    # Clamp all prediction bboxes to image bounds before COCOeval
    preds_list_clamped = []
    for p in preds_list:
        try:
            iid = int(p.get('image_id', -1))
        except Exception:
            continue
        gi = gt_info.get(iid, None)
        if gi is None:
            continue
        W = int(gi.get('w', 0));
        H = int(gi.get('h', 0))
        bbox = p.get('bbox', None)
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        bb, oob = clamp_bbox_xywh(bbox, W, H)
        if oob:
            bbox_oob_cnt += 1
        bbox_cnt += 1
        preds_list_clamped.append({
            'image_id': iid,
            'category_id': int(p.get('category_id', target_cid)),
            'bbox': bb,
            'score': float(p.get('score', 0.0)),
        })
    coco_all = coco_eval_basic(args.gt, preds_list_clamped, tmp_dir)
    bbox_oob_rate = float(bbox_oob_cnt) / float(bbox_cnt) if bbox_cnt > 0 else 0.0

    empty_fp_rate_global = (float(empty_fp_frames_total) / float(empty_frames_total)) if empty_frames_total > 0 else 0.0
    empty_sahi_on_ratio_global = (
            float(empty_sahi_on_total) / float(empty_frames_total)) if empty_frames_total > 0 else 0.0
    empty_avg_tiles_global = (float(empty_tiles_sum) / float(empty_frames_total)) if empty_frames_total > 0 else 0.0

    avg_tiles_mean = mean(tiles_all)
    avg_tiles_p90 = percentile(tiles_all, 90.0)
    lat_mean = mean(lat_all)
    lat_p90 = percentile(lat_all, 90.0)
    fps = (1000.0 / lat_mean) if (lat_mean and lat_mean > 0.0) else None

    small_tau_out: Dict[str, Any] = {}
    for tau in small_taus:
        subset_gt, subset_img_ids = build_small_gt_subset(full_gt, images_map, target_cid, tau)
        if len(subset_img_ids) == 0:
            small_tau_out[str(tau)] = {
                "mAP": 0.0,
                "AP50": 0.0,
                "Recall": 0.0,
                "delay": summarize(small_delay[tau]),
                "miss_streak": summarize(small_miss[tau]),
            }
            continue

        tmp_gt = os.path.join(tmp_dir, "__tmp_gt_small_tau_{:.3f}.json".format(tau).replace(".", "_"))
        with open(tmp_gt, "w") as f:
            json.dump(subset_gt, f)

        subset_preds = [p for p in preds_list_clamped if int(p.get("image_id", -1)) in set(subset_img_ids)]
        coco_small = coco_eval_basic(tmp_gt, subset_preds, tmp_dir)

        try:
            os.remove(tmp_gt)
        except Exception:
            pass

        small_tau_out[str(tau)] = {
            "mAP": float(coco_small["mAP"]),
            "AP50": float(coco_small["AP50"]),
            "Recall": float(coco_small["Recall"]),
            "delay": summarize(small_delay[tau]),
            "miss_streak": summarize(small_miss[tau]),
        }

    summary = {
        "run_meta": {
            "dataset": args.dataset if args.dataset else stats_run_meta.get("dataset", ""),
            "split": args.split if args.split else stats_run_meta.get("split", ""),
            "method": args.method,
            "exp_id": args.exp_id,
            "seed": int(args.seed),
            "setting": args.setting,
            "gt_target_category_id": int(target_cid),
            "pred_score_thr": float(
                pred_slim_meta.get('pred_score_thr', pred_slim_meta.get('score_thr', args.conf_thr))),
            "small_def": "bbox_area / image_area <= tau",
            "sahi": {
                "img_size": int(stats_run_meta.get("img_size", args.img_size)) if str(
                    stats_run_meta.get("img_size", "")).isdigit() else int(args.img_size),
                "slice_h": int(args.slice_height),
                "slice_w": int(args.slice_width),
                "overlap_h": float(args.overlap_height_ratio),
                "overlap_w": float(args.overlap_width_ratio),
                "merge_nms_iou": float(args.merge_nms_iou),
                "conf_thres": float(args.conf_thr),
            },
            "keyframe": {"key_interval": int(args.key_interval) if args.method in KEYFRAME_METHODS else None},
            "v6": {
                "budget": str(args.v6_budget),
                "merge_mode": str(args.v6_merge_mode),
                "ttl": str(args.v6_ttl),
                "cooldown": str(args.v6_cooldown),
                "k_mid": str(args.v6_k_mid),
                "k_max": str(args.v6_k_max),
            } if args.method == "temporal_sahi_v6" else None,
            "local_verifier_policy": (
                stats_run_meta.get("local_verifier_policy", stats_run_meta.get("v7"))
                if isinstance(stats_run_meta, dict)
                else None
            ) if args.method in V7_FAMILY_METHODS else None,
        },
        "quality": {
            "mAP": float(coco_all["mAP"]),
            "AP50": float(coco_all["AP50"]),
            "Recall": float(coco_all["Recall"]),
            "delay": summarize(delay_all),
            "miss_streak": summarize(miss_all),
            "empty_fp_rate": float(empty_fp_rate_global),
            "num_tp_frames": int(tp_frames_total),
            "num_fp_frames": int(fp_frames_total),
            "num_fn_frames": int(fn_frames_total),
            "bbox_oob_rate": float(bbox_oob_rate),
        },
        "safety": {
            "taus": [float(t) for t in safety_taus],
            "EFFR": {},
            "EFPD": {},
            "EFFR_on": {str(t): None for t in safety_taus},
            "EFFR_off": {str(t): None for t in safety_taus},
            "EFPD_on": {str(t): None for t in safety_taus},
            "EFPD_off": {str(t): None for t in safety_taus},
        },

        "sanity": {
            "bbox_oob_rate": float(bbox_oob_rate),
            "num_pred_raw": int(pred_slim_meta.get('num_in', 0)) if str(
                pred_slim_meta.get('num_in', '')).isdigit() else pred_slim_meta.get('num_in', ''),
            "num_pred_kept": int(pred_slim_meta.get('num_out', 0)) if str(
                pred_slim_meta.get('num_out', '')).isdigit() else pred_slim_meta.get('num_out', ''),
            "sahi_on_ratio": float(mean(sahi_on_all)) if mean(sahi_on_all) is not None else 0.0,
            "avg_tiles_per_sahi_frame": float(total_tiles_sahi_all) / float(
                num_sahi_frames_all) if num_sahi_frames_all > 0 else 0.0,
        },
        "cost": {
            "sahi_on_ratio": mean(sahi_on_all),
            "avg_tiles_per_frame": {"mean": avg_tiles_mean, "p90": avg_tiles_p90},
            "latency_ms": {"mean": lat_mean, "p90": lat_p90},
            "fps": fps,
            "empty_wasted_compute": {
                "empty_sahi_on_ratio": float(empty_sahi_on_ratio_global),
                "empty_avg_tiles": float(empty_avg_tiles_global),
            },
            "num_sahi_frames": int(num_sahi_frames_all),
            "total_tiles": float(total_tiles_all),
            "avg_tiles_per_sahi_frame": float(total_tiles_sahi_all) / float(
                num_sahi_frames_all) if num_sahi_frames_all > 0 else 0.0,
            "num_pred_boxes": int(num_pred_boxes_all),
            "num_pred_raw": int(pred_slim_meta.get('num_in', 0)) if str(
                pred_slim_meta.get('num_in', '')).isdigit() else pred_slim_meta.get('num_in', ''),
            "num_pred_kept": int(pred_slim_meta.get('num_out', 0)) if str(
                pred_slim_meta.get('num_out', '')).isdigit() else pred_slim_meta.get('num_out', ''),
            "pred_score_thr": float(
                pred_slim_meta.get('pred_score_thr', pred_slim_meta.get('score_thr', args.conf_thr))),
        },
        "small_tau": small_tau_out,
        "paths": {
            "per_video_csv": "per_video.csv",
            "per_frame_jsonl_gz": "per_frame.jsonl.gz" if int(args.save_per_frame) == 1 else None,
        },
    }

    # Step2: confirmation statistics from v7 log (optional)
    confirm_stats = {}
    try:
        import glob
        stats_dir = os.path.dirname(str(args.stats)) if args.stats is not None else str(args.out_dir)
        cand = sorted(glob.glob(os.path.join(stats_dir, "*log*.json")))
        if len(cand) > 0:
            log_path = cand[-1]
            with open(log_path, "r") as f:
                logs = json.load(f)
            ttc_all = []
            newly_total = 0
            conf_frames = 0
            unconf_frames = 0
            for r in logs:
                cm = r.get("confirm", {}) if isinstance(r, dict) else {}
                conf_frames += 1 if int(cm.get("num_confirmed", 0)) > 0 else 0
                unconf_frames += 1 if int(cm.get("num_unconfirmed", 0)) > 0 else 0
                newly_total += int(cm.get("newly_confirmed", 0))
                ttc_list = cm.get("ttc_list", [])
                if isinstance(ttc_list, list):
                    for v in ttc_list:
                        try:
                            ttc_all.append(int(v))
                        except Exception:
                            pass
            if len(ttc_all) > 0:
                ttc_all_sorted = sorted(ttc_all)
                p90 = ttc_all_sorted[int(0.9 * (len(ttc_all_sorted) - 1))]
                confirm_stats = {
                    "log_path": os.path.basename(log_path),
                    "confirm_frames": int(conf_frames),
                    "unconfirmed_frames": int(unconf_frames),
                    "newly_confirmed_total": int(newly_total),
                    "ttc_count": int(len(ttc_all)),
                    "ttc_mean": float(sum(ttc_all) / max(1, len(ttc_all))),
                    "ttc_p90": int(p90),
                }
            else:
                confirm_stats = {
                    "log_path": os.path.basename(log_path),
                    "confirm_frames": int(conf_frames),
                    "unconfirmed_frames": int(unconf_frames),
                    "newly_confirmed_total": int(newly_total),
                    "ttc_count": 0,
                }
    except Exception:
        confirm_stats = {}

    # Step3: dual-budget ledger statistics from v7 log (optional)
    budget_stats = {}
    try:
        import glob
        stats_dir = os.path.dirname(str(args.stats)) if args.stats is not None else str(args.out_dir)
        cand = sorted(glob.glob(os.path.join(stats_dir, "*log*.json")))
        if len(cand) > 0:
            log_path = cand[-1]
            with open(log_path, "r") as f:
                logs = json.load(f)
            from collections import Counter
            reason_cnt = Counter()
            bucket_cnt = Counter()
            pos_likely_cnt = 0
            frames = 0
            frames_with_pos_keys = 0
            for r in logs:
                if not isinstance(r, dict):
                    continue
                pl = r.get("plan", {}) if isinstance(r.get("plan", {}), dict) else {}
                frames += 1
                reason_cnt[str(pl.get("escalate_reason", ""))] += 1
                cd = pl.get("cooldown", {}) if isinstance(pl.get("cooldown", {}), dict) else {}
                if "pos_tokens" in cd or "pos_explore_rate" in cd:
                    frames_with_pos_keys += 1
                if bool(cd.get("pos_likely", False)):
                    pos_likely_cnt += 1
                bucket_cnt[str(cd.get("bucket_used", ""))] += 1
            budget_stats = {
                "log_path": os.path.basename(log_path),
                "frames": int(frames),
                "frames_with_pos_keys": int(frames_with_pos_keys),
                "pos_likely_frames": int(pos_likely_cnt),
                "pos_budget_explore": int(reason_cnt.get("pos_budget_explore", 0)),
                "empty_budget_skip": int(reason_cnt.get("empty_budget_skip", 0)),
                "seed_empty_explore": int(reason_cnt.get("seed_empty_explore", 0)),
                "verify_burst": int(reason_cnt.get("verify_burst", 0)),
                "bucket_used": dict(bucket_cnt),
            }
    except Exception:
        budget_stats = {}

    if budget_stats:
        summary["budget"] = budget_stats

    if confirm_stats:
        summary["confirm"] = confirm_stats

    # Step1: Safety KPI (EFFR/EFPD) with fixed tau list
    try:
        safety_taus = [0.3, 0.5, 0.7]
        effr: Dict[str, float] = {}
        efpd: Dict[str, float] = {}
        effr_on: Dict[str, float] = {}
        effr_off: Dict[str, float] = {}
        efpd_on: Dict[str, float] = {}
        efpd_off: Dict[str, float] = {}
        empty_ids: List[int] = [int(iid) for iid, gi in gt_info.items() if
                                isinstance(gi, dict) and bool(gi.get("is_empty", False))]
        num_empty = int(len(empty_ids))
        num_empty_on = 0
        num_empty_off = 0
        if isinstance(cost_map, dict):
            num_empty_on = sum(1 for iid in empty_ids if int(cost_map.get(int(iid), {}).get("sahi_on", 0)) > 0)
            num_empty_off = int(num_empty - num_empty_on)
        for tau in safety_taus:
            fp_frames = 0
            fp_boxes = 0
            fp_frames_on = 0
            fp_frames_off = 0
            fp_boxes_on = 0
            fp_boxes_off = 0
            for iid in empty_ids:
                preds = pred_by_img.get(int(iid), [])
                cnt = 0
                for p in preds:
                    try:
                        s = float(p.get("score", 0.0))
                    except Exception:
                        s = 0.0
                    if s >= float(tau):
                        cnt += 1
                so = int(cost_map.get(int(iid), {}).get("sahi_on", 0)) if isinstance(cost_map, dict) else 0
                if cnt > 0:
                    fp_frames += 1
                    fp_boxes += cnt
                    if so > 0:
                        fp_frames_on += 1
                        fp_boxes_on += cnt
                    else:
                        fp_frames_off += 1
                        fp_boxes_off += cnt
            key = str(float(tau))
            effr[key] = float(fp_frames / num_empty) if num_empty > 0 else 0.0
            efpd[key] = float(fp_boxes / num_empty) if num_empty > 0 else 0.0
            effr_on[key] = float(fp_frames_on / num_empty_on) if num_empty_on > 0 else 0.0
            effr_off[key] = float(fp_frames_off / num_empty_off) if num_empty_off > 0 else 0.0
            efpd_on[key] = float(fp_boxes_on / num_empty_on) if num_empty_on > 0 else 0.0
            efpd_off[key] = float(fp_boxes_off / num_empty_off) if num_empty_off > 0 else 0.0
        summary["safety"] = {
            "taus": [float(x) for x in safety_taus],
            "EFFR": effr,
            "EFPD": efpd,
            "EFFR_on": effr_on,
            "EFFR_off": effr_off,
            "EFPD_on": efpd_on,
            "EFPD_off": efpd_off,
            "empty_frames": int(num_empty),
            "empty_frames_sahi_on": int(num_empty_on),
            "empty_frames_sahi_off": int(num_empty_off),
        }
    except Exception as e:
        summary["safety"] = {"error": repr(e)}

    # Step3: Budget/ledger + empty/pos SAHI-on ratio split using local-verifier policy log if available
    try:
        budget_extra: Dict[str, Any] = {}
        verifier_log_path = None
        if isinstance(stats_run_meta, dict):
            verifier_meta = stats_run_meta.get("local_verifier_policy")
            if not isinstance(verifier_meta, dict):
                verifier_meta = stats_run_meta.get("v7") if isinstance(stats_run_meta.get("v7"), dict) else None
            if isinstance(verifier_meta, dict):
                verifier_log_path = verifier_meta.get("log_path")
        if isinstance(verifier_log_path, str) and verifier_log_path.strip() != "" and os.path.exists(verifier_log_path):
            v7_logs = json.load(open(verifier_log_path, "r"))
            # Build map image_id -> pos_likely / seed_empty
            pos_likely_by_img: Dict[int, bool] = {}
            seed_empty_by_img: Dict[int, bool] = {}
            bucket_used_by_img: Dict[int, str] = {}
            for r in v7_logs:
                if not isinstance(r, dict):
                    continue
                try:
                    iid = int(r.get("image_id"))
                except Exception:
                    continue
                cond = r.get("cond", {}) if isinstance(r.get("cond", {}), dict) else {}
                seed_empty_by_img[iid] = bool(cond.get("seed_empty", False))
                plan = r.get("plan", {}) if isinstance(r.get("plan", {}), dict) else {}
                cd = plan.get("cooldown", {}) if isinstance(plan.get("cooldown", {}), dict) else {}
                pos_likely = bool(cd.get("pos_likely", False))
                pos_likely_by_img[iid] = pos_likely
                bucket_used_by_img[iid] = str(cd.get("bucket_used", "none"))

            # sahi_on based on cost_map if present else infer from log tile_plan tiles
            def _infer_sahi_on_from_log(rec: Dict[str, Any]) -> bool:
                tp = rec.get("tile_plan", {}) if isinstance(rec.get("tile_plan", {}), dict) else {}
                tiles = tp.get("tiles", [])
                if isinstance(tiles, list) and len(tiles) > 0:
                    return True
                pl = rec.get("plan", {}) if isinstance(rec.get("plan", {}), dict) else {}
                try:
                    return int(pl.get("K", 0)) > 0
                except Exception:
                    return False

            sahi_on_empty = 0
            sahi_on_pos = 0
            frames_empty = 0
            frames_pos = 0

            # Also compute by seed_empty (control-side)
            sahi_on_seed = 0
            sahi_on_nonseed = 0
            frames_seed = 0
            frames_nonseed = 0

            if isinstance(cost_map, dict):
                for iid, cm in cost_map.items():
                    try:
                        iid_i = int(iid)
                    except Exception:
                        continue
                    so = int(cm.get("sahi_on", 0)) if isinstance(cm, dict) else 0
                    # pos_likely split
                    if bool(pos_likely_by_img.get(iid_i, False)):
                        frames_pos += 1
                        sahi_on_pos += int(so > 0)
                    else:
                        frames_empty += 1
                        sahi_on_empty += int(so > 0)
                    # seed_empty split
                    if bool(seed_empty_by_img.get(iid_i, False)):
                        frames_seed += 1
                        sahi_on_seed += int(so > 0)
                    else:
                        frames_nonseed += 1
                        sahi_on_nonseed += int(so > 0)
            else:
                # fallback: use v7 log inference
                for r in v7_logs:
                    if not isinstance(r, dict):
                        continue
                    try:
                        iid_i = int(r.get("image_id"))
                    except Exception:
                        continue
                    so = int(_infer_sahi_on_from_log(r))
                    if bool(pos_likely_by_img.get(iid_i, False)):
                        frames_pos += 1
                        sahi_on_pos += so
                    else:
                        frames_empty += 1
                        sahi_on_empty += so
                    if bool(seed_empty_by_img.get(iid_i, False)):
                        frames_seed += 1
                        sahi_on_seed += so
                    else:
                        frames_nonseed += 1
                        sahi_on_nonseed += so

            budget_extra.update({
                "v7_log_path": str(v7_log_path),
                "frames_with_pos_keys": int(sum(1 for _ in pos_likely_by_img.values())),  # keys present for mapped ids
                "pos_likely_frames": int(sum(1 for v in pos_likely_by_img.values() if v)),
                "sahi_on_ratio_pos_likely": float(sahi_on_pos / frames_pos) if frames_pos > 0 else 0.0,
                "sahi_on_ratio_non_pos_likely": float(sahi_on_empty / frames_empty) if frames_empty > 0 else 0.0,
                "sahi_on_ratio_seed_empty": float(sahi_on_seed / frames_seed) if frames_seed > 0 else 0.0,
                "sahi_on_ratio_non_seed_empty": float(sahi_on_nonseed / frames_nonseed) if frames_nonseed > 0 else 0.0,
            })
        # Merge into budget section (keep existing counts)
        if "budget" not in summary or not isinstance(summary.get("budget"), dict):
            summary["budget"] = {}
        summary["budget"].update(budget_extra)
    except Exception as e:
        if "budget" not in summary or not isinstance(summary.get("budget"), dict):
            summary["budget"] = {}
        summary["budget"]["step3_error"] = repr(e)
    summary_path = os.path.join(args.out_dir, "summary.json")

    # Step5: ablation switches (if run_config.json exists)
    ablation = None
    try:
        rc_path = os.path.join(os.path.dirname(args.stats), "run_config.json")
        if os.path.exists(rc_path):
            rc = json.load(open(rc_path, "r"))
            a = rc.get("args", {})
            ablation = {
                "use_confirmation": int(a.get("use_confirmation", 1)),
                "use_dual_budget": int(a.get("use_dual_budget", 1)),
                "use_targeted_verify": int(a.get("use_targeted_verify", 1)),
            }
    except Exception:
        ablation = None
    if ablation is not None:
        summary["ablation"] = ablation

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved:", summary_path)
    print("Saved:", per_video_csv)
    if int(args.save_per_frame) == 1:
        print("Saved:", pf_path)


if __name__ == "__main__":
    main()
