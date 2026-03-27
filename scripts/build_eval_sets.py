#!/usr/bin/env python
# -*- coding: utf-8 -*-

# scripts/build_eval_sets.py
# Build conditional evaluation sets (track_id lists) from GT COCO only.
# Python 3.9 compatible. Comments in English only.

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import os

def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: str, obj: Any) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def bbox_area_xywh(b: List[float]) -> float:
    return float(b[2]) * float(b[3])


def bbox_center_xywh(b: List[float]) -> Tuple[float, float]:
    x, y, w, h = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    return x + 0.5 * w, y + 0.5 * h


def iou_xywh(a: List[float], b: List[float]) -> float:
    ax, ay, aw, ah = map(float, a)
    bx, by, bw, bh = map(float, b)
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def build_frame_order(images: List[Dict[str, Any]]) -> Dict[int, Tuple[Any, Any, int]]:
    order: Dict[int, Tuple[Any, Any, int]] = {}
    for img in images:
        img_id = int(img["id"])
        video_key = img.get("video_id", 0)
        if "frame_id" in img:
            frame_key = img["frame_id"]
        elif "frame_index" in img:
            frame_key = img["frame_index"]
        else:
            frame_key = img_id
        order[img_id] = (video_key, frame_key, img_id)
    return order


def group_by_video(images: List[Dict[str, Any]]) -> Dict[Any, List[int]]:
    order = build_frame_order(images)
    by_video: Dict[Any, List[Tuple[Any, Any, int]]] = defaultdict(list)
    for img_id, (vk, fk, iid) in order.items():
        by_video[vk].append((vk, fk, iid))
    out: Dict[Any, List[int]] = {}
    for vk, lst in by_video.items():
        lst.sort(key=lambda x: (x[1], x[2]))
        out[vk] = [int(x[2]) for x in lst]
    return out


@dataclass
class Track:
    track_id: int
    video_id: Any
    frames: List[int]
    gt_boxes: List[List[float]]


def build_pseudo_tracks(
    by_video_frames: Dict[Any, List[int]],
    gt_by_img: Dict[int, List[List[float]]],
    link_iou_thr: float,
) -> List[Track]:
    tracks: List[Track] = []
    next_tid = 1

    for vid, frame_ids in by_video_frames.items():
        active: Dict[int, List[float]] = {}

        for img_id in frame_ids:
            gts = gt_by_img.get(img_id, [])
            if len(gts) == 0:
                active = {}
                continue

            assigned = set()
            new_active: Dict[int, List[float]] = {}

            for tid, last_box in active.items():
                best_j = -1
                best_iou = 0.0
                for j, g in enumerate(gts):
                    if j in assigned:
                        continue
                    v = iou_xywh(last_box, g)
                    if v > best_iou:
                        best_iou = v
                        best_j = j
                if best_j >= 0 and best_iou >= link_iou_thr:
                    assigned.add(best_j)
                    g = gts[best_j]
                    new_active[tid] = g
                    for tr in tracks:
                        if tr.track_id == tid:
                            tr.frames.append(img_id)
                            tr.gt_boxes.append(g)
                            break

            for j, g in enumerate(gts):
                if j in assigned:
                    continue
                tid = next_tid
                next_tid += 1
                tracks.append(Track(track_id=tid, video_id=vid, frames=[img_id], gt_boxes=[g]))
                new_active[tid] = g

            active = new_active

    return tracks


def main() -> None:
    parser = argparse.ArgumentParser(description="Build conditional eval sets (track_id lists) from GT.")
    parser.add_argument("--gt", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)

    parser.add_argument("--category-id", type=int, default=14)
    parser.add_argument("--small-max-area", type=float, default=1024.0)

    parser.add_argument("--link-iou-thr", type=float, default=0.3)

    # Sets params
    parser.add_argument("--s2-max-len", type=int, default=10)
    parser.add_argument("--s3-small-frac", type=float, default=0.5)
    parser.add_argument("--motion-top-p", type=float, default=0.2)
    parser.add_argument("--motion-bot-p", type=float, default=0.2)

    # S1: new-appearance heavy (GT gap before track start)
    parser.add_argument("--s1-gap-frames", type=int, default=5)

    args = parser.parse_args()

    gt = load_json(args.gt)
    images = gt.get("images", [])
    ann = gt.get("annotations", [])

    by_video = group_by_video(images)

    cat_id = int(args.category_id)
    gt_by_img: Dict[int, List[List[float]]] = defaultdict(list)
    for a in ann:
        if int(a.get("category_id", -1)) != cat_id:
            continue
        b = a.get("bbox", None)
        if b is None:
            continue
        gt_by_img[int(a["image_id"])].append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])

    tracks = build_pseudo_tracks(by_video, gt_by_img, link_iou_thr=float(args.link_iou_thr))

    # Build per-video GT presence for S1 (gap before track start)
    gt_present: Dict[Any, set] = defaultdict(set)
    for img_id, boxes in gt_by_img.items():
        # find video_id via images map
        # slow path: build reverse map once
        pass

    img_id_to_vid = {}
    for img in images:
        img_id_to_vid[int(img["id"])] = img.get("video_id", 0)

    # Track stats
    track_len = {}
    track_small_frac = {}
    track_avg_motion = {}

    for tr in tracks:
        track_len[tr.track_id] = len(tr.frames)
        # small fraction
        small_cnt = 0
        for b in tr.gt_boxes:
            if bbox_area_xywh(b) <= float(args.small_max_area):
                small_cnt += 1
        track_small_frac[tr.track_id] = float(small_cnt) / float(max(1, len(tr.frames)))

        # avg motion (GT center displacement per step)
        if len(tr.gt_boxes) <= 1:
            track_avg_motion[tr.track_id] = 0.0
        else:
            ds = []
            for i in range(1, len(tr.gt_boxes)):
                x0, y0 = bbox_center_xywh(tr.gt_boxes[i - 1])
                x1, y1 = bbox_center_xywh(tr.gt_boxes[i])
                ds.append(float(np.hypot(x1 - x0, y1 - y0)))
            track_avg_motion[tr.track_id] = float(np.mean(ds)) if len(ds) > 0 else 0.0

    # S1: tracks that start after a GT-absent gap of >= s1_gap_frames in that video
    gap_frames = int(args.s1_gap_frames)
    s1_ids = []
    for tr in tracks:
        vid = tr.video_id
        frames = by_video.get(vid, [])
        if len(frames) == 0:
            continue
        start_img = tr.frames[0]
        idx = frames.index(start_img) if start_img in frames else None
        if idx is None:
            continue
        # count consecutive frames before start with zero GT
        gap = 0
        j = idx - 1
        while j >= 0:
            prev_img = frames[j]
            if len(gt_by_img.get(prev_img, [])) > 0:
                break
            gap += 1
            j -= 1
        if gap >= gap_frames:
            s1_ids.append(tr.track_id)

    # S2: short dwell
    s2_ids = [tid for tid, L in track_len.items() if int(L) <= int(args.s2_max_len)]

    # S3: hard-small
    s3_ids = [tid for tid, frac in track_small_frac.items() if float(frac) >= float(args.s3_small_frac)]

    # S4: high-motion (top p)
    p_top = float(args.motion_top_p)
    mot_vals = np.asarray([track_avg_motion[tid] for tid in track_avg_motion], dtype=np.float32)
    thr_hi = float(np.quantile(mot_vals, 1.0 - p_top)) if mot_vals.size > 0 else 1e30
    s4_ids = [tid for tid, m in track_avg_motion.items() if float(m) >= thr_hi]

    # S5: low-motion small = S3 ∩ bottom p motion
    p_bot = float(args.motion_bot_p)
    thr_lo = float(np.quantile(mot_vals, p_bot)) if mot_vals.size > 0 else -1.0
    s5_ids = [tid for tid in s3_ids if float(track_avg_motion.get(tid, 0.0)) <= thr_lo]

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    def dump(name: str, track_ids: List[int]) -> None:
        save_json(f"{out_dir}/{name}.json", {"type": "track_id_list", "track_ids": sorted(list(set(track_ids))) })

    dump("all", [tr.track_id for tr in tracks])
    dump("new_appearance", s1_ids)
    dump("short_dwell", s2_ids)
    dump("hard_small", s3_ids)
    dump("high_motion", s4_ids)
    dump("low_motion_small", s5_ids)

    # Save a track index for later filtering (track_id -> frames)
    track_index = {tr.track_id: tr.frames for tr in tracks}
    save_json(f"{out_dir}/track_index.json", track_index)

    print("Saved eval sets to:", out_dir)
    print("Tracks:", len(tracks))
    print("S1 new_appearance:", len(s1_ids))
    print("S2 short_dwell:", len(s2_ids))
    print("S3 hard_small:", len(s3_ids))
    print("S4 high_motion:", len(s4_ids))
    print("S5 low_motion_small:", len(s5_ids))


if __name__ == "__main__":
    main()
