#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/make_window_sparse_split.py

Adds PASS_MIN sample-size constraints for MAIN splits (sparse_20/15/10):
- num_event_windows >= pass_min_event_windows
- num_videos_with_events >= pass_min_videos_with_events

Keeps:
- sparse_5_appx appendix rules
- presence tolerance checks
"""

import argparse
import json
import math
import os
import random
from collections import defaultdict
from statistics import mean, median


def w_start(w):
    return w.get("start", w.get("start_frame"))


def w_end(w):
    return w.get("end", w.get("end_frame"))


def load_json(p):
    with open(p, "r") as f:
        return json.load(f)


def load_video_list_ids(p):
    """Load video ids from a json file.

    Supports either a plain list (recommended) or a wrapped dict.
    If a dict is provided, the first list-like value is used, preferring
    common keys such as 'videos' or 'video_ids'.
    """
    obj = load_json(p)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        preferred = [
            "videos",
            "video_ids",
            "vids",
            "dev_videos",
            "holdout_videos",
            "val_videos",
            "train_videos",
        ]
        for k in preferred:
            v = obj.get(k)
            if isinstance(v, list):
                return v
        for v in obj.values():
            if isinstance(v, list):
                return v
    raise ValueError("video_list json must be a list or a dict containing a list")


def save_json(obj, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)


def percentile(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (q / 100.0)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return s[lo]
    w = k - lo
    return s[lo] * (1 - w) + s[hi] * w


def stats(vals):
    if not vals:
        return {"min": None, "max": None, "mean": None, "median": None, "p90": None}
    return {
        "min": int(min(vals)),
        "max": int(max(vals)),
        "mean": float(mean(vals)),
        "median": float(median(vals)),
        "p90": float(percentile(vals, 90)),
    }


def build_has_target(coco, cat_id):
    m = defaultdict(int)
    for a in coco.get("annotations", []):
        if int(a.get("category_id", -1)) == int(cat_id):
            m[int(a["image_id"])] = 1
    return m


def presence_flags(frame_ids, has_target):
    return [int(has_target.get(int(fid), 0)) for fid in frame_ids]


def pos_segments(flags):
    segs = []
    i = 0
    n = len(flags)
    while i < n:
        if flags[i] == 1:
            j = i
            while j + 1 < n and flags[j + 1] == 1:
                j += 1
            segs.append((i, j))
            i = j + 1
        else:
            i += 1
    return segs


def max_run_len(flags, val):
    best = cur = 0
    for f in flags:
        if f == val:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def window_pos_count(flags, s, e):
    return int(sum(flags[s:e + 1]))


def make_event_windows(video_id, frame_ids, flags, L, pre_min, pre_max, post_min, post_max,
                       short_pre_min, short_pre_max, rng):
    n = len(frame_ids)
    L = min(int(L), n)
    wins = []
    for (s, e) in pos_segments(flags):
        if n < 32:
            pre = rng.randint(short_pre_min, short_pre_max)
            post = rng.randint(short_pre_min, short_pre_max)
        else:
            pre = rng.randint(pre_min, pre_max)
            post = rng.randint(post_min, post_max)

        start = max(0, s - pre)
        end = start + L - 1
        if end >= n:
            end = n - 1
            start = max(0, end - (L - 1))

        if e > end:
            end = min(n - 1, e + post)
            start = max(0, end - (L - 1))
            end = start + L - 1
            if end >= n:
                end = n - 1
                start = max(0, end - (L - 1))

        # Enforce that the event segment overlaps the window (guarantees event coverage)
        if not (start <= e and end >= s):
            continue

        # posf is kept only for reporting/debugging; it is NOT a filtering criterion.
        posf = window_pos_count(flags, start, end)

        wins.append({
            "video_id": video_id,
            "start": int(start),
            "end": int(end),
            "type": "event",
            "pos_frames_in_window": int(posf),
            "len_frames": int(end - start + 1),
        })
    return wins


def make_neg_windows(video_id, frame_ids, flags, L, neg_eps, step):
    n = len(frame_ids)
    L = min(int(L), n)
    step = max(1, int(step))
    wins = []
    for s in range(0, n - L + 1, step):
        e = s + L - 1
        posf = window_pos_count(flags, s, e)
        if posf / float(L) <= float(neg_eps):
            wins.append({
                "video_id": video_id,
                "start": int(s),
                "end": int(e),
                "type": "neg",
                "pos_frames_in_window": int(posf),
                "len_frames": int(L),
                "pos_ratio_in_window": float(posf / float(L)),
            })
    return wins


def select_event_balanced(event_pool, target_count, max_per_video, rng):
    by_vid = defaultdict(list)
    for w in event_pool:
        by_vid[w["video_id"]].append(w)
    vids = list(by_vid.keys())
    rng.shuffle(vids)
    for vid in vids:
        rng.shuffle(by_vid[vid])

    selected = []
    used = defaultdict(int)
    max_per_video = max(1, int(max_per_video))

    made = True
    while len(selected) < target_count and made:
        made = False
        for vid in vids:
            if len(selected) >= target_count:
                break
            if used[vid] >= max_per_video or not by_vid[vid]:
                continue
            selected.append(by_vid[vid].pop())
            used[vid] += 1
            made = True
    return selected


def select_windows(event_pool, neg_pool, target, tol, min_event, max_per_video, rng, pass_E, pass_V):
    max_per_video = max(1, int(max_per_video))
    event_pool = list(event_pool)
    neg_pool = list(neg_pool)
    rng.shuffle(event_pool)
    rng.shuffle(neg_pool)

    # Event count target
    event_target = max(int(min_event), int(pass_E))
    event_target = min(event_target, len(event_pool))

    selected_event = select_event_balanced(event_pool, event_target, max_per_video, rng)

    # Try to increase distinct videos with events
    def num_videos(sel):
        return len({w["video_id"] for w in sel})

    safe_cap = min(len(event_pool), max(event_target, pass_V * 2))
    while pass_V > 0 and num_videos(selected_event) < pass_V and len(selected_event) < safe_cap:
        # increase by small increments
        selected_event = select_event_balanced(event_pool, len(selected_event) + 5, max_per_video, rng)
        if len(selected_event) >= safe_cap:
            break

    # Compute required neg
    P = sum(int(w["pos_frames_in_window"]) for w in selected_event)
    T_event = sum(int(w["len_frames"]) for w in selected_event)
    avg_neg_len = int(round(mean([w["len_frames"] for w in neg_pool]))) if neg_pool else 50
    desired_total = int(math.ceil(P / float(target))) if target > 0 else T_event
    neg_frames_needed = max(0, desired_total - T_event)
    num_neg_needed = int(math.ceil(neg_frames_needed / float(avg_neg_len))) if avg_neg_len > 0 else 0

    used_cnt = defaultdict(int)
    for w in selected_event:
        used_cnt[w["video_id"]] += 1

    selected_neg = []
    for w in neg_pool:
        if len(selected_neg) >= num_neg_needed:
            break
        if used_cnt[w["video_id"]] >= max_per_video:
            continue
        selected_neg.append(w)
        used_cnt[w["video_id"]] += 1

    def achieved(selE, selN):
        sel = selE + selN
        pos = sum(int(w["pos_frames_in_window"]) for w in sel)
        tot = sum(int(w["len_frames"]) for w in sel)
        return (pos / float(tot)) if tot > 0 else 0.0, pos, tot

    ach, pos, tot = achieved(selected_event, selected_neg)

    # Adjust to tolerance (add neg first)
    for _ in range(10):
        if abs(ach - target) <= tol:
            break
        if ach > target + tol:
            added = False
            for w in neg_pool:
                if used_cnt[w["video_id"]] >= max_per_video:
                    continue
                selected_neg.append(w)
                used_cnt[w["video_id"]] += 1
                added = True
                break
            if not added:
                break
        else:
            if not selected_neg:
                break
            w = selected_neg.pop()
            used_cnt[w["video_id"]] -= 1
        ach, pos, tot = achieved(selected_event, selected_neg)

    debug = {
        "event_pool_size": len(event_pool),
        "neg_pool_size": len(neg_pool),
        "selected_event_windows": len(selected_event),
        "selected_neg_windows": len(selected_neg),
        "pos_frames": int(pos),
        "total_frames": int(tot),
        "achieved_ratio": float(ach),
        "pass_E_min": int(pass_E),
        "pass_V_min": int(pass_V),
    }
    return selected_event, selected_neg, debug


def flatten_image_ids(seqs_map, windows):
    by_vid = defaultdict(list)
    for w in windows:
        by_vid[w["video_id"]].append(w)
    out = []
    for vid in sorted(by_vid):
        frames = seqs_map[vid]
        ws = sorted(by_vid[vid], key=lambda x: w_start(x))
        for w in ws:
            s, e = w_start(w), w_end(w)
            out.extend(frames[s:e + 1])
    return out


def run_count_and_gaps(windows):
    by_vid = defaultdict(list)
    for w in windows:
        by_vid[w["video_id"]].append(w)
    run_counts = {}
    gaps = []
    for vid, ws in by_vid.items():
        ws = sorted(ws, key=lambda x: w_start(x))
        prev_end = None
        runs = 0
        for w in ws:
            s, e = w_start(w), w_end(w)
            if prev_end is None:
                runs += 1
                prev_end = e
            elif s <= prev_end + 1:
                prev_end = max(prev_end, e)
            else:
                gaps.append(int(s - prev_end - 1))
                runs += 1
                prev_end = e
        run_counts[vid] = runs
    return run_counts, gaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences_json", required=True)
    ap.add_argument("--ann_json", required=True)
    ap.add_argument("--video_list", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--split_name", required=True)
    ap.add_argument("--presence_target", type=float, required=True)
    ap.add_argument("--file_prefix", type=str, default="split")

    ap.add_argument("--l_event", type=int, default=50)
    ap.add_argument("--l_neg", type=int, default=50)
    ap.add_argument("--pre_idle_min", type=int, default=8)
    ap.add_argument("--pre_idle_max", type=int, default=16)
    ap.add_argument("--post_idle_min", type=int, default=8)
    ap.add_argument("--post_idle_max", type=int, default=16)
    ap.add_argument("--short_video_idle_min", type=int, default=4)
    ap.add_argument("--short_video_idle_max", type=int, default=8)

    ap.add_argument("--neg_eps", type=float, default=0.05)
    ap.add_argument("--presence_tolerance", type=float, default=0.05)
    ap.add_argument("--neg_step", type=int, default=5)
    ap.add_argument("--max_windows_per_video", type=int, default=10)

    ap.add_argument("--min_event_windows_sparse5", type=int, default=30)
    ap.add_argument("--min_event_windows_appx", type=int, default=10)
    ap.add_argument("--min_videos_with_events_appx", type=int, default=10)
    ap.add_argument("--min_pos_segments_total_appx", type=int, default=10)

    ap.add_argument("--pass_min_event_windows", type=int, default=80)
    ap.add_argument("--pass_min_videos_with_events", type=int, default=30)

    ap.add_argument("--target_category_id", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(int(args.seed))
    seqs = load_json(args.sequences_json)
    coco = load_json(args.ann_json)
    allowed_videos = set(load_video_list_ids(args.video_list))
    if not allowed_videos:
        print("[WARN] video_list is empty: %s" % args.video_list)
    has_target = build_has_target(coco, args.target_category_id)

    seqs_map = {}
    event_pool, neg_pool = [], []
    pos_seg_lens, neg_run_lens = [], []
    pos_seg_count_by_video = {}
    hard_case_frames = 0

    for v in seqs:
        vid = v["video_id"]
        if vid not in allowed_videos:
            continue
        frame_ids = [int(x) for x in v["frame_ids"]]
        if not frame_ids:
            continue
        seqs_map[vid] = frame_ids
        flags = presence_flags(frame_ids, has_target)

        segs = pos_segments(flags)
        pos_seg_count_by_video[vid] = len(segs)
        for (s, e) in segs:
            pos_seg_lens.append(e - s + 1)

        neg_run_lens.append(max_run_len(flags, 0))

        run = 0
        for f in flags:
            if f == 0:
                run += 1
            else:
                if run >= 6:
                    hard_case_frames += run
                run = 0
        if run >= 6:
            hard_case_frames += run

        event_pool.extend(make_event_windows(
            vid, frame_ids, flags,
            args.l_event, args.pre_idle_min, args.pre_idle_max,
            args.post_idle_min, args.post_idle_max,
            args.short_video_idle_min, args.short_video_idle_max,
            rng
        ))
        neg_pool.extend(make_neg_windows(
            vid, frame_ids, flags, args.l_neg, args.neg_eps, args.neg_step
        ))

    neg_eps_used = args.neg_eps
    if not neg_pool:
        for eps in [0.10, 0.15, 0.20, 0.30]:
            tmp = []
            for vid, frame_ids in seqs_map.items():
                flags = presence_flags(frame_ids, has_target)
                tmp.extend(make_neg_windows(vid, frame_ids, flags, args.l_neg, eps, args.neg_step))
            if tmp:
                neg_pool = tmp
                neg_eps_used = eps
                break

    split_lower = args.split_name.lower()
    is_appx5 = "sparse_5_appx" in split_lower
    is_main_sparse = any(x in split_lower for x in ["sparse_20", "sparse_15", "sparse_10"])
    is_floor_mode = any(x in split_lower for x in ["floor", "min"])

    if is_appx5:
        min_event = int(args.min_event_windows_appx)
        pass_E = 0
        pass_V = 0
    else:
        min_event = int(args.min_event_windows_sparse5) if "sparse_5" in split_lower else 0
        pass_E = int(args.pass_min_event_windows) if is_main_sparse else 0
        pass_V = int(args.pass_min_videos_with_events) if is_main_sparse else 0

    selE, selN, dbg = select_windows(
        event_pool, neg_pool,
        args.presence_target, args.presence_tolerance,
        min_event, args.max_windows_per_video, rng,
        pass_E, pass_V
    )

    selected = selE + selN

    image_ids = flatten_image_ids(seqs_map, selected)
    pos_frames = sum(has_target.get(int(i), 0) for i in image_ids)
    total_frames = len(image_ids)
    achieved = (pos_frames / total_frames) if total_frames else 0.0

    run_counts, gap_lens = run_count_and_gaps(selected)
    event_lens = [w["len_frames"] for w in selected if w["type"] == "event"]
    neg_lens = [w["len_frames"] for w in selected if w["type"] == "neg"]

    videos_with_events = sorted(list({w["video_id"] for w in selected if w["type"] == "event"}))
    num_videos_with_events = len(videos_with_events)
    num_pos_segments_total_selected = sum(int(pos_seg_count_by_video.get(v, 0)) for v in videos_with_events)

    target_effective = float(args.presence_target)
    target_clamped = False
    if is_floor_mode and achieved > (args.presence_target + args.presence_tolerance):
        target_effective = float(achieved)
        target_clamped = True

    auto = {"FAIL": [], "WARN": []}
    if abs(achieved - args.presence_target) > args.presence_tolerance:
        if target_clamped:
            auto["WARN"].append("presence_ratio target clamped to achieved floor")
        else:
            auto["FAIL"].append("presence_ratio tolerance violated")

    if is_main_sparse:
        if len(event_lens) < pass_E:
            auto["FAIL"].append("PASS_MIN: num_event_windows too small")
        if num_videos_with_events < pass_V:
            auto["FAIL"].append("PASS_MIN: num_videos_with_events too small")

    if is_appx5:
        if len(event_lens) < int(args.min_event_windows_appx):
            auto["FAIL"].append("sparse-5(appx) min_event_windows violated")
        if num_videos_with_events < int(args.min_videos_with_events_appx):
            auto["FAIL"].append("sparse-5(appx) num_videos_with_events too small")
        if num_pos_segments_total_selected < int(args.min_pos_segments_total_appx):
            auto["FAIL"].append("sparse-5(appx) num_pos_segments_total too small")
    else:
        if "sparse_5" in split_lower and len(event_lens) < int(args.min_event_windows_sparse5):
            auto["FAIL"].append("sparse-5 min_event_windows violated")

    rc_vals = list(run_counts.values())
    if rc_vals:
        if median(rc_vals) > 1:
            auto["WARN"].append("median(run_count_per_video) > 1")
        p90 = percentile(rc_vals, 90)
        if p90 is not None and p90 > 2:
            auto["WARN"].append("p90(run_count_per_video) > 2")
    if gap_lens and max(gap_lens) > 50:
        auto["WARN"].append("max(run_gap_len) > 50")

    meta = {
        "split_name": args.split_name,
        "presence_ratio_target": args.presence_target,
        "presence_ratio_requested_target": args.presence_target,
        "presence_ratio_effective_target": target_effective,
        "target_clamped": target_clamped,
        "presence_ratio_achieved": achieved,
        "presence_tolerance": args.presence_tolerance,
        "seed": args.seed,
        "num_videos_used": len(run_counts),
        "num_frames_selected": total_frames,
        "num_event_windows": len(event_lens),
        "num_neg_windows": len(neg_lens),
        "num_videos_with_events": num_videos_with_events,
        "num_pos_segments_total_selected": num_pos_segments_total_selected,
        "event_window_len_stats": stats(event_lens),
        "neg_window_len_stats": stats(neg_lens),
        "num_pos_segments_total": len(pos_seg_lens),
        "pos_segment_len_stats": stats(pos_seg_lens),
        "neg_run_len_stats_over_videos": stats(neg_run_lens),
        "run_count_per_video_stats": stats(rc_vals),
        "run_gap_len_stats": stats(gap_lens),
        "hard_case_frame_count_proxy": hard_case_frames,
        "hard_case_ratio_proxy": (hard_case_frames / total_frames) if total_frames else 0.0,
        "generator_debug": {
            "neg_eps_used": neg_eps_used,
            **dbg
        },
        "auto_checks": auto
    }

    base = os.path.join(args.out_dir, f"{args.file_prefix}_{args.split_name}")
    save_json({"image_ids": image_ids,
               "presence_ratio_target": meta["presence_ratio_target"],
               "presence_ratio_achieved": meta["presence_ratio_achieved"],
               "seed": meta["seed"]},
              base + "_image_ids.json")
    save_json([{
        "video_id": w["video_id"],
        "start_frame": w_start(w),
        "end_frame": w_end(w),
        "type": w["type"],
        "pos_frames_in_window": w.get("pos_frames_in_window", 0),
        "len_frames": w.get("len_frames", w_end(w) - w_start(w) + 1),
    } for w in selected], base + "_windows.json")
    save_json(meta, base + "_meta_report.json")

    print(f"[split={args.split_name}] achieved={achieved:.4f} target={args.presence_target:.4f} "
          f"event={len(event_lens)} videos_with_events={num_videos_with_events} FAIL={len(auto['FAIL'])}")


if __name__ == "__main__":
    main()
