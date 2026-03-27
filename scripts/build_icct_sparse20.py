#!/usr/bin/env python3
"""build_icct_sparse20.py

Minimal-change pipeline: Convert ICCT COCO into an FBD-compatible dataset root.

Goal
- Produce:
    <out_root>/annotations/fbdsv24_vid_dev.json
    <out_root>/annotations/fbdsv24_vid_dev_sequences.json
    <out_root>/annotations/fbdsv24_vid_holdout.json
    <out_root>/annotations/fbdsv24_vid_holdout_sequences.json
  plus dev/holdout location/video id lists and a split_report.

- Store images at:
    <out_root>/images/<original_file_name>
  and set rel_path in sequences (and images[i].file_name in COCO) to:
    images/<original_file_name>

This script is Python 3.9 compatible.
"""

import argparse
import json
import os
import random
import shutil
from typing import Any, Dict, List, Tuple, Optional, Hashable


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _write_json(path: str, obj: Any) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _parse_file_name(file_name: str) -> Tuple[str, str, str]:
    """Return (location, session_relpath, image_relpath).

    Expected file_name patterns (both appear in ICCT dumps):

    A) <country>/<camera>/<session>/<file>
    B) <country>/<camera>/<file>

    - location = "{country}_{camera}"
    - session_relpath = "{country}/{camera}/{session}"
    - image_relpath = original file_name
    """
    parts = file_name.split("/")
    if len(parts) < 3:
        raise ValueError("Unexpected file_name format: %s" % file_name)

    country = parts[0]
    camera = parts[1]

    # Some locations do not have a dedicated session folder.
    # In that case we treat the camera folder as the session unit.
    if len(parts) >= 4:
        session = parts[2]
        session_relpath = "%s/%s/%s" % (country, camera, session)
    else:
        session_relpath = "%s/%s" % (country, camera)

    location = "%s_%s" % (country, camera)
    return location, session_relpath, file_name


def normalize_coco_to_one_class(coco: Dict[str, Any], class_name: str) -> Dict[str, Any]:
    """Normalize ICCT COCO into a single-class COCO with stable numeric ids.

    Design-team rule:
      - Remove 'empty' category from categories.
      - Delete annotations whose category_id belongs to 'empty'.
      - Delete annotations whose category_id belongs to human/person (privacy; images may be absent).
      - Remaining annotations are treated as 'animal' (category_id=1).

    Critical compatibility rule:
      - Enforce COCO invariant: every annotations[*].image_id must reference images[*].id.
      - We assign stable numeric image ids based on file_name ordering, then rewrite annotation.image_id accordingly.
      - If an annotation refers to an unknown image_id, we drop it and report counts (fail-fast if any remain).

    Output:
      - images.id: 1..N stable (sorted by file_name)
      - annotations.image_id: uses the remapped images.id
    """
    categories = coco.get("categories", [])
    empty_ids = set()
    human_ids = set()

    for cat in categories:
        name = str(cat.get("name", "")).strip().lower()
        if name == "empty" or name == "blank" or ("empty" in name):
            cid = cat.get("id")
            if cid is not None:
                empty_ids.add(cid)

    for cat in categories:
        name = str(cat.get("name", "")).strip().lower()
        if name == "human" or name == "person" or ("human" in name) or ("person" in name):
            cid = cat.get("id")
            if cid is not None:
                human_ids.add(cid)

    drop_cat_ids = empty_ids.union(human_ids)

    images = coco.get("images", [])
    anns = coco.get("annotations", [])

    # Build mapping from old image id -> file_name
    old_id_to_file: Dict[Hashable, str] = {}
    file_names: List[str] = []
    for img in images:
        old_id = img.get("id")
        fn = img.get("file_name", None)
        if old_id is None or not isinstance(fn, str):
            continue
        old_id_to_file[old_id] = fn
        file_names.append(fn)

    # Stable new numeric ids by sorted file_name
    file_names_sorted = sorted(set(file_names))
    file_to_new_id: Dict[str, int] = {}
    for i, fn in enumerate(file_names_sorted, start=1):
        file_to_new_id[fn] = i

    # Remap images
    new_images: List[Dict[str, Any]] = []
    missing_img_meta = 0
    for img in images:
        old_id = img.get("id")
        fn = img.get("file_name", None)
        if old_id is None or not isinstance(fn, str):
            missing_img_meta += 1
            continue
        if fn not in file_to_new_id:
            missing_img_meta += 1
            continue
        img_rec = dict(img)
        img_rec["id"] = int(file_to_new_id[fn])
        new_images.append(img_rec)

    # Remap annotations
    new_annotations: List[Dict[str, Any]] = []
    dropped_ref_missing = 0
    dropped_category = 0
    next_ann_id = 1

    for ann in anns:
        if ann.get("category_id") in drop_cat_ids:
            dropped_category += 1
            continue

        old_img_id = ann.get("image_id", None)
        if old_img_id is None or old_img_id not in old_id_to_file:
            dropped_ref_missing += 1
            continue

        fn = old_id_to_file[old_img_id]
        if fn not in file_to_new_id:
            dropped_ref_missing += 1
            continue

        ann_rec = dict(ann)
        ann_rec["id"] = int(next_ann_id)
        ann_rec["image_id"] = int(file_to_new_id[fn])
        ann_rec["category_id"] = 1

        # COCOeval stability fields
        if "iscrowd" not in ann_rec:
            ann_rec["iscrowd"] = 0
        if "area" not in ann_rec:
            bbox = ann_rec.get("bbox", None)
            if isinstance(bbox, list) and len(bbox) == 4:
                bw = float(bbox[2]);
                bh = float(bbox[3])
                ann_rec["area"] = max(0.0, bw) * max(0.0, bh)
            else:
                ann_rec["area"] = 0.0

        new_annotations.append(ann_rec)
        next_ann_id += 1

    # Validate COCO invariant
    img_id_set = set(int(im["id"]) for im in new_images if "id" in im)
    ann_img_set = set(int(a["image_id"]) for a in new_annotations if "image_id" in a)
    num_missing_refs = len(list(ann_img_set - img_id_set))

    out = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": new_images,
        "annotations": new_annotations,
        "categories": [{"id": 1, "name": class_name}],
        "_icct_meta": {
            "removed_empty_category_ids": sorted(list(empty_ids)),
            "removed_human_category_ids": sorted(list(human_ids)),
            "num_images_in": len(images),
            "num_images_out": len(new_images),
            "num_annotations_in": len(anns),
            "num_annotations_out": len(new_annotations),
            "dropped_ann_category": int(dropped_category),
            "dropped_ann_ref_missing": int(dropped_ref_missing),
            "missing_img_meta": int(missing_img_meta),
            "coco_invariant_missing_image_ids_count": int(num_missing_refs),
        },
    }

    if num_missing_refs != 0:
        raise ValueError(
            "COCO invariant violated after normalize: annotations reference missing images. count=%d"
            % num_missing_refs
        )

    return out


def filter_missing_images_and_anns(
        coco_1class: Dict[str, Any],
        icct_image_root: str,
        skip_missing: int,
) -> Tuple[Dict[str, Any], Dict[int, str]]:
    """Filter COCO images/annotations by checking source file existence."""
    images = coco_1class.get("images", [])
    anns = coco_1class.get("annotations", [])

    keep_img_ids = set()
    id_to_fn: Dict[int, str] = {}

    missing = 0
    for im in images:
        try:
            iid = int(im.get("id"))
        except Exception:
            continue
        fn = im.get("file_name", None)
        if not isinstance(fn, str):
            continue
        id_to_fn[iid] = fn

        if int(skip_missing) == 1:
            rel = fn
            if rel.startswith("images/"):
                rel = rel[len("images/"):]
            src = os.path.join(icct_image_root, rel)
            if not os.path.exists(src):
                missing += 1
                continue

        keep_img_ids.add(iid)

    new_images = [im for im in images if int(im.get("id", -1)) in keep_img_ids]
    new_anns = [a for a in anns if int(a.get("image_id", -1)) in keep_img_ids]

    out = dict(coco_1class)
    out["images"] = new_images
    out["annotations"] = new_anns

    meta = dict(out.get("_icct_meta", {}))
    meta["skip_missing_enabled"] = int(skip_missing)
    meta["missing_images_dropped_pre_split"] = int(missing)
    meta["num_images_after_missing"] = int(len(new_images))
    meta["num_annotations_after_missing"] = int(len(new_anns))
    out["_icct_meta"] = meta

    return out, id_to_fn


def drop_clips_by_min_len(
        clip_to_images: Dict[str, List[Dict[str, Any]]],
        video_to_location: Dict[str, str],
        min_len: int,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, str], Dict[str, Any]]:
    """Drop clips whose remaining frames < min_len."""
    if int(min_len) <= 0:
        return clip_to_images, video_to_location, {"min_len": int(min_len), "dropped_clips": 0,
                                                   "kept_clips": int(len(clip_to_images))}

    kept: Dict[str, List[Dict[str, Any]]] = {}
    kept_loc: Dict[str, str] = {}
    dropped = 0
    for vid, imgs in clip_to_images.items():
        if len(imgs) < int(min_len):
            dropped += 1
            continue
        kept[vid] = imgs
        kept_loc[vid] = video_to_location.get(vid, "")
    return kept, kept_loc, {"min_len": int(min_len), "dropped_clips": int(dropped), "kept_clips": int(len(kept))}


def build_image_pos_counts(coco: Dict[str, Any]) -> Dict[Hashable, int]:
    """Return image_id -> number of annotations (bbox) in that image.

    Note: ICCT uses string ids for both images and annotations.image_id.
    We therefore treat ids as hashable keys without casting to int.
    """
    counts: Dict[Hashable, int] = {}
    for ann in coco.get("annotations", []):
        img_id = ann.get("image_id")
        if img_id is None:
            continue
        counts[img_id] = counts.get(img_id, 0) + 1
    return counts


def group_images_into_clips(coco: Dict[str, Any]) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, str]]:
    """Group image dicts into clips by session_relpath.

    Returns:
      clip_to_images: video_id -> list of image dicts
      video_to_location: video_id -> location
    """
    clip_to_images: Dict[str, List[Dict[str, Any]]] = {}
    video_to_location: Dict[str, str] = {}

    for img in coco.get("images", []):
        file_name = img.get("file_name", None)
        if not isinstance(file_name, str):
            continue

        location, session_relpath, _ = _parse_file_name(file_name)
        video_id = "%s__%s" % (location, session_relpath)

        clip_to_images.setdefault(video_id, []).append(img)
        video_to_location[video_id] = location

    return clip_to_images, video_to_location


def compute_clip_presence(clip_to_images: Dict[str, List[Dict[str, Any]]], image_pos_counts: Dict[Hashable, int]) -> \
        Dict[str, float]:
    """Return video_id -> presence ratio r(c) = pos_frames / all_frames."""
    ratios: Dict[str, float] = {}
    for vid, imgs in clip_to_images.items():
        total = 0
        pos = 0
        for img in imgs:
            img_id = img.get("id")
            if img_id is None:
                continue
            total += 1
            if image_pos_counts.get(img_id, 0) > 0:
                pos += 1
        ratios[vid] = (float(pos) / float(total)) if total > 0 else 0.0
    return ratios


def build_clip_presence_table(
        clip_to_images: Dict[str, List[Dict[str, Any]]],
        video_to_location: Dict[str, str],
        image_pos_counts: Dict[Hashable, int],
) -> List[Dict[str, Any]]:
    """Build a per-clip stats table for debugging and reporting.

    Fields:
      - video_id
      - location
      - num_frames
      - num_pos_frames
      - presence_ratio
      - session_relpath
    """
    rows: List[Dict[str, Any]] = []
    for vid, imgs in clip_to_images.items():
        total = 0
        pos = 0
        for img in imgs:
            img_id = img.get("id")
            if img_id is None:
                continue
            total += 1
            if image_pos_counts.get(img_id, 0) > 0:
                pos += 1

        ratio = (float(pos) / float(total)) if total > 0 else 0.0
        loc = video_to_location.get(vid, "")

        # video_id format: "{location}__{country}/{camera}/{session}"
        session_relpath = ""
        if "__" in vid:
            session_relpath = vid.split("__", 1)[1]

        rows.append(
            {
                "video_id": vid,
                "location": loc,
                "session_relpath": session_relpath,
                "num_frames": total,
                "num_pos_frames": pos,
                "presence_ratio": ratio,
            }
        )

    # Sort by presence then size for easy inspection
    rows.sort(key=lambda r: (float(r.get("presence_ratio", 0.0)), -int(r.get("num_frames", 0))))
    return rows


def filter_clips_by_presence(clip_to_images: Dict[str, List[Dict[str, Any]]], clip_presence: Dict[str, float],
                             threshold: float) -> Dict[str, List[Dict[str, Any]]]:
    kept: Dict[str, List[Dict[str, Any]]] = {}
    for vid, imgs in clip_to_images.items():
        if clip_presence.get(vid, 1.0) <= threshold:
            kept[vid] = imgs
    return kept


def split_locations_by_frames(
        kept_clip_to_images: Dict[str, List[Dict[str, Any]]],
        video_to_location: Dict[str, str],
        seed: int,
        dev_frac: float,
) -> Tuple[List[str], List[str]]:
    """Leakage-free split: assign entire locations to dev or holdout.

    We try to match dev_frac by total frames.
    """
    loc_frames: Dict[str, int] = {}
    for vid, imgs in kept_clip_to_images.items():
        loc = video_to_location[vid]
        loc_frames[loc] = loc_frames.get(loc, 0) + len(imgs)

    locations = list(loc_frames.keys())
    locations.sort(key=lambda x: loc_frames.get(x, 0), reverse=True)
    rng = random.Random(seed)
    i = 0
    while i < len(locations):
        j = i + 1
        while j < len(locations) and loc_frames.get(locations[j], 0) == loc_frames.get(locations[i], 0):
            j += 1
        if j - i > 1:
            sub = locations[i:j]
            rng.shuffle(sub)
            locations[i:j] = sub
        i = j

    total_frames = sum(loc_frames.values())
    target_dev_frames = int(dev_frac * float(total_frames))

    dev_locs: List[str] = []
    hold_locs: List[str] = []

    dev_frames = 0
    for loc in locations:
        if dev_frames < target_dev_frames:
            dev_locs.append(loc)
            dev_frames += loc_frames[loc]
        else:
            hold_locs.append(loc)

    # Ensure non-empty splits
    if len(dev_locs) == 0 and len(hold_locs) > 0:
        dev_locs.append(hold_locs.pop(0))
    if len(hold_locs) == 0 and len(dev_locs) > 0:
        hold_locs.append(dev_locs.pop(-1))

    return dev_locs, hold_locs


def collect_split_video_ids(kept_clip_to_images: Dict[str, List[Dict[str, Any]]], video_to_location: Dict[str, str],
                            dev_locs: List[str], hold_locs: List[str]) -> Tuple[List[str], List[str]]:
    dev_set = set(dev_locs)
    hold_set = set(hold_locs)

    dev_vids: List[str] = []
    hold_vids: List[str] = []

    for vid in kept_clip_to_images.keys():
        loc = video_to_location[vid]
        if loc in dev_set:
            dev_vids.append(vid)
        elif loc in hold_set:
            hold_vids.append(vid)

    dev_vids.sort()
    hold_vids.sort()
    return dev_vids, hold_vids


def export_split_coco_and_sequences(
        coco_1class: Dict[str, Any],
        kept_clip_to_images: Dict[str, List[Dict[str, Any]]],
        dev_vids: List[str],
        hold_vids: List[str],
        out_root: str,
) -> None:
    """Write FBD-compatible dev/holdout COCO and sequences."""

    ann_root = os.path.join(out_root, "annotations")
    _ensure_dir(ann_root)

    # Build quick indexes
    img_by_id: Dict[int, Dict[str, Any]] = {}
    for img in coco_1class.get("images", []):
        try:
            img_id = int(img.get("id"))
        except Exception:
            continue
        img_by_id[img_id] = img

    anns_by_img: Dict[int, List[Dict[str, Any]]] = {}
    for ann in coco_1class.get("annotations", []):
        try:
            img_id = int(ann.get("image_id"))
        except Exception:
            continue
        anns_by_img.setdefault(img_id, []).append(ann)

    def _build_split(vids: List[str]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        split_img_ids: List[int] = []
        split_images: List[Dict[str, Any]] = []
        split_anns: List[Dict[str, Any]] = []
        sequences: List[Dict[str, Any]] = []

        for vid in vids:
            imgs = kept_clip_to_images.get(vid, [])

            # Sort frames by file_name (stable and available)
            imgs_sorted = sorted(imgs, key=lambda x: str(x.get("file_name", "")))

            frames_list: List[Dict[str, Any]] = []
            for img in imgs_sorted:
                try:
                    img_id = int(img.get("id"))
                except Exception:
                    continue

                if img_id not in img_by_id:
                    continue

                rel_file = str(img_by_id[img_id].get("file_name"))
                rel_path = os.path.join("images", rel_file)

                # Copy image record but rewrite file_name to be data_root-relative
                img_rec = dict(img_by_id[img_id])
                img_rec["file_name"] = rel_path

                split_images.append(img_rec)
                split_img_ids.append(img_id)

                for ann in anns_by_img.get(img_id, []):
                    split_anns.append(ann)

                frames_list.append({"image_id": img_id, "rel_path": rel_path})

            sequences.append({"video_id": vid, "frames": frames_list})

        # Deduplicate images by id (defensive)
        seen = set()
        uniq_images: List[Dict[str, Any]] = []
        for img in split_images:
            try:
                img_id = int(img.get("id"))
            except Exception:
                continue
            if img_id in seen:
                continue
            seen.add(img_id)
            uniq_images.append(img)

        out_coco = {
            "info": coco_1class.get("info", {}),
            "licenses": coco_1class.get("licenses", []),
            "images": uniq_images,
            "annotations": split_anns,
            "categories": coco_1class.get("categories", [{"id": 1, "name": "animal"}]),
        }
        return out_coco, sequences

    dev_coco, dev_seq = _build_split(dev_vids)
    hold_coco, hold_seq = _build_split(hold_vids)

    _write_json(os.path.join(ann_root, "fbdsv24_vid_dev.json"), dev_coco)
    _write_json(os.path.join(ann_root, "fbdsv24_vid_dev_sequences.json"), dev_seq)
    _write_json(os.path.join(ann_root, "fbdsv24_vid_holdout.json"), hold_coco)
    _write_json(os.path.join(ann_root, "fbdsv24_vid_holdout_sequences.json"), hold_seq)


def materialize_images(
        coco_split: Dict[str, Any],
        icct_image_root: str,
        out_root: str,
        link_mode: str,
) -> None:
    """Create images under out_root/images/ matching COCO image file_name entries."""
    for img in coco_split.get("images", []):
        rel_path = img.get("file_name", None)
        if not isinstance(rel_path, str):
            continue
        if not rel_path.startswith("images/"):
            continue

        orig_rel = rel_path[len("images/"):]
        src = os.path.join(icct_image_root, orig_rel)
        dst = os.path.join(out_root, rel_path)

        _ensure_dir(os.path.dirname(dst))

        if not os.path.exists(src):
            continue

        if os.path.exists(dst):
            continue

        if link_mode == "symlink":
            os.symlink(os.path.abspath(src), dst)
        elif link_mode == "copy":
            try:
                shutil.copy2(src, dst)
            except FileNotFoundError:
                try:
                    shutil.copyfile(src, dst)
                except Exception:
                    continue
            except Exception:
                try:
                    shutil.copyfile(src, dst)
                except Exception:
                    continue
        elif link_mode == "none":
            pass
        else:
            raise ValueError("Unknown link_mode: %s" % link_mode)


def build_split_report(
        kept_clip_to_images: Dict[str, List[Dict[str, Any]]],
        clip_presence: Dict[str, float],
        video_to_location: Dict[str, str],
        dev_locs: List[str],
        hold_locs: List[str],
        dev_vids: List[str],
        hold_vids: List[str],
) -> Dict[str, Any]:
    def _summarize(vids: List[str], locs: List[str]) -> Dict[str, Any]:
        total_frames = 0
        total_pos = 0
        r_list: List[float] = []

        for vid in vids:
            imgs = kept_clip_to_images.get(vid, [])
            total_frames += len(imgs)
            r = float(clip_presence.get(vid, 0.0))
            r_list.append(r)
            total_pos += int(round(r * float(len(imgs))))

        overall_presence = (float(total_pos) / float(total_frames)) if total_frames > 0 else 0.0

        r_list_sorted = sorted(r_list)

        def _pct(p: float) -> float:
            if not r_list_sorted:
                return 0.0
            idx = int(p * float(len(r_list_sorted) - 1))
            return float(r_list_sorted[idx])

        return {
            "num_locations": len(locs),
            "num_clips": len(vids),
            "num_frames": total_frames,
            "overall_presence": overall_presence,
            "clip_presence_p25": _pct(0.25),
            "clip_presence_p50": _pct(0.50),
            "clip_presence_p75": _pct(0.75),
        }

    return {
        "definition": {
            "clip": "session folder (lowest-level directory in images[i].file_name)",
            "location": "<country>_<camera>",
            "video_id": "{location}__{country}/{camera}/{session}",
            "positive_frame": "image with >=1 bbox",
            "negative_frame": "image with 0 bbox",
            "sparse_threshold": "r(c) <= threshold",
        },
        "dev": _summarize(dev_vids, dev_locs),
        "holdout": _summarize(hold_vids, hold_locs),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--icct_coco_json", required=True)
    parser.add_argument("--icct_image_root", required=True)
    parser.add_argument("--out_root", required=True)

    parser.add_argument("--presence_threshold", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev_frac", type=float, default=0.70)

    parser.add_argument("--skip_missing", type=int, default=1, choices=[0, 1],
                        help="If 1, drop frames whose source file does not exist BEFORE any clip stats/split.")
    parser.add_argument("--min_len", type=int, default=10,
                        help="After missing removal, drop clips with remaining frames < min_len (0 disables).")

    parser.add_argument("--link_mode", choices=["symlink", "copy", "none"], default="symlink")
    parser.add_argument("--stop_after", choices=["normalize", "index", "filter", "split", "export"], default="export")

    args = parser.parse_args()

    out_root = os.path.abspath(args.out_root)
    ann_root = os.path.join(out_root, "annotations")
    _ensure_dir(ann_root)

    # Stage 1: normalize
    coco = _read_json(args.icct_coco_json)
    coco_1class = normalize_coco_to_one_class(coco, class_name="animal")
    coco_1class, _id_to_fn = filter_missing_images_and_anns(
        coco_1class=coco_1class,
        icct_image_root=args.icct_image_root,
        skip_missing=int(args.skip_missing),
    )

    _write_json(os.path.join(ann_root, "icct_1class_total.json"), coco_1class)
    if args.stop_after == "normalize":
        print("[stop_after=normalize] wrote:", os.path.join(ann_root, "icct_1class_total.json"))
        return

    # Stage 2-3: clip grouping and filter
    image_pos_counts = build_image_pos_counts(coco_1class)
    clip_to_images, video_to_location = group_images_into_clips(coco_1class)
    clip_to_images, video_to_location, minlen_report = drop_clips_by_min_len(
        clip_to_images=clip_to_images,
        video_to_location=video_to_location,
        min_len=int(args.min_len),
    )
    clip_presence = compute_clip_presence(clip_to_images, image_pos_counts)

    clip_table = build_clip_presence_table(clip_to_images, video_to_location, image_pos_counts)
    _write_json(os.path.join(ann_root, "icct_clip_presence.json"), clip_table)

    if args.stop_after == "index":
        print("[stop_after=index] wrote:", os.path.join(ann_root, "icct_clip_presence.json"))
        return

    kept_clip_to_images = filter_clips_by_presence(clip_to_images, clip_presence, args.presence_threshold)

    kept_index = {
        "presence_threshold": args.presence_threshold,
        "num_clips_all": len(clip_to_images),
        "num_clips_kept": len(kept_clip_to_images),
    }
    _write_json(os.path.join(ann_root, "icct_sparse_clips_index.json"), kept_index)

    if args.stop_after == "filter":
        print("[stop_after=filter] wrote:", os.path.join(ann_root, "icct_sparse_clips_index.json"))
        return

    # Stage 4: split
    dev_locs, hold_locs = split_locations_by_frames(kept_clip_to_images, video_to_location, seed=args.seed,
                                                    dev_frac=args.dev_frac)
    dev_vids, hold_vids = collect_split_video_ids(kept_clip_to_images, video_to_location, dev_locs, hold_locs)

    _write_json(os.path.join(ann_root, "dev_locations.json"), dev_locs)
    _write_json(os.path.join(ann_root, "holdout_locations.json"), hold_locs)
    _write_json(os.path.join(ann_root, "dev_videos.json"), dev_vids)
    _write_json(os.path.join(ann_root, "holdout_videos.json"), hold_vids)

    if args.stop_after == "split":
        print("[stop_after=split] wrote location/video lists under:", ann_root)
        return

    # Stage 5: export COCO+sequences
    export_split_coco_and_sequences(
        coco_1class=coco_1class,
        kept_clip_to_images=kept_clip_to_images,
        dev_vids=dev_vids,
        hold_vids=hold_vids,
        out_root=out_root,
    )

    # Materialize images (symlink/copy)
    dev_coco = _read_json(os.path.join(ann_root, "fbdsv24_vid_dev.json"))
    hold_coco = _read_json(os.path.join(ann_root, "fbdsv24_vid_holdout.json"))

    materialize_images(dev_coco, args.icct_image_root, out_root, link_mode=args.link_mode)
    materialize_images(hold_coco, args.icct_image_root, out_root, link_mode=args.link_mode)

    report = build_split_report(
        kept_clip_to_images=kept_clip_to_images,
        clip_presence=clip_presence,
        video_to_location=video_to_location,
        dev_locs=dev_locs,
        hold_locs=hold_locs,
        dev_vids=dev_vids,
        hold_vids=hold_vids,
    )
    report["missing"] = coco_1class.get("_icct_meta", {})
    report["min_len"] = minlen_report if "minlen_report" in locals() else {"min_len": int(args.min_len)}
    _write_json(os.path.join(ann_root, "split_report.json"), report)

    print("done")
    print("out_root:", out_root)
    print("dev clips:", len(dev_vids), "holdout clips:", len(hold_vids))


if __name__ == "__main__":
    main()
