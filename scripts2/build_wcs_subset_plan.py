#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
import zipfile
from collections import defaultdict
from typing import Any, Dict, List


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def load_json_from_single_zip(path: str) -> Dict[str, Any]:
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".json")]
        if len(names) != 1:
            raise ValueError("Expected exactly one json in zip: {}".format(path))
        return json.loads(z.read(names[0]))


def score_subset(
    total_images: int,
    total_empty: int,
    target_images: int,
    target_ratio: float,
    image_tolerance: int,
) -> float:
    if total_images <= 0:
        return 1e9
    ratio = float(total_empty) / float(total_images)
    ratio_err = abs(ratio - float(target_ratio))
    image_err = abs(int(total_images) - int(target_images)) / max(float(image_tolerance), 1.0)
    return ratio_err + (0.05 * image_err)


def optimize_target_bundle(
    rows: List[Dict[str, Any]],
    target_ratio: float,
    target_images: int,
    image_tolerance: int,
    min_loc_images: int,
    random_trials: int,
    seed: int,
) -> Dict[str, Any]:
    pool = [r for r in rows if int(r["num_images"]) >= int(min_loc_images)]
    if not pool:
        raise ValueError("No WCS locations satisfy min_loc_images={}".format(min_loc_images))

    best_locations: List[Dict[str, Any]] = []
    best_images = 0
    best_empty = 0
    best_score = 1e9

    for trial in range(int(random_trials)):
        rng = random.Random(int(seed) + int(trial))
        shuffled = list(pool)
        rng.shuffle(shuffled)
        shuffled.sort(
            key=lambda r: (
                abs(float(r["empty_ratio"]) - float(target_ratio)),
                abs(int(r["num_images"]) - min(int(target_images), 2000)),
            )
        )

        chosen: List[Dict[str, Any]] = []
        total_images = 0
        total_empty = 0
        for row in shuffled:
            new_images = total_images + int(row["num_images"])
            new_empty = total_empty + int(row["num_empty"])
            cur_score = score_subset(total_images, total_empty, target_images, target_ratio, image_tolerance)
            new_score = score_subset(new_images, new_empty, target_images, target_ratio, image_tolerance)
            if (
                new_score <= cur_score
                or total_images < int(target_images) * 0.85
                or len(chosen) < 4
            ):
                chosen.append(row)
                total_images = new_images
                total_empty = new_empty
            if total_images >= int(target_images) + int(image_tolerance):
                break

        improved = True
        while improved:
            improved = False
            cur_score = score_subset(total_images, total_empty, target_images, target_ratio, image_tolerance)
            for row in list(chosen):
                cand_images = total_images - int(row["num_images"])
                cand_empty = total_empty - int(row["num_empty"])
                if cand_images <= 0:
                    continue
                cand_score = score_subset(cand_images, cand_empty, target_images, target_ratio, image_tolerance)
                if cand_score < cur_score:
                    chosen.remove(row)
                    total_images = cand_images
                    total_empty = cand_empty
                    cur_score = cand_score
                    improved = True
            chosen_ids = {str(r["location"]) for r in chosen}
            for row in shuffled:
                if str(row["location"]) in chosen_ids:
                    continue
                cand_images = total_images + int(row["num_images"])
                cand_empty = total_empty + int(row["num_empty"])
                cand_score = score_subset(cand_images, cand_empty, target_images, target_ratio, image_tolerance)
                if cand_score < cur_score:
                    chosen.append(row)
                    chosen_ids.add(str(row["location"]))
                    total_images = cand_images
                    total_empty = cand_empty
                    cur_score = cand_score
                    improved = True
            for i, out_row in enumerate(list(chosen)):
                chosen_ids = {str(r["location"]) for r in chosen}
                for in_row in shuffled:
                    if str(in_row["location"]) in chosen_ids:
                        continue
                    cand_images = total_images - int(out_row["num_images"]) + int(in_row["num_images"])
                    cand_empty = total_empty - int(out_row["num_empty"]) + int(in_row["num_empty"])
                    if cand_images <= 0:
                        continue
                    cand_score = score_subset(cand_images, cand_empty, target_images, target_ratio, image_tolerance)
                    if cand_score < cur_score:
                        chosen[i] = in_row
                        total_images = cand_images
                        total_empty = cand_empty
                        cur_score = cand_score
                        improved = True
                        break

        final_score = score_subset(total_images, total_empty, target_images, target_ratio, image_tolerance)
        if final_score < best_score:
            best_locations = sorted(chosen, key=lambda r: str(r["location"]))
            best_images = int(total_images)
            best_empty = int(total_empty)
            best_score = float(final_score)

    best_nonempty = int(best_images - best_empty)
    return {
        "target_empty_ratio": float(target_ratio),
        "target_num_images": int(target_images),
        "image_tolerance": int(image_tolerance),
        "num_locations": len(best_locations),
        "locations": [str(r["location"]) for r in best_locations],
        "num_images": int(best_images),
        "num_empty": int(best_empty),
        "num_nonempty": int(best_nonempty),
        "empty_ratio": float(best_empty) / float(best_images) if best_images > 0 else 0.0,
        "score": float(best_score),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build WCS high-empty subset plan from full metadata.")
    parser.add_argument("--full-json-zip", required=True)
    parser.add_argument("--splits-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--min-images", type=int, default=500)
    parser.add_argument("--min-empty-ratio", type=float, default=0.7)
    parser.add_argument("--bundle-sizes", type=int, nargs="+", default=[3, 5, 8, 10, 11])
    parser.add_argument("--target-empty-ratios", type=float, nargs="*", default=[])
    parser.add_argument("--target-images", type=int, default=10000)
    parser.add_argument("--target-image-tolerance", type=int, default=1200)
    parser.add_argument("--target-min-loc-images", type=int, default=200)
    parser.add_argument("--target-random-trials", type=int, default=400)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    meta = load_json_from_single_zip(args.full_json_zip)
    splits = load_json(args.splits_json)
    split_locs = set(map(str, splits[args.split]))

    img_loc: Dict[str, str] = {}
    img_file: Dict[str, str] = {}
    for im in meta.get("images", []):
        loc = str(im.get("location", ""))
        if loc not in split_locs:
            continue
        iid = str(im["id"])
        img_loc[iid] = loc
        img_file[iid] = str(im["file_name"])

    stats = defaultdict(lambda: {"images": 0, "empty": 0, "nonempty": 0})
    for ann in meta.get("annotations", []):
        iid = str(ann.get("image_id", ""))
        loc = img_loc.get(iid)
        if loc is None:
            continue
        stats[loc]["images"] += 1
        if int(ann.get("category_id", -1)) == 0:
            stats[loc]["empty"] += 1
        else:
            stats[loc]["nonempty"] += 1

    rows: List[Dict[str, Any]] = []
    for loc, s in stats.items():
        images = int(s["images"])
        empty = int(s["empty"])
        nonempty = int(s["nonempty"])
        rows.append({
            "location": str(loc),
            "num_images": images,
            "num_empty": empty,
            "num_nonempty": nonempty,
            "empty_ratio": float(empty) / float(images) if images > 0 else 0.0,
        })

    rows.sort(key=lambda r: (-float(r["empty_ratio"]), -int(r["num_images"]), str(r["location"])))

    candidates = [
        r for r in rows
        if int(r["num_images"]) >= int(args.min_images) and float(r["empty_ratio"]) >= float(args.min_empty_ratio)
    ]

    bundles: List[Dict[str, Any]] = []
    for n in args.bundle_sizes:
        subset = candidates[: int(n)]
        if not subset:
            continue
        total_images = sum(int(r["num_images"]) for r in subset)
        total_empty = sum(int(r["num_empty"]) for r in subset)
        total_nonempty = sum(int(r["num_nonempty"]) for r in subset)
        bundles.append({
            "bundle_size": int(n),
            "locations": [str(r["location"]) for r in subset],
            "num_images": int(total_images),
            "num_empty": int(total_empty),
            "num_nonempty": int(total_nonempty),
            "empty_ratio": float(total_empty) / float(total_images) if total_images > 0 else 0.0,
        })

    target_bundles: List[Dict[str, Any]] = []
    for target_ratio in args.target_empty_ratios:
        target_bundles.append(
            optimize_target_bundle(
                rows=rows,
                target_ratio=float(target_ratio),
                target_images=int(args.target_images),
                image_tolerance=int(args.target_image_tolerance),
                min_loc_images=int(args.target_min_loc_images),
                random_trials=int(args.target_random_trials),
                seed=int(args.seed),
            )
        )

    summary = {
        "split": str(args.split),
        "min_images": int(args.min_images),
        "min_empty_ratio": float(args.min_empty_ratio),
        "target_images": int(args.target_images),
        "target_image_tolerance": int(args.target_image_tolerance),
        "num_split_locations": len(rows),
        "split_totals": {
            "num_images": int(sum(int(r["num_images"]) for r in rows)),
            "num_empty": int(sum(int(r["num_empty"]) for r in rows)),
            "num_nonempty": int(sum(int(r["num_nonempty"]) for r in rows)),
            "empty_ratio": (
                float(sum(int(r["num_empty"]) for r in rows)) / float(sum(int(r["num_images"]) for r in rows))
                if rows else 0.0
            ),
        },
        "num_candidate_locations": len(candidates),
        "candidate_locations": candidates,
        "bundles": bundles,
        "target_bundles": target_bundles,
    }

    with open(os.path.join(args.out_dir, "wcs_subset_plan_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    for bundle in bundles:
        tag = "top{}_locs".format(int(bundle["bundle_size"]))
        txt_path = os.path.join(args.out_dir, "{}.txt".format(tag))
        with open(txt_path, "w") as f:
            for loc in bundle["locations"]:
                f.write(str(loc) + "\n")

    for bundle in target_bundles:
        target_tag = int(round(float(bundle["target_empty_ratio"]) * 100.0))
        txt_path = os.path.join(args.out_dir, "target{:02d}_locs.txt".format(target_tag))
        with open(txt_path, "w") as f:
            for loc in bundle["locations"]:
                f.write(str(loc) + "\n")

    print("Saved:", os.path.join(args.out_dir, "wcs_subset_plan_summary.json"))
    for bundle in bundles:
        print("Saved:", os.path.join(args.out_dir, "top{}_locs.txt".format(int(bundle["bundle_size"]))))
    for bundle in target_bundles:
        print("Saved:", os.path.join(args.out_dir, "target{:02d}_locs.txt".format(int(round(float(bundle["target_empty_ratio"]) * 100.0)))))


if __name__ == "__main__":
    main()
