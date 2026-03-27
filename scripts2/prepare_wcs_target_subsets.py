#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import zipfile
from typing import Any, Dict, List, Set


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


def save_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def write_lines(path: str, lines: List[str]) -> None:
    with open(path, "w") as f:
        for line in lines:
            f.write(str(line) + "\n")


def rel_symlink(src: str, dst: str) -> None:
    ensure_dir(os.path.dirname(dst))
    if os.path.lexists(dst):
        return
    rel_src = os.path.relpath(src, os.path.dirname(dst))
    os.symlink(rel_src, dst)


def subset_tag_from_ratio(target_ratio: float) -> str:
    return "target{:02d}".format(int(round(float(target_ratio) * 100.0)))


def build_download_script(script_path: str, root_dir: str, raw_image_root: str, union_list_rel: str) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        'ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'RAW_IMAGE_ROOT="{}"'.format(raw_image_root),
        'LIST_FILE="$ROOT_DIR/{}"'.format(union_list_rel),
        'mkdir -p "$RAW_IMAGE_ROOT"',
        'cat "$LIST_FILE" | xargs -I{} -P 8 bash -lc \'rel=\"$1\"; dst=\"$2/$1\"; mkdir -p "$(dirname "$dst")"; aws s3 cp --no-sign-request "s3://us-west-2.opendata.source.coop/agentmorris/lila-wildlife/wcs-unzipped/$rel" "$dst" >/dev/null\' _ {} "$RAW_IMAGE_ROOT"',
        'echo "Download complete: $LIST_FILE"',
    ]
    with open(script_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(script_path, 0o755)


def build_parent_sync_script(script_path: str, raw_image_root: str, parent_list_rel: str) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        'ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'RAW_IMAGE_ROOT="{}"'.format(raw_image_root),
        'LIST_FILE="$ROOT_DIR/{}"'.format(parent_list_rel),
        'mkdir -p "$RAW_IMAGE_ROOT"',
        'cat "$LIST_FILE" | xargs -I{} -P 6 bash -lc \'rel=\"$1\"; dst=\"$2/$1\"; mkdir -p "$dst"; aws s3 sync --no-sign-request "s3://us-west-2.opendata.source.coop/agentmorris/lila-wildlife/wcs-unzipped/$rel" "$dst" >/dev/null\' _ {} "$RAW_IMAGE_ROOT"',
        'echo "Parent sync complete: $LIST_FILE"',
    ]
    with open(script_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(script_path, 0o755)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare WCS target-ratio subset folders and missing lists.")
    parser.add_argument("--subset-plan-json", required=True)
    parser.add_argument("--full-json-zip", required=True)
    parser.add_argument("--splits-json", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--raw-image-root", required=True)
    parser.add_argument("--subset-root", required=True)
    parser.add_argument("--target-tags", nargs="*", default=["target20", "target15", "target10", "target05"])
    args = parser.parse_args()

    ensure_dir(args.subset_root)
    plan = load_json(args.subset_plan_json)
    meta = load_json_from_single_zip(args.full_json_zip)
    splits = load_json(args.splits_json)
    split_locs = set(map(str, splits[args.split]))
    availability_mode = str(plan.get("availability_mode", "all_split_images"))

    bundles_by_tag = {
        subset_tag_from_ratio(float(bundle["target_empty_ratio"])): bundle
        for bundle in plan.get("target_bundles", [])
    }
    selected_tags = [tag for tag in args.target_tags if tag in bundles_by_tag]
    if not selected_tags:
        raise ValueError("No target tags matched in subset plan.")

    images = [im for im in meta.get("images", []) if str(im.get("location", "")) in split_locs]
    image_ids_in_split = {str(im["id"]) for im in images}
    annotations = [ann for ann in meta.get("annotations", []) if str(ann.get("image_id", "")) in image_ids_in_split]
    categories = meta.get("categories", [])
    info = meta.get("info", {})
    licenses = meta.get("licenses", [])

    union_missing: Set[str] = set()
    root_summary: Dict[str, Any] = {"subsets": []}

    for tag in selected_tags:
        bundle = bundles_by_tag[tag]
        target_dir = os.path.join(args.subset_root, tag)
        lists_dir = os.path.join(target_dir, "lists")
        ann_dir = os.path.join(target_dir, "annotations")
        status_dir = os.path.join(target_dir, "status")
        image_dir = os.path.join(target_dir, "images")
        for d in [lists_dir, ann_dir, status_dir, image_dir]:
            ensure_dir(d)

        locs = set(map(str, bundle["locations"]))
        subset_images = [im for im in images if str(im.get("location", "")) in locs]
        if availability_mode == "existing_raw_only":
            subset_images = [
                im for im in subset_images
                if os.path.isfile(os.path.join(args.raw_image_root, str(im["file_name"])))
            ]
        subset_image_ids = {str(im["id"]) for im in subset_images}
        subset_anns = [ann for ann in annotations if str(ann["image_id"]) in subset_image_ids]
        subset_manifest = {
            "info": info,
            "licenses": licenses,
            "images": subset_images,
            "annotations": subset_anns,
            "categories": categories,
        }
        save_json(os.path.join(ann_dir, "manifest_coco_bbox.json"), subset_manifest)

        location_lines = sorted(locs, key=lambda x: (len(x), x))
        all_files = sorted(str(im["file_name"]) for im in subset_images)
        existing_files: List[str] = []
        missing_files: List[str] = []
        for rel_path in all_files:
            src_path = os.path.join(args.raw_image_root, rel_path)
            dst_path = os.path.join(image_dir, rel_path)
            if os.path.isfile(src_path):
                rel_symlink(src_path, dst_path)
                existing_files.append(rel_path)
            else:
                missing_files.append(rel_path)
                union_missing.add(rel_path)

        write_lines(os.path.join(lists_dir, "locations.txt"), location_lines)
        write_lines(os.path.join(lists_dir, "all_files.txt"), all_files)
        write_lines(os.path.join(lists_dir, "existing_files.txt"), existing_files)
        write_lines(os.path.join(lists_dir, "missing_files.txt"), missing_files)

        summary = {
            "tag": tag,
            "target_empty_ratio": bundle["target_empty_ratio"],
            "actual_empty_ratio": bundle["empty_ratio"],
            "target_num_images": bundle["target_num_images"],
            "num_locations": len(location_lines),
            "num_images": len(all_files),
            "num_existing_files": len(existing_files),
            "num_missing_files": len(missing_files),
            "existing_ratio": float(len(existing_files)) / float(len(all_files)) if all_files else 0.0,
            "locations": location_lines,
        }
        save_json(os.path.join(status_dir, "subset_status.json"), summary)
        root_summary["subsets"].append(summary)

    union_list = sorted(union_missing)
    union_parents = sorted({os.path.dirname(p) for p in union_list})
    union_list_rel = "union_missing_files.txt"
    write_lines(os.path.join(args.subset_root, union_list_rel), union_list)
    parent_list_rel = "union_missing_parent_dirs.txt"
    write_lines(os.path.join(args.subset_root, parent_list_rel), union_parents)
    build_download_script(
        script_path=os.path.join(args.subset_root, "download_union_missing.sh"),
        root_dir=args.subset_root,
        raw_image_root=args.raw_image_root,
        union_list_rel=union_list_rel,
    )
    build_parent_sync_script(
        script_path=os.path.join(args.subset_root, "sync_union_parent_dirs.sh"),
        raw_image_root=args.raw_image_root,
        parent_list_rel=parent_list_rel,
    )
    root_summary["union_missing_files"] = len(union_list)
    root_summary["union_missing_parent_dirs"] = len(union_parents)
    root_summary["target_tags"] = selected_tags
    save_json(os.path.join(args.subset_root, "subset_root_status.json"), root_summary)
    print("Prepared subset root:", args.subset_root)
    for entry in root_summary["subsets"]:
        print(
            "{} images={} existing={} missing={}".format(
                entry["tag"], entry["num_images"], entry["num_existing_files"], entry["num_missing_files"]
            )
        )
    print("Union missing files:", len(union_list))


if __name__ == "__main__":
    main()
