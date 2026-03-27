#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from typing import Any, Dict, List, Optional


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def infer_single_category_id_from_gt(gt_json_path: str) -> Optional[int]:
    """
    Infer target category_id from GT COCO.
    If categories has exactly one entry with an integer id, return it.
    Otherwise return None.
    """
    obj = load_json(gt_json_path)
    cats = obj.get("categories", [])
    if not isinstance(cats, list):
        return None
    if len(cats) != 1:
        return None
    cid = cats[0].get("id", None)
    try:
        return int(cid)
    except Exception:
        return None


def find_category_id_by_name(gt_json_path: str, name: str) -> Optional[int]:
    """
    Find category id by (case-insensitive) category name in GT COCO.
    Return None if not found.
    """
    obj = load_json(gt_json_path)
    cats = obj.get("categories", [])
    if not isinstance(cats, list):
        return None
    name_l = str(name).strip().lower()
    for c in cats:
        n = str(c.get("name", "")).strip().lower()
        if n == name_l:
            try:
                return int(c.get("id", None))
            except Exception:
                return None
    return None


def resolve_target_category_id(gt_json_path: str, target_category_id: Optional[int], target_category_name: Optional[str]) -> int:
    """
    Resolve target category id with the following priority:
      1) target_category_id if provided
      2) target_category_name if provided and found in gt categories
      3) infer_single_category_id_from_gt if GT is single-class
      else: raise ValueError
    """
    if target_category_id is not None:
        return int(target_category_id)

    if target_category_name is not None and str(target_category_name).strip():
        cid = find_category_id_by_name(gt_json_path, target_category_name)
        if cid is None:
            raise ValueError("Category name '{}' not found in GT categories.".format(target_category_name))
        return int(cid)

    cid = infer_single_category_id_from_gt(gt_json_path)
    if cid is not None:
        return int(cid)

    raise ValueError("Cannot infer target category_id from GT. Provide --target-category-id or --target-category-name.")
