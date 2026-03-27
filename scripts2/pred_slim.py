#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gzip
import json
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

JsonLike = Union[Dict[str, Any], List[Any]]


def _try_int(x: Any) -> Any:
    try:
        return int(x)
    except Exception:
        return x


def slim_predictions(preds: List[Dict[str, Any]], target_category_id: Optional[int], score_thr: float, max_det_per_image: int, remap_category_id: Optional[int] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Convert predictions to COCO-result-like slim format:
      {image_id, category_id, bbox, score}

    Enforced:
      - Apply score threshold
      - Apply per-image top-K (max_det_per_image)
      - If target_category_id is provided, keep only that category_id

    Note:
      - We do NOT remap category_id here. Keep the GT target id to avoid confusion.
    """
    if max_det_per_image <= 0:
        max_det_per_image = 100

    kept_by_img = defaultdict(list)
    dropped_low = 0
    dropped_cat = 0
    dropped_bad = 0

    for p in preds:
        if not isinstance(p, dict):
            dropped_bad += 1
            continue
        if "image_id" not in p or "bbox" not in p:
            dropped_bad += 1
            continue
        score = float(p.get("score", 0.0))
        if score < float(score_thr):
            dropped_low += 1
            continue
        cid = _try_int(p.get("category_id", -1))
        # Optional: remap all predictions to a single category id (e.g., ICCT 1-class)
        if remap_category_id is not None:
            cid = int(remap_category_id)
        if target_category_id is not None and int(cid) != int(target_category_id):
            dropped_cat += 1
            continue
        img_id = _try_int(p.get("image_id"))
        bbox = p.get("bbox", None)
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            dropped_bad += 1
            continue
        kept_by_img[img_id].append({
            "image_id": img_id,
            "category_id": int(cid) if isinstance(cid, int) or str(cid).isdigit() else cid,
            "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
            "score": score,
        })

    out = []
    for img_id, lst in kept_by_img.items():
        lst.sort(key=lambda x: float(x["score"]), reverse=True)
        out.extend(lst[:max_det_per_image])

    meta = {
        "num_in": int(len(preds)),
        "num_out": int(len(out)),
        "dropped_low_score": int(dropped_low),
        "dropped_other_category": int(dropped_cat),
        "dropped_bad_format": int(dropped_bad),
        "pred_score_thr": float(score_thr),
        "score_thr": float(score_thr),
        "max_det_per_image": int(max_det_per_image),
        "target_category_id": int(target_category_id) if target_category_id is not None else None,
        "remap_category_id": int(remap_category_id) if remap_category_id is not None else None,
    }
    return out, meta


def save_json(path: str, obj: Any) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_jsonl_gz(path: str, items: Iterable[Dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")


def load_pred_any(path: str) -> List[Dict[str, Any]]:
    """
    Load predictions from:
      - JSON list (COCO results)
      - JSONL (one dict per line)
      - JSONL.GZ
    """
    if path.endswith('.jsonl.gz'):
        items = []
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        return items
    if path.endswith('.jsonl'):
        items = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        return items
    with open(path, 'r') as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise ValueError('Prediction file must be a list for .json format.')
    return obj
