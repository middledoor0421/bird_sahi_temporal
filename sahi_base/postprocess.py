# sahi_base/postprocess.py
# Common postprocess for frame-level predictions.
# Python 3.9 compatible. Comments in English only.

from typing import Any, Dict, List, Optional, Tuple
import math


def _xywh_to_xyxy(b: List[float]) -> Tuple[float, float, float, float]:
    x = float(b[0])
    y = float(b[1])
    w = float(b[2])
    h = float(b[3])
    return x, y, x + w, y + h


def _xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float) -> List[float]:
    return [float(x1), float(y1), float(max(0.0, x2 - x1)), float(max(0.0, y2 - y1))]


def _clip(v: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, v)))


def _iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 0.0:
        return 0.0
    return float(inter / denom)


def _nms_indices(
    boxes_xyxy: List[Tuple[float, float, float, float]],
    scores: List[float],
    iou_thr: float,
) -> List[int]:
    # Greedy NMS. Pure python for portability.
    if len(boxes_xyxy) == 0:
        return []
    order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
    keep: List[int] = []
    thr = float(iou_thr)

    while len(order) > 0:
        i = order.pop(0)
        keep.append(i)
        if len(order) == 0:
            break
        rem: List[int] = []
        bi = boxes_xyxy[i]
        for j in order:
            if _iou_xyxy(bi, boxes_xyxy[j]) <= thr:
                rem.append(j)
        order = rem
    return keep

def get_pp_preset(name: str) -> Dict[str, Any]:
    # Return a cfg dict compatible with postprocess_preds().
    n = str(name).strip().lower()

    if n in ("icct_std_v1", "icct_std", "std", "default_icct"):
        return {
            "do_clamp": True,
            "do_nms": True,
            "nms_iou": 0.35,
            "class_agnostic_nms": True,
            "max_det": 100,
            "score_thr": 0.60,
        }

    if n in ("clamp_only", "none"):
        return {
            "do_clamp": True,
            "do_nms": False,
            "nms_iou": None,
            "class_agnostic_nms": False,
            "max_det": None,
            "score_thr": 0.0,
        }

    raise ValueError(f"Unknown postprocess preset: {name}")


def get_pp_preset_notes(name: str) -> Dict[str, Any]:
    # Optional: store non-enforced knobs for logging/documentation.
    # tile_topK is not applied in postprocess_preds; keep as a note for later steps.
    n = str(name).strip().lower()
    if n in ("icct_std_v1", "icct_std", "std", "default_icct"):
        return {"tile_topK": 100}
    return {}

def postprocess_preds(
    preds: List[Dict[str, Any]],
    image_w: int,
    image_h: int,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Apply common postprocess to COCO-style predictions.

    Expected pred format:
      {
        "image_id": int,
        "category_id": int,
        "bbox": [x, y, w, h],
        "score": float,
      }

    cfg keys (all optional):
      - "score_thr": float
      - "max_det": int
      - "nms_iou": float
      - "class_agnostic_nms": bool
      - "do_nms": bool
      - "do_clamp": bool
    """
    if cfg is None:
        cfg = {}

    score_thr = float(cfg.get("score_thr", 0.0) or 0.0)
    max_det = cfg.get("max_det", None)
    nms_iou = cfg.get("nms_iou", None)
    class_agnostic = bool(cfg.get("class_agnostic_nms", False))
    do_nms = bool(cfg.get("do_nms", False))
    do_clamp = bool(cfg.get("do_clamp", True))

    if max_det is not None:
        max_det = int(max_det)
        if max_det <= 0:
            max_det = None

    W = int(image_w)
    H = int(image_h)

    num_in = int(len(preds))
    oob_fixed = 0

    # 1) clamp + score filter + drop degenerate boxes
    filtered: List[Dict[str, Any]] = []
    for p in preds:
        try:
            sc = float(p.get("score", 0.0))
        except Exception:
            sc = 0.0
        if score_thr > 0.0 and sc < score_thr:
            continue

        b = p.get("bbox", None)
        if not isinstance(b, (list, tuple)) or len(b) != 4:
            continue
        x1, y1, x2, y2 = _xywh_to_xyxy([float(b[0]), float(b[1]), float(b[2]), float(b[3])])

        if do_clamp:
            cx1 = _clip(x1, 0.0, float(W))
            cy1 = _clip(y1, 0.0, float(H))
            cx2 = _clip(x2, 0.0, float(W))
            cy2 = _clip(y2, 0.0, float(H))
            if (cx1 != x1) or (cy1 != y1) or (cx2 != x2) or (cy2 != y2):
                oob_fixed += 1
            x1, y1, x2, y2 = cx1, cy1, cx2, cy2

        if (x2 - x1) <= 0.0 or (y2 - y1) <= 0.0:
            continue

        pp = dict(p)
        pp["bbox"] = _xyxy_to_xywh(x1, y1, x2, y2)
        pp["score"] = float(sc)
        filtered.append(pp)

    # 2) NMS (optional)
    out = filtered
    if do_nms and nms_iou is not None and float(nms_iou) > 0.0 and len(filtered) > 0:
        boxes_xyxy: List[Tuple[float, float, float, float]] = []
        scores: List[float] = []
        labels: List[int] = []
        for p in filtered:
            b = p["bbox"]
            x1, y1, x2, y2 = _xywh_to_xyxy(b)
            boxes_xyxy.append((x1, y1, x2, y2))
            scores.append(float(p["score"]))
            labels.append(int(p.get("category_id", 0)))

        if class_agnostic:
            keep = _nms_indices(boxes_xyxy, scores, float(nms_iou))
            out = [filtered[i] for i in keep]
        else:
            # per-class NMS
            by_cls: Dict[int, List[int]] = {}
            for i, c in enumerate(labels):
                by_cls.setdefault(int(c), []).append(i)

            kept_all: List[int] = []
            for c, idxs in by_cls.items():
                b_c = [boxes_xyxy[i] for i in idxs]
                s_c = [scores[i] for i in idxs]
                k_local = _nms_indices(b_c, s_c, float(nms_iou))
                kept_all.extend([idxs[j] for j in k_local])

            # keep stable ordering by score desc
            kept_all = sorted(kept_all, key=lambda i: scores[i], reverse=True)
            out = [filtered[i] for i in kept_all]

    # 3) max_det (optional)
    if max_det is not None and len(out) > max_det:
        out = sorted(out, key=lambda p: float(p.get("score", 0.0)), reverse=True)[:max_det]

    meta = {
        "num_in": int(num_in),
        "num_after_filter": int(len(filtered)),
        "num_out": int(len(out)),
        "oob_fixed": int(oob_fixed),
        "score_thr": float(score_thr),
        "do_nms": bool(do_nms),
        "nms_iou": float(nms_iou) if nms_iou is not None else None,
        "class_agnostic_nms": bool(class_agnostic),
        "max_det": int(max_det) if max_det is not None else None,
        "do_clamp": bool(do_clamp),
    }
    return out, meta
