import argparse
import gzip
import json
import os
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw


def load_coco_gt(gt_path: str, target_cid: int) -> Tuple[Dict[int, dict], Dict[int, bool]]:
    # Returns:
    # - images_by_id: {image_id: image_info}
    # - is_empty: {image_id: True if no target GT boxes}
    gt = json.load(open(gt_path, "r"))
    images_by_id = {int(im["id"]): im for im in gt.get("images", [])}

    has_target = {int(im_id): False for im_id in images_by_id.keys()}
    for ann in gt.get("annotations", []):
        try:
            if int(ann.get("category_id", -1)) != target_cid:
                continue
            im_id = int(ann["image_id"])
            if im_id in has_target:
                has_target[im_id] = True
        except Exception:
            continue

    is_empty = {im_id: (not has_target.get(im_id, False)) for im_id in images_by_id.keys()}
    return images_by_id, is_empty


def load_preds_index(pred_path: str, target_cid: int) -> Dict[int, List[dict]]:
    # Returns {image_id: [pred, ...]} with only target_cid
    by_img: Dict[int, List[dict]] = {}
    opener = gzip.open if pred_path.endswith(".gz") else open
    with opener(pred_path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            try:
                if int(d.get("category_id", -1)) != target_cid:
                    continue
                im_id = int(d["image_id"])
            except Exception:
                continue
            by_img.setdefault(im_id, []).append(d)
    return by_img


def max_score(preds: List[dict]) -> float:
    m = 0.0
    for p in preds:
        try:
            s = float(p.get("score", 0.0))
        except Exception:
            s = 0.0
        if s > m:
            m = s
    return m


def draw_boxes(
    img: Image.Image,
    preds: List[dict],
    score_thr: float,
    color: Tuple[int, int, int],
    label_prefix: str
) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)

    for p in preds:
        try:
            s = float(p.get("score", 0.0))
        except Exception:
            continue
        if s < score_thr:
            continue
        b = p.get("bbox", None)
        if not (isinstance(b, list) and len(b) == 4):
            continue
        x, y, w, h = [float(v) for v in b]
        x1, y1, x2, y2 = x, y, x + w, y + h
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1 + 2, y1 + 2), f"{label_prefix}{s:.2f}", fill=color)

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--data-root", required=True, help="Root to join with gt images[].file_name")
    ap.add_argument("--pred-yolo", required=True)
    ap.add_argument("--pred-full", required=True)
    ap.add_argument("--target-cid", type=int, default=1)
    ap.add_argument("--tau", type=float, default=0.7, help="High-confidence threshold for FP")
    ap.add_argument("--yolo-low", type=float, default=0.7, help="YOLO must be < this to be 'not high-conf'")
    ap.add_argument("--save-topk", type=int, default=1, help="How many examples to save")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--draw-yolo-score-thr", type=float, default=0.3, help="Draw YOLO boxes above this")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    images_by_id, is_empty = load_coco_gt(args.gt, args.target_cid)
    yolo_by = load_preds_index(args.pred_yolo, args.target_cid)
    full_by = load_preds_index(args.pred_full, args.target_cid)

    candidates = []
    for im_id, empty in is_empty.items():
        if not empty:
            continue
        y_preds = yolo_by.get(im_id, [])
        f_preds = full_by.get(im_id, [])
        y_m = max_score(y_preds) if y_preds else 0.0
        f_m = max_score(f_preds) if f_preds else 0.0

        if (f_m >= args.tau) and (y_m < args.yolo_low):
            candidates.append((im_id, f_m, y_m))

    candidates.sort(key=lambda x: (-x[1], x[2]))  # full high first, yolo low first

    # Save a CSV-like text for traceability
    with open(os.path.join(args.out_dir, "candidates.txt"), "w") as f:
        f.write("image_id\tfull_max\tyolo_max\n")
        for im_id, f_m, y_m in candidates[:200]:
            f.write(f"{im_id}\t{f_m:.6f}\t{y_m:.6f}\n")

    saved = 0
    for im_id, f_m, y_m in candidates:
        info = images_by_id.get(im_id, {})
        rel = info.get("file_name", None)
        if not rel:
            continue
        img_path = os.path.join(args.data_root, rel)
        if not os.path.exists(img_path):
            continue

        img = Image.open(img_path).convert("RGB")
        y_img = draw_boxes(img, yolo_by.get(im_id, []), args.draw_yolo_score_thr, (0, 255, 0), "yolo:")
        f_img = draw_boxes(y_img, full_by.get(im_id, []), args.tau, (255, 0, 0), "full:")

        out_path = os.path.join(args.out_dir, f"example_im{im_id}_full_ge{args.tau}_yolo_lt{args.yolo_low}.png")
        f_img.save(out_path)

        meta_path = os.path.join(args.out_dir, f"example_im{im_id}.json")
        with open(meta_path, "w") as mf:
            json.dump(
                {
                    "image_id": im_id,
                    "image_path": img_path,
                    "full_max_score": f_m,
                    "yolo_max_score": y_m,
                    "tau_full": args.tau,
                    "yolo_low": args.yolo_low,
                    "num_full_preds": len(full_by.get(im_id, [])),
                    "num_yolo_preds": len(yolo_by.get(im_id, [])),
                },
                mf,
                indent=2,
            )

        saved += 1
        if saved >= args.save_topk:
            break

    print("num_candidates:", len(candidates))
    print("saved_examples:", saved)
    print("out_dir:", args.out_dir)


if __name__ == "__main__":
    main()