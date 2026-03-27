#!/usr/bin/env python3
from __future__ import annotations

import gzip
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/mnt/ssd960/jm/sahi/data/FBD-SV-2024")
ANN_PATH = DATA_ROOT / "annotations" / "fbdsv24_vid_val.json"
OUTPUT_DIR = REPO_ROOT / "output" / "thesis2" / "failure_figures" / "2026-03-23_presubmission_candidates"


RUNS = {
    "selected": {
        "label": "Ours (selected)",
        "infer_dir": REPO_ROOT / "output" / "thesis2" / "phase4_candidates" / "experiments" / "2026-03-11_v7_noconfirm_fbdsv_val" / "infer",
    },
    "notarget": {
        "label": "No targeted verify",
        "infer_dir": REPO_ROOT / "output" / "thesis2" / "phase4_candidates" / "experiments" / "2026-03-11_v7_notargetverify_fbdsv_val" / "infer",
    },
    "crop": {
        "label": "CropVerifier backend",
        "infer_dir": REPO_ROOT / "output" / "thesis2" / "phase4_candidates" / "experiments" / "2026-03-12_v7_cropverify_fbdsv_val" / "infer",
    },
    "nodual": {
        "label": "No dual budget",
        "infer_dir": REPO_ROOT / "output" / "thesis2" / "phase4_candidates" / "experiments" / "2026-03-11_v7_nodual_fbdsv_val" / "infer",
    },
}


@dataclass(frozen=True)
class PanelSpec:
    run_key: str
    image_id: int
    panel_title: str


@dataclass(frozen=True)
class FigureSpec:
    stem: str
    title: str
    caption: str
    panels: list[PanelSpec]
    crop_mode: str  # "full" or "focus"


FIGURES = [
    FigureSpec(
        stem="case1_late_opening_selected",
        title="Case 1. Weak pre-signal followed by late opening",
        caption=(
            "A brief far-small bird remains in seed-empty mode for two frames before the scheduler "
            "opens three verifier regions and recovers the target on the third frame."
        ),
        panels=[
            PanelSpec("selected", 1484, "t=51: skip"),
            PanelSpec("selected", 1485, "t=52: skip"),
            PanelSpec("selected", 1486, "t=53: late recovery"),
        ],
        crop_mode="focus",
    ),
    FigureSpec(
        stem="case2_targeted_vs_notarget",
        title="Case 2. Targeted allocation versus untargeted skip",
        caption=(
            "On the same frame, targeted allocation opens three tiles that cover the tiny bird, "
            "whereas the no-targeted variant keeps verification off."
        ),
        panels=[
            PanelSpec("selected", 1122, "Ours (selected)"),
            PanelSpec("notarget", 1122, "No targeted verify"),
        ],
        crop_mode="full",
    ),
    FigureSpec(
        stem="case3_crop_backend_check",
        title="Case 3. Narrow backend check with the same selected regions",
        caption=(
            "Both backends receive the same three selected regions, but the crop-based verifier "
            "emits a bird detection while the SAHI backend misses the tiny target."
        ),
        panels=[
            PanelSpec("selected", 2235, "SAHI backend"),
            PanelSpec("crop", 2235, "CropVerifier backend"),
        ],
        crop_mode="focus",
    ),
    FigureSpec(
        stem="case4_nodual_overopening",
        title="Case 4. Over-opening without utility under no-dual-budget control",
        caption=(
            "The selected policy keeps verification off on an empty seed-empty frame, "
            "while the no-dual variant opens all six regions without target utility."
        ),
        panels=[
            PanelSpec("selected", 1432, "Ours (selected)"),
            PanelSpec("nodual", 1432, "No dual budget"),
        ],
        crop_mode="full",
    ),
]


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def load_coco():
    coco = load_json(ANN_PATH)
    images = {im["id"]: im for im in coco["images"]}
    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)
    return images, anns_by_image


def load_v7_log(path: Path) -> dict[int, dict]:
    arr = load_json(path)
    return {row["image_id"]: row for row in arr}


def load_preds(path: Path) -> dict[int, list[dict]]:
    by_image: dict[int, list[dict]] = defaultdict(list)
    with gzip.open(path, "rt") as f:
        for line in f:
            row = json.loads(line)
            by_image[row["image_id"]].append(row)
    return by_image


def build_run_cache():
    cache = {}
    for run_key, meta in RUNS.items():
        infer_dir = meta["infer_dir"]
        pred_name = "pred_slim_temporal_sahi_v7_exp0_val.jsonl.gz"
        cache[run_key] = {
            "label": meta["label"],
            "log": load_v7_log(infer_dir / "v7_log.json"),
            "pred": load_preds(infer_dir / pred_name),
        }
    return cache


def image_path(images: dict[int, dict], image_id: int) -> Path:
    return DATA_ROOT / images[image_id]["file_name"]


def target_gt_boxes(anns_by_image: dict[int, list[dict]], image_id: int) -> list[list[float]]:
    return [ann["bbox"] for ann in anns_by_image[image_id] if ann["category_id"] == 14]


def target_pred_boxes(preds_by_image: dict[int, list[dict]], image_id: int) -> list[list[float]]:
    return [pred["bbox"] for pred in preds_by_image.get(image_id, []) if pred["category_id"] == 14]


def union_box(boxes: Iterable[list[float]]) -> list[float] | None:
    boxes = list(boxes)
    if not boxes:
        return None
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[0] + b[2] for b in boxes)
    y2 = max(b[1] + b[3] for b in boxes)
    return [x1, y1, x2 - x1, y2 - y1]


def focus_window(
    gt_boxes: list[list[float]],
    pred_boxes: list[list[float]],
    width: int,
    height: int,
    margin: float = 120.0,
) -> tuple[float, float, float, float]:
    box = union_box(gt_boxes + pred_boxes)
    if box is None:
        return (0.0, 0.0, float(width), float(height))
    x, y, w, h = box
    x1 = max(0.0, x - margin)
    y1 = max(0.0, y - margin)
    x2 = min(float(width), x + w + margin)
    y2 = min(float(height), y + h + margin)
    if x2 - x1 < 240:
        extra = (240 - (x2 - x1)) / 2
        x1 = max(0.0, x1 - extra)
        x2 = min(float(width), x2 + extra)
    if y2 - y1 < 220:
        extra = (220 - (y2 - y1)) / 2
        y1 = max(0.0, y1 - extra)
        y2 = min(float(height), y2 + extra)
    return (x1, y1, x2, y2)


def draw_panel(ax, img, gt_boxes, pred_boxes, tile_rects, crop_window, title, footer):
    x1, y1, x2, y2 = crop_window
    ax.imshow(img)
    ax.set_xlim(x1, x2)
    ax.set_ylim(y2, y1)
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for tile in tile_rects:
        tx1, ty1, tx2, ty2 = tile
        ax.add_patch(
            Rectangle(
                (tx1, ty1),
                tx2 - tx1,
                ty2 - ty1,
                fill=False,
                edgecolor="#ff8c00",
                linewidth=2.0,
                linestyle="--",
            )
        )
    for box in gt_boxes:
        ax.add_patch(
            Rectangle(
                (box[0], box[1]),
                box[2],
                box[3],
                fill=False,
                edgecolor="#39d353",
                linewidth=2.4,
            )
        )
    for box in pred_boxes:
        ax.add_patch(
            Rectangle(
                (box[0], box[1]),
                box[2],
                box[3],
                fill=False,
                edgecolor="#00bcd4",
                linewidth=2.2,
            )
        )
    ax.text(
        0.02,
        0.02,
        footer,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="white",
        bbox={"facecolor": (0, 0, 0, 0.68), "edgecolor": "none", "pad": 5},
    )


def render_figures():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    images, anns_by_image = load_coco()
    run_cache = build_run_cache()
    case_rows = []

    for fig_spec in FIGURES:
        n = len(fig_spec.panels)
        fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.8), dpi=180)
        if n == 1:
            axes = [axes]

        for ax, panel in zip(axes, fig_spec.panels):
            run = run_cache[panel.run_key]
            log_row = run["log"][panel.image_id]
            gt_boxes = target_gt_boxes(anns_by_image, panel.image_id)
            pred_boxes = target_pred_boxes(run["pred"], panel.image_id)
            img = Image.open(image_path(images, panel.image_id)).convert("RGB")
            width, height = img.size

            if fig_spec.crop_mode == "focus":
                crop_window = focus_window(gt_boxes, pred_boxes, width, height)
            else:
                crop_window = (0.0, 0.0, float(width), float(height))

            footer = (
                f"K={log_row['plan']['K']} | mode={log_row['plan']['mode']} | "
                f"trigger={log_row['plan']['risk']:.2f} | preds={len(pred_boxes)}"
            )
            if panel.run_key == "crop":
                footer += " | crop backend"
            draw_panel(
                ax=ax,
                img=img,
                gt_boxes=gt_boxes,
                pred_boxes=pred_boxes,
                tile_rects=log_row["selected_tile_rects"],
                crop_window=crop_window,
                title=panel.panel_title,
                footer=footer,
            )

        fig.suptitle(fig_spec.title, fontsize=14, y=0.98)
        fig.subplots_adjust(top=0.82, bottom=0.08, wspace=0.06)
        out_path = OUTPUT_DIR / f"{fig_spec.stem}.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

        case_rows.append(
            {
                "stem": fig_spec.stem,
                "title": fig_spec.title,
                "caption": fig_spec.caption,
                "png": str(out_path.relative_to(REPO_ROOT)),
                "panels": [
                    {
                        "run_key": p.run_key,
                        "image_id": p.image_id,
                        "file_name": images[p.image_id]["file_name"],
                    }
                    for p in fig_spec.panels
                ],
            }
        )

    with open(OUTPUT_DIR / "cases.json", "w") as f:
        json.dump(case_rows, f, indent=2)

    with open(OUTPUT_DIR / "README.md", "w") as f:
        f.write("# Failure Figure Candidates\n\n")
        f.write("These figures are pre-submission candidate assets and are not yet wired into the manuscript.\n\n")
        for idx, row in enumerate(case_rows, start=1):
            f.write(f"## {idx}. {row['title']}\n\n")
            f.write(f"- File: `{row['png']}`\n")
            f.write(f"- Caption: {row['caption']}\n")
            f.write("- Panels:\n")
            for panel in row["panels"]:
                f.write(
                    f"  - `{panel['run_key']}` on image_id `{panel['image_id']}` "
                    f"(`{panel['file_name']}`)\n"
                )
            f.write("\n")


if __name__ == "__main__":
    render_figures()
