#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
from typing import Any, Dict, List

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from scripts2.pred_slim import load_pred_any


def save_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def _load_image_ids(path: str) -> List[int]:
    with open(path, 'r') as f:
        obj = json.load(f)
    if isinstance(obj, dict) and 'image_ids' in obj:
        obj = obj['image_ids']
    if not isinstance(obj, list):
        raise ValueError("Invalid image ids json. Expect a list or a dict with 'image_ids'.")
    return [int(x) for x in obj]


def main() -> None:
    parser = argparse.ArgumentParser(description='scripts2: COCOeval for bbox predictions.')
    parser.add_argument('--gt', type=str, required=True)
    parser.add_argument('--pred', type=str, required=True, help='Predictions: .json list or .jsonl(.gz).')
    parser.add_argument('--image-ids-json', type=str, default=None)
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--small-max-area', type=float, default=32.0 * 32.0)
    parser.add_argument('--max-dets', type=int, default=100)
    args = parser.parse_args()

    coco_gt = COCO(args.gt)
    if 'info' not in coco_gt.dataset:
        coco_gt.dataset['info'] = {}
    if 'licenses' not in coco_gt.dataset:
        coco_gt.dataset['licenses'] = []

    pred_data = load_pred_any(args.pred)

    image_ids_filter = None
    if args.image_ids_json:
        image_ids_filter = set(_load_image_ids(args.image_ids_json))

    if image_ids_filter is not None:
        coco_gt.dataset['images'] = [im for im in coco_gt.dataset.get('images', []) if int(im.get('id', -1)) in image_ids_filter]
        coco_gt.dataset['annotations'] = [a for a in coco_gt.dataset.get('annotations', []) if int(a.get('image_id', -1)) in image_ids_filter]
        coco_gt.createIndex()
        pred_data = [p for p in pred_data if int(p.get('image_id', -1)) in image_ids_filter]

    if len(pred_data) == 0:
        metrics = {
            'AP': 0.0,
            'AP50': 0.0,
            'AP_small': 0.0,
            'AR_small': 0.0,
            'small_max_area': float(args.small_max_area),
            'max_dets': int(args.max_dets),
            'note': 'Empty predictions.',
        }
        print(metrics)
        if args.out:
            save_json(args.out, metrics)
        return

    pred_path = args.pred
    if args.pred.endswith('.jsonl') or args.pred.endswith('.jsonl.gz'):
        tmp_path = (args.out + '.tmp_pred.json') if args.out else 'tmp_pred.json'
        with open(tmp_path, 'w') as f:
            json.dump(pred_data, f)
        pred_path = tmp_path

    coco_dt = coco_gt.loadRes(pred_path)
    evaluator = COCOeval(coco_gt, coco_dt, iouType='bbox')

    small_max_area = float(args.small_max_area)
    evaluator.params.areaRng = [
        [0.0, 1e12],
        [0.0, small_max_area],
        [small_max_area, 1e12],
    ]
    evaluator.params.areaRngLbl = ['all', 'small', 'large']

    max_dets = int(args.max_dets)
    evaluator.params.maxDets = [1, 10, max_dets]

    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    metrics = {
        'AP': float(evaluator.stats[0]),
        'AP50': float(evaluator.stats[1]),
        'AP_small': float(evaluator.stats[3]),
        'AR_small': float(evaluator.stats[9]),
        'small_max_area': small_max_area,
        'max_dets': max_dets,
    }

    if args.out:
        save_json(args.out, metrics)
        print('Saved metrics to:', args.out)


if __name__ == '__main__':
    main()
