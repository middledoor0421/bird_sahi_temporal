# Generic COCO-video frame dataset loader.
# Python 3.9 compatible. Comments in English only.

import json
import os
from typing import Any, Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np


class CocoVideoFrameDataset:
    """
    Generic frame-level loader for datasets exported in the shared COCO+sequence format
    used by this workspace.

    Expected files under <data_root>/annotations:
      - {annotation_prefix}_{split}.json
      - {annotation_prefix}_{split}_sequences.json
    """

    def __init__(
        self,
        data_root: str,
        split: str,
        annotation_prefix: str = "fbdsv24_vid",
        dataset_name: str = "dataset",
    ) -> None:
        self.data_root = os.path.abspath(data_root)
        self.split = str(split)
        self.annotation_prefix = str(annotation_prefix)
        self.dataset_name = str(dataset_name)

        ann_root = os.path.join(self.data_root, "annotations")
        seq_json_path = os.path.join(ann_root, "{}_{}_sequences.json".format(self.annotation_prefix, self.split))
        gt_json_path = os.path.join(ann_root, "{}_{}.json".format(self.annotation_prefix, self.split))

        if not os.path.isfile(seq_json_path):
            raise FileNotFoundError("Sequence json not found: {}".format(seq_json_path))
        if not os.path.isfile(gt_json_path):
            raise FileNotFoundError("GT annotation json not found: {}".format(gt_json_path))

        with open(seq_json_path, "r") as f:
            self.sequences = json.load(f)
        with open(gt_json_path, "r") as f:
            gt = json.load(f)

        self._num_frames = self._estimate_num_frames(self.sequences)
        self.gt_by_image = self._build_gt_index(gt)

    @staticmethod
    def _build_gt_index(gt: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
        gt_by_image: Dict[int, List[Dict[str, Any]]] = {}
        for ann in gt.get("annotations", []):
            try:
                img_id = int(ann["image_id"])
            except Exception:
                continue

            bbox = ann.get("bbox", None)
            if bbox is None:
                continue

            entry = {
                "bbox": bbox,
                "area": ann.get("area", 0.0),
                "category_id": ann.get("category_id"),
                "iscrowd": ann.get("iscrowd", 0),
                "track_id": ann.get("track_id", -1),
            }
            gt_by_image.setdefault(img_id, []).append(entry)
        return gt_by_image

    @staticmethod
    def _estimate_num_frames(sequences: Any) -> int:
        total = 0
        if not isinstance(sequences, list):
            return total

        for seq in sequences:
            if not isinstance(seq, dict):
                continue
            if "frames" in seq and isinstance(seq["frames"], list):
                total += len(seq["frames"])
            else:
                total += len(seq.get("frame_ids", []))
        return total

    def __len__(self) -> int:
        return self._num_frames

    @staticmethod
    def _pick_first(d: Dict[str, Any], keys: List[str]) -> Optional[Any]:
        for k in keys:
            if k in d and d[k] is not None:
                return d[k]
        return None

    def iter_frames(self) -> Generator[Tuple[np.ndarray, int, Dict[str, Any]], None, None]:
        for seq in self.sequences:
            if not isinstance(seq, dict):
                continue

            video_id = seq.get("video_id", seq.get("id", None))

            if "frames" in seq and isinstance(seq["frames"], list):
                frames_list = seq["frames"]
                for local_idx, sample in enumerate(frames_list):
                    if not isinstance(sample, dict):
                        continue

                    img_id_val = self._pick_first(sample, ["image_id", "img_id", "id"])
                    rel_path_val = self._pick_first(sample, ["rel_path", "path", "file_name", "file", "frame_file"])
                    if img_id_val is None or rel_path_val is None:
                        continue

                    try:
                        image_id = int(img_id_val)
                    except Exception:
                        continue

                    img_path = os.path.join(self.data_root, str(rel_path_val))
                    frame = cv2.imread(img_path)
                    if frame is None:
                        continue

                    meta: Dict[str, Any] = {
                        "dataset_name": self.dataset_name,
                        "video_id": video_id,
                        "frame_idx": local_idx,
                        "rel_path": str(rel_path_val),
                        "split": self.split,
                        "gt_bboxes": self.gt_by_image.get(image_id, []),
                    }
                    yield frame, image_id, meta
                continue

            frame_ids = seq.get("frame_ids", [])
            frame_files = seq.get("frame_files", [])
            if not isinstance(frame_ids, list) or not isinstance(frame_files, list):
                continue
            if len(frame_ids) != len(frame_files):
                raise ValueError("frame_ids and frame_files length mismatch in video_id='{}'".format(str(video_id)))

            for local_idx, (img_id, rel_path) in enumerate(zip(frame_ids, frame_files)):
                img_path = os.path.join(self.data_root, str(rel_path))
                frame = cv2.imread(img_path)
                if frame is None:
                    continue

                try:
                    image_id = int(img_id)
                except Exception:
                    continue

                meta = {
                    "dataset_name": self.dataset_name,
                    "video_id": video_id,
                    "frame_idx": local_idx,
                    "rel_path": str(rel_path),
                    "split": self.split,
                    "gt_bboxes": self.gt_by_image.get(image_id, []),
                }
                yield frame, image_id, meta
