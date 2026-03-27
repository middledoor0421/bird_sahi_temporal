# datasets/fbdsv.py
# FBD-SV-2024 (VID) datasets utility.
# Python 3.9 compatible. Comments in English only.

from datasets.coco_video_dataset import CocoVideoFrameDataset


class FbdsvDataset(CocoVideoFrameDataset):
    def __init__(self, data_root: str, split: str = "val") -> None:
        super().__init__(
            data_root=data_root,
            split=split,
            annotation_prefix="fbdsv24_vid",
            dataset_name="fbdsv",
        )
