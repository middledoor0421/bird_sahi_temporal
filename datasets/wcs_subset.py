# datasets/wcs_subset.py

from datasets.coco_video_dataset import CocoVideoFrameDataset


class WcsSubsetDataset(CocoVideoFrameDataset):
    def __init__(self, data_root: str, split: str = "val"):
        super().__init__(
            data_root=data_root,
            split=split,
            annotation_prefix="fbdsv24_vid",
            dataset_name="wcs_subset",
        )
