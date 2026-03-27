# ICCT sparse video dataset wrapper.
# Python 3.9 compatible. Comments in English only.

from datasets.coco_video_dataset import CocoVideoFrameDataset


class IcctSparse20Dataset(CocoVideoFrameDataset):
    def __init__(self, data_root: str, split: str = "dev") -> None:
        super().__init__(
            data_root=data_root,
            split=split,
            annotation_prefix="fbdsv24_vid",
            dataset_name="icct_sparse20",
        )
