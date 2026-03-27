# utils/category_mapper.py
# Python 3.9 compatible. Comments in English only.

from typing import Dict, Optional


def map_label_to_category_id(
    label_id: int,
    class_mapping: Optional[Dict[int, int]] = None,
) -> int:
    """Map model label_id to dataset category_id.

    Default behavior:
      - If class_mapping is None: return label_id (identity mapping).
      - Else: return class_mapping[label_id] if exists, otherwise return label_id.
    """
    lid = int(label_id)
    if class_mapping is not None and lid in class_mapping:
        return int(class_mapping[lid])
    return int(lid)
