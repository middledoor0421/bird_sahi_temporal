#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Convert FBD-SV-2024 VID (video detection) annotations to COCO format.

Assumed directory structure:

DATA_ROOT/
  VID/
    images/
      train/
        bird_1/
          000000.JPEG
          000001.JPEG
          ...
        bird_2/
          ...
      val/
        ...
    labels/
      train/
        bird_1/
          000000.xml
          000001.xml
          ...
        bird_2/
          ...
      val/
        ...

This script will produce:
  - COCO-style annotation json (images / annotations / categories)
  - sequence json listing frames for each video_id
"""

import os
import json
import argparse
from glob import glob
import xml.etree.ElementTree as ET

from PIL import Image


def parse_voc_xml(xml_path):
    """Parse Pascal VOC style xml file and return image size and object dicts."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_node = root.find("size")
    if size_node is not None:
        width = int(size_node.find("width").text)
        height = int(size_node.find("height").text)
    else:
        width, height = None, None

    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name", default="bird")
        difficult = int(obj.findtext("difficult", default="0"))
        track_id_text = obj.findtext("trackid")
        try:
            track_id = int(track_id_text) if track_id_text is not None else -1
        except ValueError:
            track_id = -1

        bbox_node = obj.find("bndbox")
        if bbox_node is None:
            continue

        xmin = float(bbox_node.findtext("xmin", default="0"))
        ymin = float(bbox_node.findtext("ymin", default="0"))
        xmax = float(bbox_node.findtext("xmax", default="0"))
        ymax = float(bbox_node.findtext("ymax", default="0"))

        xmin = max(0.0, xmin)
        ymin = max(0.0, ymin)
        w = max(0.0, xmax - xmin)
        h = max(0.0, ymax - ymin)

        objects.append(
            {
                "name": name,
                "difficult": difficult,
                "track_id": track_id,
                "bbox": [xmin, ymin, w, h],
            }
        )

    return width, height, objects


def get_image_size(img_path):
    """Load image only to get width and height."""
    with Image.open(img_path) as im:
        w, h = im.size
    return w, h


def convert_split(data_root, split, out_coco, out_seq):
    """Convert one split (train/val) to COCO and sequence json."""
    vid_img_root = os.path.join(data_root, "VID", "images", split)
    vid_lbl_root = os.path.join(data_root, "VID", "labels", split)

    if not os.path.isdir(vid_img_root):
        raise FileNotFoundError("Image root not found: {}".format(vid_img_root))

    video_ids = sorted(
        d for d in os.listdir(vid_img_root) if os.path.isdir(os.path.join(vid_img_root, d))
    )

    images = []
    annotations = []
    sequences = []

    image_id = 1
    ann_id = 1

    for video_id in video_ids:
        img_dir = os.path.join(vid_img_root, video_id)
        lbl_dir = os.path.join(vid_lbl_root, video_id)

        frame_paths = sorted(
            glob(os.path.join(img_dir, "*.jpg"))
            + glob(os.path.join(img_dir, "*.jpeg"))
            + glob(os.path.join(img_dir, "*.JPEG"))
            + glob(os.path.join(img_dir, "*.png"))
        )

        if not frame_paths:
            continue

        seq_frame_ids = []
        seq_frame_files = []

        for frame_path in frame_paths:
            file_name = os.path.relpath(frame_path, data_root).replace("\\", "/")

            frame_basename = os.path.splitext(os.path.basename(frame_path))[0]
            xml_path = os.path.join(lbl_dir, frame_basename + ".xml")

            if os.path.isfile(xml_path):
                width, height, objs = parse_voc_xml(xml_path)
            else:
                width, height = get_image_size(frame_path)
                objs = []

            if width is None or height is None:
                width, height = get_image_size(frame_path)

            frame_idx = int(frame_basename)

            images.append(
                {
                    "id": image_id,
                    "file_name": file_name,
                    "width": int(width),
                    "height": int(height),
                    "video_id": video_id,
                    "frame_id": frame_idx,
                }
            )

            seq_frame_ids.append(image_id)
            seq_frame_files.append(file_name)

            for obj in objs:
                bbox = obj["bbox"]
                area = bbox[2] * bbox[3]
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": bbox,
                        "area": float(area),
                        "iscrowd": 0,
                        "difficult": int(obj["difficult"]),
                        "track_id": int(obj["track_id"]),
                    }
                )
                ann_id += 1

            image_id += 1

        sequences.append(
            {
                "video_id": video_id,
                "split": split,
                "frame_ids": seq_frame_ids,
                "frame_files": seq_frame_files,
            }
        )

    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "bird"}],
    }

    os.makedirs(os.path.dirname(out_coco), exist_ok=True)
    with open(out_coco, "w") as f:
        json.dump(coco_dict, f)

    os.makedirs(os.path.dirname(out_seq), exist_ok=True)
    with open(out_seq, "w") as f:
        json.dump(sequences, f)

    print("Saved COCO annotations to:", out_coco)
    print("Saved sequence list to:", out_seq)
    print("Total images:", len(images))
    print("Total annotations:", len(annotations))
    print("Total sequences:", len(sequences))


def main():
    parser = argparse.ArgumentParser(
        description="Convert FBD-SV-2024 VID annotations to COCO format."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Path to FBD-SV-2024 root directory.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val"],
        help="Dataset split to convert.",
    )
    parser.add_argument(
        "--out-coco",
        type=str,
        default=None,
        help="Output COCO json path. If None, use default under data-root.",
    )
    parser.add_argument(
        "--out-seq",
        type=str,
        default=None,
        help="Output sequence json path. If None, use default under data-root.",
    )

    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)

    if args.out_coco is None:
        out_coco = os.path.join(data_root, "annotations", "fbdsv24_vid_{}.json".format(args.split))
    else:
        out_coco = args.out_coco

    if args.out_seq is None:
        out_seq = os.path.join(
            data_root, "annotations", "fbdsv24_vid_{}_sequences.json".format(args.split)
        )
    else:
        out_seq = args.out_seq

    convert_split(data_root, args.split, out_coco, out_seq)


if __name__ == "__main__":
    main()
