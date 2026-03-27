import argparse
import json
import os
import random
from collections import defaultdict

def _load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def _save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

def _compute_video_meta(sequences, ann, target_category_id):
    # image_id -> has_target
    has_target = defaultdict(int)
    for a in ann.get('annotations', []):
        if int(a.get('category_id', -1)) == int(target_category_id):
            has_target[int(a['image_id'])] = 1

    video_meta = {}
    for v in sequences:
        vid = v['video_id']
        frame_ids = v.get('frame_ids', [])
        flags = [has_target[int(fid)] for fid in frame_ids]
        video_len = len(frame_ids)
        if video_len == 0:
            continue

        pos_total = sum(flags)

        # segments + max runs
        num_pos_segments = 0
        max_pos_run = 0
        max_neg_run = 0

        cur = flags[0]
        run = 1
        for f in flags[1:]:
            if f == cur:
                run += 1
            else:
                if cur == 1:
                    num_pos_segments += 1
                    max_pos_run = max(max_pos_run, run)
                else:
                    max_neg_run = max(max_neg_run, run)
                cur = f
                run = 1
        if cur == 1:
            num_pos_segments += 1
            max_pos_run = max(max_pos_run, run)
        else:
            max_neg_run = max(max_neg_run, run)

        presence_ratio = pos_total / max(1, video_len)

        video_meta[vid] = {
            'video_len': video_len,
            'num_pos_segments': num_pos_segments,
            'pos_total_frames': pos_total,
            'presence_ratio_video': presence_ratio,
            'max_neg_run_len': max_neg_run,
            'max_pos_run_len': max_pos_run,
        }
    return video_meta

def _assign_bins(meta):
    L = int(meta['video_len'])
    S = int(meta['num_pos_segments'])
    R = float(meta['presence_ratio_video'])

    if L <= 30:
        len_bin = 'L<=30'
    elif L <= 60:
        len_bin = '31-60'
    elif L <= 120:
        len_bin = '61-120'
    else:
        len_bin = '>120'

    if S == 0:
        seg_bin = '0'
    elif S == 1:
        seg_bin = '1'
    else:
        seg_bin = '>=2'

    if R == 0:
        ratio_bin = '0'
    elif R <= 0.10:
        ratio_bin = '(0,0.1]'
    elif R <= 0.30:
        ratio_bin = '(0.1,0.3]'
    else:
        ratio_bin = '>0.3'

    meta['bins'] = {
        'video_len_bin': len_bin,
        'num_pos_segments_bin': seg_bin,
        'presence_ratio_bin': ratio_bin,
    }
    return (len_bin, seg_bin, ratio_bin)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sequences_json', required=True)
    ap.add_argument('--ann_json', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--target_category_id', type=int, default=14)
    ap.add_argument('--holdout_ratio', type=float, default=0.4)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)

    sequences = _load_json(args.sequences_json)
    ann = _load_json(args.ann_json)

    video_meta = _compute_video_meta(sequences, ann, args.target_category_id)

    strata = defaultdict(list)
    for vid, meta in video_meta.items():
        key = _assign_bins(meta)
        strata[key].append(vid)

    dev_videos = []
    holdout_videos = []

    for _, vids in strata.items():
        random.shuffle(vids)
        n = len(vids)
        if n == 0:
            continue
        # ensure at least 1 holdout per stratum
        k = max(1, round(n * float(args.holdout_ratio)))
        k = min(k, n - 1) if n > 1 else 1
        holdout_videos.extend(vids[:k])
        dev_videos.extend(vids[k:])

    _save_json(dev_videos, os.path.join(args.out_dir, 'dev_videos.json'))
    _save_json(holdout_videos, os.path.join(args.out_dir, 'holdout_videos.json'))
    _save_json(video_meta, os.path.join(args.out_dir, 'video_meta.json'))

    report = {
        'seed': args.seed,
        'holdout_ratio': float(args.holdout_ratio),
        'num_videos_total': len(video_meta),
        'num_dev_videos': len(dev_videos),
        'num_holdout_videos': len(holdout_videos),
        'strata_count': len(strata),
    }
    _save_json(report, os.path.join(args.out_dir, 'split_meta_report.json'))

if __name__ == '__main__':
    main()
