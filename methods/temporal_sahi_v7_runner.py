# methods/temporal_sahi_v7_runner.py
# Python 3.9 compatible. Comments in English only.

import gzip
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from detector.yolo_wrapper import YoloDetector
from sahi_temporal.v7.engine import TemporalSahiV7Engine, V7Config


def _parse_int_list_any(x) -> Optional[List[int]]:
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        out = [int(v) for v in x]
        return out if len(out) > 0 else None
    s = str(x).strip()
    if s == "":
        return None
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
    if s == "":
        return None
    return [int(t.strip()) for t in s.split(",") if t.strip()]


def _avg(xs: List[float]) -> Optional[float]:
    if len(xs) == 0:
        return None
    return float(sum(xs)) / float(len(xs))


def _get_video_id_from_meta(meta: Any) -> Optional[str]:
    # Keep consistent with v6 runner
    if not isinstance(meta, dict):
        return None
    for k in ["video_id", "vid", "video", "clip_id", "clip", "seq_video_id", "sequence_id", "session_id"]:
        v = meta.get(k, None)
        if v is None:
            continue
        s = str(v).strip()
        if s != "":
            return s
    return None


def _open_cost_writer(path: Optional[str]):
    # Disable if path is empty or /dev/null
    if path is None:
        return None
    p = str(path).strip()
    if p == "" or p == "/dev/null":
        return None
    d = os.path.dirname(p)
    if d:
        os.makedirs(d, exist_ok=True)
    return gzip.open(p, "wt", encoding="utf-8")


class SahiTemporalV7Runner:
    """
    Runner wrapper compatible with scripts2/run_detector2.py.

    This mirrors v6 runner behavior:
      - Detect per frame through engine.process_frame
      - Reset at video boundary
      - Optional per-frame cost logging (jsonl.gz)
    """

    def __init__(
        self,
        weights: str,
        device: str = "cuda:0",
        img_size: int = 640,
        conf_thres: float = 0.25,
        class_mapping: Optional[Dict[int, int]] = None,
        slice_height: int = 512,
        slice_width: int = 512,
        overlap_height_ratio: float = 0.2,
        overlap_width_ratio: float = 0.2,
        nms_iou_threshold: float = 0.5,
        merge_mode: str = "vanilla",
        log_path: Optional[str] = None,
        # v7 args (use v6 naming pattern for compatibility)
        target_ids: Optional[List[int]] = None,
        k_mid: int = 3,
        full_ttl: int = 0,
        exp_id: int = 0,
        # backward compat keys from run_detector2 may still pass these
        v6_target_ids: Optional[str] = None,
        v6_K_mid: Optional[int] = None,
        v6_full_ttl: Optional[int] = None,
        cost_per_frame_path: Optional[str] = None,
        # v7 confirmation knobs (wired from run_detector2)
        confirm_iou: Optional[float] = None,
        confirm_len: Optional[int] = None,
        confirm_max_age: Optional[int] = None,
        v7_confirm_iou: float = 0.4,
        v7_confirm_len: int = 2,
        v7_confirm_max_age: int = 2,
        # v7 engine-level postprocess knobs
        pp_do_clamp: Optional[bool] = None,
        pp_do_nms: Optional[bool] = None,
        pp_nms_iou: Optional[float] = None,
        pp_class_agnostic_nms: Optional[bool] = None,
        pp_max_det: Optional[int] = None,
        pp_score_thr: Optional[float] = None,
        v7_pp_do_clamp: bool = True,
        v7_pp_do_nms: bool = False,
        v7_pp_nms_iou: float = 0.0,
        v7_pp_class_agnostic_nms: bool = False,
        v7_pp_max_det: int = 0,
        v7_pp_score_thr: float = 0.0,
        use_confirmation: bool = True,
        use_dual_budget: bool = True,
        use_targeted_verify: bool = True,
        verifier_backend: str = "sahi",
        crop_margin_ratio: float = 0.2,
        crop_input_size: int = 960,
        empty_explore_rate: float = 0.10,
        empty_explore_cap: float = 2.0,
        empty_explore_cost: float = 1.0,
        pos_explore_rate: float = 0.15,
        pos_explore_cap: float = 2.0,
        pos_explore_cost: float = 1.0,
        pos_motion_local_peak_thr: float = 1e9,
        pos_diff_ratio_thr: float = 1.5,
        tile_score_mode: str = "mean",
        tile_apply_row_quota: bool = False,
        tile_quota_mode: str = "top1",
        tile_use_memory: bool = False,
        tile_memory_ttl: int = 0,
        tile_cand_extra: int = 6,
        force_verify_mode: str = "policy",
        force_verify_key_interval: int = 4,
    ) -> None:
        self.weights = str(weights)
        self.device = str(device)
        self.img_size = int(img_size)
        self.conf_thres = float(conf_thres)
        self.class_mapping = class_mapping
        self.log_path = log_path
        self.cost_per_frame_path = cost_per_frame_path

        # Harmonize target ids with legacy args (same pattern as v6 runner)
        if target_ids is None:
            target_ids = _parse_int_list_any(v6_target_ids) if v6_target_ids is not None else None
        if v6_K_mid is not None:
            k_mid = int(v6_K_mid)
        if v6_full_ttl is not None:
            full_ttl = int(v6_full_ttl)
        if confirm_iou is not None:
            v7_confirm_iou = float(confirm_iou)
        if confirm_len is not None:
            v7_confirm_len = int(confirm_len)
        if confirm_max_age is not None:
            v7_confirm_max_age = int(confirm_max_age)
        if pp_do_clamp is not None:
            v7_pp_do_clamp = bool(pp_do_clamp)
        if pp_do_nms is not None:
            v7_pp_do_nms = bool(pp_do_nms)
        if pp_nms_iou is not None:
            v7_pp_nms_iou = float(pp_nms_iou)
        if pp_class_agnostic_nms is not None:
            v7_pp_class_agnostic_nms = bool(pp_class_agnostic_nms)
        if pp_max_det is not None:
            v7_pp_max_det = int(pp_max_det)
        if pp_score_thr is not None:
            v7_pp_score_thr = float(pp_score_thr)

        # Shared detector
        self.detector = YoloDetector(
            weights=self.weights,
            device=self.device,
            img_size=self.img_size,
            conf_thres=self.conf_thres,
            class_mapping=None,
        )

        cfg = V7Config(
            slice_height=int(slice_height),
            slice_width=int(slice_width),
            overlap_h=float(overlap_height_ratio),
            overlap_w=float(overlap_width_ratio),
            merge_mode=str(merge_mode),
            nms_iou=float(nms_iou_threshold),
            target_category_ids=target_ids,
            K_mid=int(k_mid),
            full_ttl=int(full_ttl),
            confirm_iou=float(v7_confirm_iou),
            confirm_len=int(v7_confirm_len),
            confirm_max_age=int(v7_confirm_max_age),
            pp_do_clamp=bool(v7_pp_do_clamp),
            pp_do_nms=bool(v7_pp_do_nms),
            pp_nms_iou=float(v7_pp_nms_iou),
            pp_class_agnostic_nms=bool(v7_pp_class_agnostic_nms),
            pp_max_det=int(v7_pp_max_det),
            pp_score_thr=float(v7_pp_score_thr),
            use_confirmation=bool(use_confirmation),
            use_dual_budget=bool(use_dual_budget),
            use_targeted_verify=bool(use_targeted_verify),
            verifier_backend=str(verifier_backend),
            crop_margin_ratio=float(crop_margin_ratio),
            crop_input_size=int(crop_input_size),
            empty_explore_rate=float(empty_explore_rate),
            empty_explore_cap=float(empty_explore_cap),
            empty_explore_cost=float(empty_explore_cost),
            pos_explore_rate=float(pos_explore_rate),
            pos_explore_cap=float(pos_explore_cap),
            pos_explore_cost=float(pos_explore_cost),
            pos_motion_local_peak_thr=float(pos_motion_local_peak_thr),
            pos_diff_ratio_thr=float(pos_diff_ratio_thr),
            tile_score_mode=str(tile_score_mode),
            tile_apply_row_quota=bool(tile_apply_row_quota),
            tile_quota_mode=str(tile_quota_mode),
            tile_use_memory=bool(tile_use_memory),
            tile_memory_ttl=int(tile_memory_ttl),
            tile_cand_extra=int(tile_cand_extra),
            force_verify_mode=str(force_verify_mode),
            force_verify_key_interval=int(force_verify_key_interval),
        )

        self.engine = TemporalSahiV7Engine(
            detector=self.detector,
            cfg=cfg,
            class_mapping=self.class_mapping,
            exp_id=int(exp_id),
        )

    def run(
        self,
        dataset,
        score_threshold: float = 0.0,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        preds: List[Dict[str, Any]] = []
        t0 = time.time()

        cost_f = _open_cost_writer(getattr(self, "cost_per_frame_path", None))

        num_frames = 0
        do_sahi_frames = 0
        verifier_frames = 0
        full_frames = 0
        subset_frames = 0
        K_sum = 0
        tiles_sum = 0
        verifier_regions_sum = 0

        try:
            from tqdm import tqdm
            it = tqdm(dataset.iter_frames(), desc="Temporal SAHI v7 inference")
        except ImportError:
            it = dataset.iter_frames()

        last_video_id: Optional[str] = None

        for frame, image_id, meta in it:
            vid = _get_video_id_from_meta(meta)
            if vid is not None and vid != last_video_id:
                try:
                    self.engine.reset_for_new_video(vid)
                except Exception:
                    pass
                last_video_id = vid

            if frame is None:
                continue

            try:
                if hasattr(it, "set_postfix"):
                    it.set_postfix({"image_id": int(image_id)})
            except Exception:
                pass

            if not isinstance(frame, np.ndarray):
                frame = np.asarray(frame)

            gt_bboxes = None
            frame_idx = None
            if isinstance(meta, dict):
                gt_bboxes = meta.get("gt_bboxes", None)
                frame_idx = meta.get("frame_idx", None)

            # Per-frame latency around engine.process_frame
            t_call0 = time.time()
            out, short = self.engine.process_frame(
                frame=frame,
                image_id=int(image_id),
                score_threshold=float(score_threshold),
                gt_bboxes=gt_bboxes,
            )
            t_call1 = time.time()

            preds.extend(out)

            # Update counters
            num_frames += 1
            if bool(short.get("do_sahi", False)):
                do_sahi_frames += 1
            if bool(short.get("verifier_executed", False)):
                verifier_frames += 1
            if str(short.get("mode", "")) == "full":
                full_frames += 1
            if str(short.get("mode", "")) == "subset":
                subset_frames += 1

            K_sum += int(short.get("K", 0))
            tiles_sum += int(short.get("tiles_selected", 0))
            verifier_regions_sum += int(short.get("num_verifier_regions", 0))

            # Write per-frame cost record (small scalars only)
            if cost_f is not None:
                if frame_idx is None:
                    frame_idx = int(num_frames - 1)

                sahi_on = 1 if bool(short.get("do_sahi", False)) else 0
                tiles_used = int(short.get("tiles_selected", 0))
                if tiles_used == 0:
                    tiles_used = int(short.get("K", 0))

                rec = {
                    "video_id": last_video_id if last_video_id is not None else "",
                    "t": int(frame_idx),
                    "image_id": int(image_id),
                    "sahi_on": int(sahi_on),
                    "tiles_used": int(tiles_used),
                    "verifier_backend": str(short.get("verifier_backend", "sahi")),
                    "verifier_executed": int(1 if bool(short.get("verifier_executed", False)) else 0),
                    "num_verifier_regions": int(short.get("num_verifier_regions", 0)),
                    "verifier_cost_proxy": float(short.get("verifier_cost_proxy", 0.0)),
                    "latency_ms": float((t_call1 - t_call0) * 1000.0),
                }
                cost_f.write(json.dumps(rec) + "\n")

        if cost_f is not None:
            cost_f.close()

        elapsed = time.time() - t0
        fps = float(num_frames) / float(elapsed) if elapsed > 0.0 else 0.0

        # Aggregate coverage (same pattern as v6 runner)
        cov_all: List[float] = []
        cov_small: List[float] = []
        cov_small_seed6p: List[float] = []
        cov_small_b01: List[float] = []
        cov_small_b25: List[float] = []
        cov_small_b6p: List[float] = []

        for rec in self.engine.logs:
            cov = rec.get("coverage", {})
            ca = cov.get("coverage_at_k", None)
            cs = cov.get("coverage_at_k_small", None)
            if isinstance(ca, (int, float)):
                cov_all.append(float(ca))
            if isinstance(cs, (int, float)):
                cov_small.append(float(cs))

            cond = rec.get("cond", {})
            seed_empty = bool(cond.get("seed_empty", False))
            bucket = str(cond.get("no_det_bucket", ""))

            if seed_empty and bucket == "6p" and isinstance(cs, (int, float)):
                cov_small_seed6p.append(float(cs))

            if bucket == "0_1" and isinstance(cs, (int, float)):
                cov_small_b01.append(float(cs))
            elif bucket == "2_5" and isinstance(cs, (int, float)):
                cov_small_b25.append(float(cs))
            elif bucket == "6p" and isinstance(cs, (int, float)):
                cov_small_b6p.append(float(cs))

        stats: Dict[str, Any] = {
            "method": "temporal_sahi_v7",
            "num_frames": int(num_frames),
            "elapsed_sec": float(elapsed),
            "fps": float(fps),
            "architecture": "yolo_plus_local_verifier",
            "verifier_backend": str(getattr(self.engine.cfg, "verifier_backend", "sahi")),
            "verify_schedule": str(getattr(self.engine.cfg, "force_verify_mode", "policy")),
            "force_verify_mode": str(getattr(self.engine.cfg, "force_verify_mode", "policy")),
            "force_verify_key_interval": int(getattr(self.engine.cfg, "force_verify_key_interval", 4)),
            "do_sahi_frames": int(do_sahi_frames),
            "verifier_frames": int(verifier_frames),
            "full_frames": int(full_frames),
            "subset_frames": int(subset_frames),
            "avg_K": float(K_sum) / float(num_frames) if num_frames > 0 else 0.0,
            "avg_tiles_selected": float(tiles_sum) / float(num_frames) if num_frames > 0 else 0.0,
            "avg_verifier_regions": float(verifier_regions_sum) / float(num_frames) if num_frames > 0 else 0.0,
            "log_path": self.log_path,
            "cost_per_frame_path": getattr(self, "cost_per_frame_path", None),
            "coverage_summary": {
                "coverage_at_k_all": _avg(cov_all),
                "coverage_at_k_small": _avg(cov_small),
                "coverage_at_k_small_seed_empty_6p": _avg(cov_small_seed6p),
                "coverage_at_k_small_no_det_0_1": _avg(cov_small_b01),
                "coverage_at_k_small_no_det_2_5": _avg(cov_small_b25),
                "coverage_at_k_small_no_det_6p": _avg(cov_small_b6p),
            },
        }

        if self.log_path is not None:
            d = os.path.dirname(str(self.log_path))
            if d:
                os.makedirs(d, exist_ok=True)
            with open(self.log_path, "w") as f:
                json.dump(self.engine.logs, f)

        return preds, stats
