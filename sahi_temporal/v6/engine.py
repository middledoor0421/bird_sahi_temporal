# sahi_temporal/v6/engine.py
# Python 3.9 compatible. Comments in English only.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sahi_base.tiler import compute_slice_grid
from utils.category_mapper import map_label_to_category_id

from .tracker import TrackerV6
from .evidence import EvidenceBuilderV6, EvidenceBuilderConfig
from .policy import PolicyConfigV6, PolicyV6
from .tile_planner import TilePlannerV6, TilePlannerConfigV6
from .executor import TileExecutorV6
from sahi_base.postprocess import postprocess_preds, get_pp_preset

def _xywh_to_xyxy(b: List[float]) -> List[float]:
    x, y, w, h = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    return [x, y, x + w, y + h]


def _center_xyxy(xyxy: List[float]) -> Tuple[float, float]:
    return (0.5 * (float(xyxy[0]) + float(xyxy[2])), 0.5 * (float(xyxy[1]) + float(xyxy[3])))


def _hit_center_in_any_tile(gt_xyxy: List[float], tile_rects_xyxy: List[List[float]]) -> int:
    cx, cy = _center_xyxy(gt_xyxy)
    for r in tile_rects_xyxy:
        x1, y1, x2, y2 = float(r[0]), float(r[1]), float(r[2]), float(r[3])
        if (x1 <= cx < x2) and (y1 <= cy < y2):
            return 1
    return 0


def _no_det_bucket(no_det_streak: int) -> str:
    v = int(no_det_streak)
    if v <= 1:
        return "0_1"
    if v <= 5:
        return "2_5"
    return "6p"


@dataclass
class V6Config:
    slice_height: int = 512
    slice_width: int = 512
    overlap_h: float = 0.2
    overlap_w: float = 0.2

    merge_mode: str = "vanilla"
    nms_iou: float = 0.5

    target_category_ids: Optional[List[int]] = None

    K_mid: int = 3
    full_ttl: int = 3


class TemporalSahiV6Engine:
    """Temporal SAHI v6.2-next engine (No-Prior).

    Notes:
        - Evidence does not have ev.cond; conditions are computed locally.
        - Coverage@K is computed using GT centers and selected tile rects.
    """

    def __init__(
        self,
        detector,
        cfg: V6Config,
        class_mapping: Optional[Dict[int, int]] = None,
        exp_id: int = 0,
    ) -> None:
        self.detector = detector
        self.cfg = cfg
        self.class_mapping = class_mapping
        self.exp_id = int(exp_id)

        self.tiles: Optional[List[Dict[str, int]]] = None
        self.executor: Optional[TileExecutorV6] = None

        self.tracker = TrackerV6(
            category_whitelist=list(cfg.target_category_ids) if cfg.target_category_ids is not None else None,
            class_aware=False,
        )

        # Evidence builder uses cfg target ids.
        ev_cfg = EvidenceBuilderConfig(target_category_ids=list(cfg.target_category_ids) if cfg.target_category_ids is not None else None)
        self.evidence_builder = EvidenceBuilderV6(cfg=ev_cfg)

        # Policy: keep K fixed at cfg.K_mid for v6.2-next (subset experiments), and keep exp switch.
        pol_cfg = PolicyConfigV6(
            K_mid_low=int(cfg.K_mid),
            K_mid_high=int(cfg.K_mid),
            K_high_min=int(cfg.K_mid),
            K_high_max=int(cfg.K_mid),
            full_ttl=int(cfg.full_ttl),
        )
        self.policy = PolicyV6(cfg=pol_cfg, exp_id=int(self.exp_id))

        self.tile_planner = TilePlannerV6(cfg=TilePlannerConfigV6())

        self.logs: List[Dict[str, Any]] = []
        self.t = 0


        # Clip boundary tracking
        self._current_video_id: Optional[str] = None
    def _ensure_tiles(self, frame: np.ndarray) -> None:
        if self.tiles is not None:
            return

        if frame is None:
            raise RuntimeError("Frame is None")

        H, W = frame.shape[:2]
        self.tiles = compute_slice_grid(
            height=int(H),
            width=int(W),
            slice_height=int(self.cfg.slice_height),
            slice_width=int(self.cfg.slice_width),
            overlap_height_ratio=float(self.cfg.overlap_h),
            overlap_width_ratio=float(self.cfg.overlap_w),
        )

        self.executor = TileExecutorV6(
            detector=self.detector,
            tiles=self.tiles,
            merge_mode=str(self.cfg.merge_mode),
            nms_iou=float(self.cfg.nms_iou),
            class_mapping=self.class_mapping,
        )

        self.evidence_builder.set_tiles(self.tiles)

    def _full_frame_detect(self, frame: np.ndarray, image_id: int, score_threshold: float) -> List[Dict[str, Any]]:
        boxes_np, scores_np, labels_np = self.detector.predict(frame, conf_thres=score_threshold)
        if boxes_np is None or len(boxes_np) == 0:
            return []

        boxes_np = np.asarray(boxes_np, dtype=np.float32)
        scores_np = np.asarray(scores_np, dtype=np.float32)
        labels_np = np.asarray(labels_np, dtype=np.int64)

        preds: List[Dict[str, Any]] = []
        for i in range(int(boxes_np.shape[0])):
            sc = float(scores_np[i])
            if score_threshold > 0.0 and sc < float(score_threshold):
                continue

            x1 = float(boxes_np[i, 0])
            y1 = float(boxes_np[i, 1])
            x2 = float(boxes_np[i, 2])
            y2 = float(boxes_np[i, 3])

            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)

            label_id = int(labels_np[i])
            category_id = map_label_to_category_id(label_id=label_id, class_mapping=self.class_mapping)

            preds.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(category_id),
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(sc),
                }
            )

        return preds

    def process_frame(
        self,
        frame: np.ndarray,
        image_id: int,
        score_threshold: float = 0.0,
        gt_bboxes: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if frame is None:
            return [], {}

        if not isinstance(frame, np.ndarray):
            frame = np.asarray(frame)

        self._ensure_tiles(frame)
        assert self.tiles is not None
        assert self.executor is not None

        # 1) Full-frame detection for control signals
        full_preds = self._full_frame_detect(frame, int(image_id), float(score_threshold))

        # 2) Tracker update
        track_state = self.tracker.update(full_preds)

        # 3) Evidence (prev_mode from policy)
        ev = self.evidence_builder.update(
            frame=frame,
            det_out=full_preds,
            track_state=track_state,
            prev_mode=self.policy.prev_mode,
        )

        # 4) Policy
        plan = self.policy.step(ev, K_full=len(self.tiles))

        # 5) Tile planning
        tile_plan = self.tile_planner.build(plan, ev, track_state, self.tiles)

        # 6) Execute
        if len(tile_plan.tiles) > 0:
            preds = self.executor.run(
                frame=frame,
                image_id=int(image_id),
                tile_ids=tile_plan.tiles,
                score_threshold=float(score_threshold),
            )
            do_sahi = True
        else:
            preds = full_preds
            do_sahi = False

        h, w = frame.shape[:2]
        pp_cfg = get_pp_preset("icct_std_v1")
        preds, _pp_meta = postprocess_preds(preds, image_w=int(w), image_h=int(h), cfg=pp_cfg)

        # 7) Feedback to policy
        self.policy.observe_result(plan=plan, ev=ev, has_target_pred=bool(len(preds) > 0))

        # ------------------------
        # Logging (coverage)
        # ------------------------
        selected_tile_rects: List[List[float]] = []
        for tid in tile_plan.tiles:
            t = self.tiles[int(tid)]
            x1 = float(t["x"])
            y1 = float(t["y"])
            x2 = x1 + float(t["w"])
            y2 = y1 + float(t["h"])
            selected_tile_rects.append([x1, y1, x2, y2])

        dropped_tiles = [i for i in range(len(self.tiles)) if i not in set([int(x) for x in tile_plan.tiles])]

        # Conditions
        seed_empty = (int(ev.track.num_active) == 0) and (not bool(ev.det.has_det))
        bucket = _no_det_bucket(int(ev.det.no_det_streak))

        # GT filtering + centers
        gt_total = 0
        gt_hit = 0
        gt_small_total = 0
        gt_small_hit = 0
        gt_centers: List[List[float]] = []

        if gt_bboxes is not None:
            allow = set(int(x) for x in (self.cfg.target_category_ids or []))
            for g in gt_bboxes:
                cid = int(g.get("category_id", -1))
                if len(allow) > 0 and cid not in allow:
                    continue
                xyxy = _xywh_to_xyxy(g.get("bbox", [0, 0, 0, 0]))
                cx, cy = _center_xyxy(xyxy)
                gt_centers.append([float(cx), float(cy)])

                gt_total += 1
                h = _hit_center_in_any_tile(xyxy, selected_tile_rects)
                gt_hit += h

                area = float(g.get("area", 0.0))
                if area <= 1024.0:
                    gt_small_total += 1
                    gt_small_hit += h

        coverage = {
            "gt_total": int(gt_total),
            "gt_hit": int(gt_hit),
            "gt_small_total": int(gt_small_total),
            "gt_small_hit": int(gt_small_hit),
            "coverage_at_k": float(gt_hit) / float(max(gt_total, 1)),
            "coverage_at_k_small": float(gt_small_hit) / float(max(gt_small_total, 1)),
        }

        # exp_cfg stored in plan.cooldown
        exp_cfg = None
        if isinstance(plan.cooldown, dict):
            exp_cfg = plan.cooldown.get("exp_cfg", None)

        log = {
            "t": int(self.t),
            "image_id": int(image_id),
            "K": int(len(tile_plan.tiles)),
            "selected_tiles": [int(x) for x in tile_plan.tiles],
            "selected_tile_rects": selected_tile_rects,
            "dropped_tiles": dropped_tiles,
            "gt_centers": gt_centers,
            "cond": {
                "seed_empty": bool(seed_empty),
                "no_det_streak": int(ev.det.no_det_streak),
                "no_det_bucket": str(bucket),
            },
            "coverage": coverage,
            "plan": {
                "mode": str(plan.mode),
                "risk": float(plan.risk),
                "K": int(plan.K),
                "protect_frac": float(plan.protect_frac),
                "explore_source": str(plan.explore_source),
                "escalate_reason": str(plan.escalate_reason),
                "exp_cfg": exp_cfg,
            },
            "tile_plan": tile_plan.to_dict(),
            "evidence": ev.to_dict(),
        }

        self.logs.append(log)
        self.t += 1

        short = {
            "do_sahi": bool(do_sahi),
            "mode": str(plan.mode),
            "K": int(len(tile_plan.tiles)),
            "tiles_selected": int(len(tile_plan.tiles)),
            "seed_empty": bool(seed_empty),
            "no_det_bucket": str(bucket),
        }

        return preds, short

    def reset_for_new_video(self, video_id: Optional[str]) -> None:
        """Reset temporal states at clip boundary.

        This is called by the runner when dataset video_id changes.
        It prevents motion buffer shape mismatch and recomputes tile grid
        when resolution changes between clips.
        """

        if video_id is None:
            return

        vid = str(video_id)
        if self._current_video_id == vid:
            return
        self._current_video_id = vid

        # Force tile grid recomputation for the new clip.
        self.tiles = None
        self.executor = None
        try:
            self.evidence_builder.clear_tiles()
        except Exception:
            pass

        # Reset evidence temporal states (motion buffer, z-score stats, streak).
        try:
            self.evidence_builder.reset_for_new_video()
        except Exception:
            pass

        # Reset tracker at clip boundary to avoid cross-clip association.
        self.tracker = TrackerV6(
            category_whitelist=list(self.cfg.target_category_ids) if self.cfg.target_category_ids is not None else None,
            class_aware=False,
        )

        # Reset policy internal state (cooldowns/prev_mode) for a clean clip start.
        pol_cfg = PolicyConfigV6(
            K_mid_low=int(self.cfg.K_mid),
            K_mid_high=int(self.cfg.K_mid),
            K_high_min=int(self.cfg.K_mid),
            K_high_max=int(self.cfg.K_mid),
            full_ttl=int(self.cfg.full_ttl),
        )
        self.policy = PolicyV6(cfg=pol_cfg, exp_id=int(self.exp_id))
