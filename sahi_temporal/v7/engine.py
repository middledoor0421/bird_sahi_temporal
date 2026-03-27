# sahi_temporal/v7/engine.py
# Python 3.9 compatible. Comments in English only.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sahi_base.tiler import compute_slice_grid
from utils.category_mapper import map_label_to_category_id

from .tracker import TrackerV7
from .evidence import EvidenceBuilderV7, EvidenceBuilderConfig
from .policy import PolicyConfigV7, PolicyV7
from .tile_planner import TilePlannerV7, TilePlannerConfigV7
from .executor import CropRecheckExecutorV7, TileExecutorV7

from .schemas import (
    V7_LOG_SCHEMA_VERSION,
    V7_METHOD_VERSION,
    V7_LOG_REQUIRED_KEYS,
    V7_LOG_REQUIRED_META_KEYS,
    ActionPlan,
    Evidence,
)

from sahi_base.postprocess import postprocess_preds, get_pp_preset
from .confirmation import ConfirmationConfig, ConfirmationGate

class _ConfirmLogger:
    """
    Lightweight confirmation logger to compute time-to-confirm statistics without
    changing the actual confirmation gate behavior.
    """
    def __init__(self, iou_thr: float, confirm_len: int, max_age: int) -> None:
        self.iou_thr = float(iou_thr)
        self.confirm_len = int(confirm_len)
        self.max_age = int(max_age)
        # candidates: list of dicts with keys: bbox_xyxy, first_t, last_t, hits, confirmed
        self._cands: List[Dict[str, Any]] = []
        self.ttc_values: List[int] = []
        self.newly_confirmed_total = 0

    def reset(self) -> None:
        self._cands = []
        self.ttc_values = []
        self.newly_confirmed_total = 0

    @staticmethod
    def _iou(a: List[float], b: List[float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = area_a + area_b - inter
        return float(inter / denom) if denom > 0.0 else 0.0

    def update(self, t: int, preds: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Update candidates from current-frame predictions (xywh) and return per-frame stats.
        """
        tt = int(t)
        boxes = [_xywh_to_xyxy(p["bbox"]) for p in preds if "bbox" in p]
        used = set()

        # age out old candidates
        kept = []
        for c in self._cands:
            if (tt - int(c["last_t"])) <= self.max_age:
                kept.append(c)
        self._cands = kept

        # match boxes to candidates
        for bi, b in enumerate(boxes):
            best_j = -1
            best_iou = 0.0
            for j, c in enumerate(self._cands):
                iou = self._iou(b, c["bbox_xyxy"])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_j >= 0 and best_iou >= self.iou_thr:
                c = self._cands[best_j]
                c["bbox_xyxy"] = b
                # consecutive hit if last seen in previous frame
                if int(c["last_t"]) == (tt - 1):
                    c["hits"] = int(c["hits"]) + 1
                else:
                    c["hits"] = 1
                    c["first_t"] = tt
                c["last_t"] = tt
                used.add(bi)
            else:
                # new candidate
                self._cands.append(
                    {
                        "bbox_xyxy": b,
                        "first_t": tt,
                        "last_t": tt,
                        "hits": 1,
                        "confirmed": False,
                    }
                )
                used.add(bi)

        newly = 0
        ttc_list: List[int] = []
        for c in self._cands:
            if (not bool(c["confirmed"])) and int(c["hits"]) >= self.confirm_len:
                c["confirmed"] = True
                newly += 1
                ttc = int(c["last_t"]) - int(c["first_t"]) + 1
                self.ttc_values.append(ttc)
                ttc_list.append(ttc)

        self.newly_confirmed_total += newly
        return {
            "newly_confirmed": int(newly),
            "ttc_list": ttc_list,
        }


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
class V7Config:
    slice_height: int = 512
    slice_width: int = 512
    overlap_h: float = 0.2
    overlap_w: float = 0.2

    merge_mode: str = "vanilla"
    nms_iou: float = 0.5

    target_category_ids: Optional[List[int]] = None

    K_mid: int = 3
    full_ttl: int = 3

    confirm_iou: float = 0.4
    confirm_len: int = 2
    confirm_max_age: int = 2

    # Step5: ablation switches
    use_confirmation: bool = True
    use_dual_budget: bool = True
    use_targeted_verify: bool = True
    verifier_backend: str = "sahi"
    crop_margin_ratio: float = 0.2
    crop_input_size: int = 960
    empty_explore_rate: float = 0.10
    empty_explore_cap: float = 2.0
    empty_explore_cost: float = 1.0
    pos_explore_rate: float = 0.15
    pos_explore_cap: float = 2.0
    pos_explore_cost: float = 1.0
    pos_motion_local_peak_thr: float = 1e9
    pos_diff_ratio_thr: float = 1.5
    tile_score_mode: str = "mean"
    tile_apply_row_quota: bool = False
    tile_quota_mode: str = "top1"
    tile_use_memory: bool = False
    tile_memory_ttl: int = 0
    tile_cand_extra: int = 6
    force_verify_mode: str = "policy"
    force_verify_key_interval: int = 4

    # Postprocess knobs for engine-level output (keep minimal changes at Step3)
    pp_do_clamp: bool = True
    pp_do_nms: bool = False
    pp_nms_iou: float = 0.0
    pp_class_agnostic_nms: bool = False
    pp_max_det: int = 0
    pp_score_thr: float = 0.0

class TemporalSahiV7Engine:
    """Temporal SAHI v7.2-next engine (No-Prior).

    Notes:
        - Evidence does not have ev.cond; conditions are computed locally.
        - Coverage@K is computed using GT centers and selected tile rects.
    """

    def __init__(
        self,
        detector,
        cfg: V7Config,
        class_mapping: Optional[Dict[int, int]] = None,
        exp_id: int = 0,
    ) -> None:
        self.detector = detector
        self.cfg = cfg
        self.class_mapping = class_mapping
        self.exp_id = int(exp_id)

        self.tiles: Optional[List[Dict[str, int]]] = None
        self.executor: Optional[Any] = None

        self.tracker = TrackerV7(
            category_whitelist=list(cfg.target_category_ids) if cfg.target_category_ids is not None else None,
            class_aware=False,
        )

        # Evidence builder uses cfg target ids.
        ev_cfg = EvidenceBuilderConfig(target_category_ids=list(cfg.target_category_ids) if cfg.target_category_ids is not None else None)
        self.evidence_builder = EvidenceBuilderV7(cfg=ev_cfg)

        # Step5: ablation switches
        self.use_confirmation = bool(getattr(cfg, "use_confirmation", True))
        self.use_dual_budget = bool(getattr(cfg, "use_dual_budget", True))
        self.use_targeted_verify = bool(getattr(cfg, "use_targeted_verify", True))
        # Policy: keep K fixed at cfg.K_mid for v7.2-next (subset experiments), and keep exp switch.
        self.policy = self._build_policy()
        self.tile_planner = TilePlannerV7(
            cfg=TilePlannerConfigV7(
                cand_extra=int(self.cfg.tile_cand_extra),
                score_mode=str(self.cfg.tile_score_mode),
                apply_row_quota=bool(self.cfg.tile_apply_row_quota),
                quota_mode=str(self.cfg.tile_quota_mode),
                use_memory=bool(self.cfg.tile_use_memory),
                memory_ttl=int(self.cfg.tile_memory_ttl),
            )
        )

        self.logs: List[Dict[str, Any]] = []
        self.t = 0

        # Clip boundary tracking
        self._current_video_id: Optional[str] = None

        self.confirm_cfg = ConfirmationConfig(
            iou_thr=float(self.cfg.confirm_iou),
            confirm_len=int(self.cfg.confirm_len),
            max_age=int(self.cfg.confirm_max_age),
        )
        self.confirm_gate = ConfirmationGate(self.confirm_cfg)
        self._confirm_logger = _ConfirmLogger(
            iou_thr=float(self.cfg.confirm_iou),
            confirm_len=int(self.cfg.confirm_len),
            max_age=int(self.cfg.confirm_max_age),
        )

    def _build_policy(self) -> PolicyV7:
        pol_cfg = PolicyConfigV7(
            K_mid_low=int(self.cfg.K_mid),
            K_mid_high=int(self.cfg.K_mid),
            K_high_min=int(self.cfg.K_mid),
            K_high_max=int(self.cfg.K_mid),
            full_ttl=int(self.cfg.full_ttl),
            empty_explore_rate=float(self.cfg.empty_explore_rate),
            empty_explore_cap=float(self.cfg.empty_explore_cap),
            empty_explore_cost=float(self.cfg.empty_explore_cost),
            pos_explore_rate=float(self.cfg.pos_explore_rate),
            pos_explore_cap=float(self.cfg.pos_explore_cap),
            pos_explore_cost=float(self.cfg.pos_explore_cost),
            pos_motion_local_peak_thr=float(self.cfg.pos_motion_local_peak_thr),
            pos_diff_ratio_thr=float(self.cfg.pos_diff_ratio_thr),
        )
        policy = PolicyV7(cfg=pol_cfg, exp_id=int(self.exp_id))
        if hasattr(policy, "use_dual_budget"):
            setattr(policy, "use_dual_budget", bool(getattr(self, "use_dual_budget", True)))
        if hasattr(policy, "use_targeted_verify"):
            setattr(policy, "use_targeted_verify", bool(getattr(self, "use_targeted_verify", True)))
        return policy

    def _build_sahi_postprocess_cfg(self) -> Dict[str, Any]:
        cfg = get_pp_preset("icct_std_v1")
        cfg["do_clamp"] = bool(self.cfg.pp_do_clamp)
        cfg["do_nms"] = bool(self.cfg.pp_do_nms)
        cfg["nms_iou"] = float(self.cfg.pp_nms_iou)
        cfg["class_agnostic_nms"] = bool(self.cfg.pp_class_agnostic_nms)
        cfg["score_thr"] = float(self.cfg.pp_score_thr)
        cfg["max_det"] = int(self.cfg.pp_max_det) if int(self.cfg.pp_max_det) > 0 else None
        return cfg

    def _build_union_postprocess_cfg(self) -> Dict[str, Any]:
        return {
            "do_clamp": True,
            "do_nms": bool(self.cfg.pp_do_nms),
            "nms_iou": float(self.cfg.pp_nms_iou) if float(self.cfg.pp_nms_iou) > 0.0 else float(self.cfg.nms_iou),
            "class_agnostic_nms": bool(self.cfg.pp_class_agnostic_nms),
            "max_det": int(self.cfg.pp_max_det) if int(self.cfg.pp_max_det) > 0 else None,
            "score_thr": 0.0,
        }

    def _build_forced_plan(self, ev: Evidence, K_full: int) -> Optional[ActionPlan]:
        mode = str(getattr(self.cfg, "force_verify_mode", "policy")).lower().strip()
        if mode in ("", "none", "policy"):
            return None

        interval = max(1, int(getattr(self.cfg, "force_verify_key_interval", 4)))
        do_verify = False
        reason = "forced_schedule_skip"

        if mode == "always":
            do_verify = int(K_full) > 0
            reason = "forced_always_verify" if do_verify else "forced_always_skip"
        elif mode == "keyframe":
            do_verify = (int(self.t) % interval) == 0 and int(K_full) > 0
            reason = "forced_keyframe_verify" if do_verify else "forced_keyframe_skip"
        else:
            return None

        return ActionPlan(
            mode="full_burst" if do_verify else "skip",
            risk=1.0 if do_verify else 0.0,
            K=int(K_full) if do_verify else 0,
            protect_frac=0.0,
            explore_source="forced_schedule",
            escalate_reason=str(reason),
            ttl={},
            cooldown={
                "forced_schedule": str(mode),
                "force_verify_mode": str(mode),
                "force_verify_key_interval": int(interval),
                "is_full_burst": bool(do_verify),
            },
        )

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

        backend = str(getattr(self.cfg, "verifier_backend", "sahi")).lower().strip()
        if backend == "crop_recheck":
            self.executor = CropRecheckExecutorV7(
                detector=self.detector,
                tiles=self.tiles,
                crop_margin_ratio=float(self.cfg.crop_margin_ratio),
                crop_input_size=int(self.cfg.crop_input_size),
                class_mapping=self.class_mapping,
            )
        else:
            self.executor = TileExecutorV7(
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

        # 4) Policy / forced schedule override
        forced_plan = self._build_forced_plan(ev, K_full=len(self.tiles))
        if forced_plan is None:
            plan = self.policy.step(ev, K_full=len(self.tiles))
        else:
            plan = forced_plan
            if hasattr(self.policy, "_prev_mode"):
                self.policy._prev_mode = str(plan.mode)

        # 5) Tile planning
        tile_plan = self.tile_planner.build(plan, ev, track_state, self.tiles)

        # 6) Execute
        # Keep full-frame YOLO output as baseline.
        base_preds = list(full_preds)

        verifier_backend = str(getattr(self.cfg, "verifier_backend", "sahi")).lower().strip()
        if len(tile_plan.tiles) > 0:
            verifier_preds = self.executor.run(
                frame=frame,
                image_id=int(image_id),
                tile_ids=tile_plan.tiles,
                score_threshold=float(score_threshold),
            )
            verifier_executed = True
        else:
            verifier_preds = []
            verifier_executed = False

        # 7) Output construction with YOLO fallback union (v7-only behavior)
        # Goal: preserve YOLO baseline quality while using SAHI (with confirmation/postprocess)
        # to add extra detections. This prevents recall collapse caused by confirmation/TTL.
        h, w = frame.shape[:2]

        # Baseline postprocess: clamp only; keep score threshold minimal to match detector output.
        base_cfg = get_pp_preset("clamp_only")
        base_cfg["score_thr"] = float(score_threshold) if float(score_threshold) > 0.0 else 0.0
        base_pp, base_pp_meta = postprocess_preds(base_preds, image_w=int(w), image_h=int(h), cfg=base_cfg)

        pp_meta: Dict[str, Any] = {"base": base_pp_meta}
        conf_meta: Dict[str, Any] = {
            "num_confirmed": 0,
            "num_unconfirmed": 0,
        }

        if verifier_executed:
            verifier_cfg = self._build_sahi_postprocess_cfg()
            verifier_pp, verifier_pp_meta = postprocess_preds(
                verifier_preds,
                image_w=int(w),
                image_h=int(h),
                cfg=verifier_cfg,
            )
            pp_meta["verifier"] = verifier_pp_meta
            if verifier_backend == "sahi":
                pp_meta["sahi"] = dict(verifier_pp_meta)

            if bool(self.use_confirmation):
                verifier_keep, verifier_unconf, _conf_meta = self.confirm_gate.apply(verifier_pp)
                log_conf = self._confirm_logger.update(t=self.t, preds=verifier_pp)
                conf_meta = dict(_conf_meta)
                conf_meta.update(log_conf)
            else:
                verifier_keep = verifier_pp
                verifier_unconf = []
                conf_meta = {
                    "num_confirmed": int(len(verifier_keep)),
                    "num_unconfirmed": 0,
                    "newly_confirmed": 0,
                    "ttc_list": [],
                }

            union_in = list(base_pp) + list(verifier_keep)
            union_cfg = self._build_union_postprocess_cfg()
            preds_final, union_meta = postprocess_preds(union_in, image_w=int(w), image_h=int(h), cfg=union_cfg)
            pp_meta["union"] = union_meta
            pp_meta["union"]["num_union_in"] = int(len(union_in))
            pp_meta["union"]["num_base_pp"] = int(len(base_pp))
            pp_meta["union"]["num_verifier_keep"] = int(len(verifier_keep))
            pp_meta["union"]["num_verifier_unconf"] = int(len(verifier_unconf))
            if verifier_backend == "sahi":
                pp_meta["union"]["num_sahi_keep"] = int(len(verifier_keep))
                pp_meta["union"]["num_sahi_unconf"] = int(len(verifier_unconf))
        else:
            preds_final = base_pp

        # 8) Feedback to policy (use confirmed preds)
        if forced_plan is None:
            self.policy.observe_result(
                plan=plan,
                ev=ev,
                has_target_pred=bool(len(preds_final) > 0),
                num_unconfirmed=int(conf_meta.get("num_unconfirmed", 0)),
                num_confirmed=int(conf_meta.get("num_confirmed", 0)),
            )

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
            "schema_version": str(V7_LOG_SCHEMA_VERSION),
            "method_version": str(V7_METHOD_VERSION),
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
                "ttl": plan.ttl if isinstance(getattr(plan, "ttl", None), dict) else {},
                "cooldown": plan.cooldown if isinstance(getattr(plan, "cooldown", None), dict) else {},
            },
            "tile_plan": tile_plan.to_dict(),
            "verifier_backend": str(verifier_backend),
            "verifier_executed": bool(verifier_executed),
            "num_verifier_regions": int(len(tile_plan.tiles)),
            "verifier_cost_proxy": float(len(tile_plan.tiles)),
            "evidence": ev.to_dict(),
            "pp": pp_meta,
            "confirm": conf_meta,
            "num_unconfirmed": int(conf_meta.get("num_unconfirmed", 0)),
        }

        # Enforce log schema contract (controller I/O)
        for k in list(V7_LOG_REQUIRED_KEYS):
            if k not in log:
                log[k] = {}

        # Enforce minimal metadata keys
        for k in list(V7_LOG_REQUIRED_META_KEYS):
            if k not in log:
                # Use safe defaults for missing meta
                if k == "t":
                    log[k] = int(self.t)
                elif k == "image_id":
                    log[k] = int(image_id)
                elif k == "schema_version":
                    log[k] = str(V7_LOG_SCHEMA_VERSION)
                elif k == "method_version":
                    log[k] = str(V7_METHOD_VERSION)
                else:
                    log[k] = None

        self.logs.append(log)
        self.t += 1

        short = {
            "do_sahi": bool(verifier_executed),
            "mode": str(plan.mode),
            "K": int(len(tile_plan.tiles)),
            "tiles_selected": int(len(tile_plan.tiles)),
            "verifier_backend": str(verifier_backend),
            "verifier_executed": bool(verifier_executed),
            "num_verifier_regions": int(len(tile_plan.tiles)),
            "verifier_cost_proxy": float(len(tile_plan.tiles)),
            "seed_empty": bool(seed_empty),
            "no_det_bucket": str(bucket),
            "pp": pp_meta,
            "confirm": conf_meta,
            "num_unconfirmed": int(conf_meta.get("num_unconfirmed", 0)),
        }

        return preds_final, short

    def reset_for_new_video(self, video_id: Optional[str]) -> None:
        """Reset temporal states at clip boundary.

        This is called by the runner when dataset video_id changes.
        It prevents motion buffer shape mismatch and recomputes tile grid
        when resolution changes between clips.
        """

        if hasattr(self.policy, "reset_for_new_video"):
            self.policy.reset_for_new_video()
        else:
            if hasattr(self.policy, "_verify_left"):
                self.policy._verify_left = 0
            if hasattr(self.policy, "_verify_cooldown_left"):
                self.policy._verify_cooldown_left = 0

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
        self.tracker = TrackerV7(
            category_whitelist=list(self.cfg.target_category_ids) if self.cfg.target_category_ids is not None else None,
            class_aware=False,
        )

        # Reset policy internal state (cooldowns/prev_mode) for a clean clip start.
        self.policy = self._build_policy()

        self.confirm_gate.reset()
        self._confirm_logger.reset()
