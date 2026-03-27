# sahi_temporal/v7/tile_planner.py
# Python 3.9 compatible. Comments in English only.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .schemas import ActionPlan, Evidence, TilePlan
from .tiles_explore import select_explore_tiles, ExploreMemory
from .tiles_protect import build_protect_tiles
from .tracker import TrackSummary


def _no_det_bucket(no_det_streak: int) -> str:
    v = int(no_det_streak)
    if v <= 1:
        return "0_1"
    if v <= 5:
        return "2_5"
    return "6p"


def _to_vec(x: Any, n: int) -> np.ndarray:
    if x is None:
        return np.zeros((n,), dtype=np.float32)
    if isinstance(x, np.ndarray):
        v = x.astype(np.float32).reshape(-1)
    else:
        try:
            v = np.asarray(list(x), dtype=np.float32).reshape(-1)
        except Exception:
            v = np.zeros((0,), dtype=np.float32)

    if v.size == n:
        return v

    out = np.zeros((n,), dtype=np.float32)
    m = min(int(n), int(v.size))
    if m > 0:
        out[:m] = v[:m]
    return out


def _norm_max(v: np.ndarray) -> np.ndarray:
    if v.size == 0:
        return v
    mx = float(np.max(v))
    if mx <= 0.0:
        return v
    return v / (mx + 1e-6)


@dataclass
class TilePlannerConfigV7:
    cand_extra: int = 6
    protect_neighbor_hops: int = 1
    score_mode: str = "mean"
    apply_row_quota: bool = False
    quota_mode: str = "top1"
    use_memory: bool = False
    memory_ttl: int = 0


class TilePlannerV7:
    def __init__(self, cfg: Optional[TilePlannerConfigV7] = None) -> None:
        self.cfg = cfg if cfg is not None else TilePlannerConfigV7()
        self.memory = ExploreMemory()

    def build(
        self,
        plan: ActionPlan,
        ev: Evidence,
        track_state: TrackSummary,
        tiles: List[Dict[str, int]],
    ) -> TilePlan:
        K = int(plan.K)
        exp_cfg = plan.cooldown.get('exp_cfg', {}) if isinstance(plan.cooldown, dict) else {}
        is_full_burst = False
        if isinstance(plan.cooldown, dict):
            if bool(plan.cooldown.get('is_full_burst', False)):
                is_full_burst = True
            if isinstance(exp_cfg, dict) and bool(exp_cfg.get('is_full_burst', False)):
                is_full_burst = True
        if str(getattr(plan, 'mode', '')).lower() == 'full_burst':
            is_full_burst = True
        if is_full_burst:
            all_ids = list(range(len(tiles)))
            return TilePlan(tiles=all_ids, tiles_protect=[], tiles_explore=all_ids, tile_mode='full_burst', K_protect=0, K_explore=len(all_ids), coverage_meta={'exp_cfg': exp_cfg, 'is_full_burst': True})
        if K <= 0:
            return TilePlan(tiles=[], tiles_protect=[], tiles_explore=[], tile_mode='off', K_protect=0, K_explore=0, coverage_meta={})

        K_protect = int(round(float(plan.protect_frac) * float(K)))
        K_protect = max(0, min(int(K_protect), int(K)))
        K_explore = int(K - K_protect)

        # Condition (computed locally; Evidence has no cond field)
        seed_empty = (int(ev.track.num_active) == 0) and (not bool(ev.det.has_det))
        bucket = _no_det_bucket(int(ev.det.no_det_streak))

        # Read experiment knobs from plan.cooldown['exp_cfg'] (robust)
        exp_cfg: Dict[str, Any] = {}
        if isinstance(plan.cooldown, dict):
            ec = plan.cooldown.get("exp_cfg", None)
            if isinstance(ec, dict):
                exp_cfg = ec

        score_mode = str(exp_cfg.get("score_mode", getattr(plan, "score_mode", self.cfg.score_mode)))
        apply_row_quota = bool(exp_cfg.get("apply_row_quota", getattr(plan, "apply_row_quota", self.cfg.apply_row_quota)))
        quota_mode = str(exp_cfg.get("quota_mode", getattr(plan, "quota_mode", self.cfg.quota_mode)))
        use_memory = bool(exp_cfg.get("use_memory", getattr(plan, "use_memory", self.cfg.use_memory)))
        memory_ttl = int(exp_cfg.get("memory_ttl", getattr(plan, "memory_ttl", self.cfg.memory_ttl)))

        # Step memory every frame
        self.memory.step()

        # Protect tiles (keep existing behavior; safe even when seed_empty)
        tracks = track_state.tracks if track_state is not None else []
        protect, protect_meta = build_protect_tiles(
            tracks=tracks,
            tiles=tiles,
            K_protect=K_protect,
            neighbor_hops=int(self.cfg.protect_neighbor_hops),
        )

        # Build explore scalar score per tile.
        # EvidenceBuilderV7 stores raw arrays in ev.raw.
        n_tiles = len(tiles)
        diff_scores = _to_vec(ev.raw.get("diff_scores", None), n_tiles)
        motion_scores = _to_vec(ev.raw.get("motion_scores", None), n_tiles)

        src = str(plan.explore_source).lower().strip()
        if src == "motion":
            base = motion_scores
        elif src == "diff":
            base = diff_scores
        else:
            base = _norm_max(diff_scores) + _norm_max(motion_scores)

        tile_scores = [float(v) for v in base.reshape(-1).tolist()]

        explore = select_explore_tiles(
            tiles=tiles,
            tile_scores=tile_scores,
            k_explore=int(K_explore),
            score_mode=score_mode,
            apply_row_quota=apply_row_quota,
            quota_mode=quota_mode,
            use_memory=use_memory,
            memory_ttl=memory_ttl,
            cond_seed_empty=bool(seed_empty),
            cond_no_det_bucket=str(bucket),
            cand_extra=int(self.cfg.cand_extra),
            memory=self.memory,
        )

        final = list(dict.fromkeys([int(x) for x in protect] + [int(x) for x in explore]))
        if len(final) > K:
            final = final[:K]


        tile_mode = 'mixed'
        if len(final) == 0:
            tile_mode = 'off'
        elif len(protect) > 0 and len(explore) == 0:
            tile_mode = 'protect'
        elif len(explore) > 0 and len(protect) == 0:
            tile_mode = 'explore'
        else:
            tile_mode = 'mixed'
        return TilePlan(
            tiles=[int(x) for x in final],
            tiles_protect=[int(x) for x in protect],
            tiles_explore=[int(x) for x in explore],
            tile_mode=str(tile_mode),
            K_protect=int(K_protect),
            K_explore=int(K_explore),
            coverage_meta={
                "seed_empty": bool(seed_empty),
                "no_det_bucket": str(bucket),
                "protect_meta": protect_meta,
                "exp_cfg": exp_cfg,
            },
        )
