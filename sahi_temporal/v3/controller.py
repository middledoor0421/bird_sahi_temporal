# sahi_temporal/v3/controller.py
# Python 3.9 compatible. Comments in English only.

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class ControllerConfig:
    heartbeat_max_gap: int = 10
    min_key_gap: int = 2

    # budget for tiles in SAHI-on frame
    max_tiles: int = 6

    # expansion policy
    neighbor_hops: int = 1
    base_seed_expand: bool = True

    # mode thresholds
    risk_tau: float = 1.0


class SAHIController:
    """
    Decide:
      - do_sahi (bool)
      - mode: "skip" | "full" | "subset"
      - tile_mode: full list or subset list (computed later)
    """

    def __init__(self, cfg: Optional[ControllerConfig] = None) -> None:
        self.cfg = cfg if cfg is not None else ControllerConfig()
        self.last_keyframe = -10**9

    def decide(
        self,
        t: int,
        risk_score: float,
        tracking_fail: bool,
    ) -> Dict:
        gap = int(t - self.last_keyframe)
        is_anchor = gap >= int(self.cfg.heartbeat_max_gap)

        if (not is_anchor) and gap < int(self.cfg.min_key_gap):
            return {"do_sahi": False, "mode": "skip", "is_anchor": False}

        # Risk gating
        do_sahi = bool(is_anchor or tracking_fail or (risk_score >= float(self.cfg.risk_tau)))

        if not do_sahi:
            return {"do_sahi": False, "mode": "skip", "is_anchor": False}

        # If anchor, prefer full
        if is_anchor:
            self.last_keyframe = int(t)
            return {"do_sahi": True, "mode": "full", "is_anchor": True}

        # Otherwise subset
        self.last_keyframe = int(t)
        return {"do_sahi": True, "mode": "subset", "is_anchor": False}
