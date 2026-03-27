# sahi_temporal/v4/controller.py
# Python 3.9 compatible. Comments in English only.

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ControllerConfigV4:
    heartbeat_max_gap: int = 10
    min_key_gap: int = 2

    risk_tau: float = 1.0
    risk_tau_full: float = 2.0

    full_on_tracking_fail: bool = True


class SAHIControllerV4:
    def __init__(self, cfg: Optional[ControllerConfigV4] = None) -> None:
        self.cfg = cfg if cfg is not None else ControllerConfigV4()
        self.last_keyframe = -10**9

    def decide(
        self,
        t: int,
        risk_score: float,
        tracking_fail: bool,
        escape_soft: bool,
        escape_hard: bool,
        strong_skip: bool,
        is_anchor: bool,
    ) -> Dict:
        gap = int(t - self.last_keyframe)

        # 1) Hard full first
        if escape_hard:
            self.last_keyframe = int(t)
            return {"do_sahi": True, "mode": "full", "risk_high": True, "is_anchor": False}

        if is_anchor:
            self.last_keyframe = int(t)
            return {"do_sahi": True, "mode": "full", "risk_high": True, "is_anchor": True}

        # 2) min gap (after hard constraints)
        if gap < int(self.cfg.min_key_gap):
            return {"do_sahi": False, "mode": "skip", "risk_high": False, "is_anchor": False}

        # 3) Soft escape: subset-first, skip forbidden
        # If you really want strong_skip to override, keep it here, but generally escape_soft implies "not low_motion".
        if escape_soft:
            self.last_keyframe = int(t)
            return {"do_sahi": True, "mode": "subset", "risk_high": True, "is_anchor": False}

        # 4) strong skip (safe region)
        if strong_skip:
            return {"do_sahi": False, "mode": "skip", "risk_high": False, "is_anchor": False}

        # 5) tracking fail -> full
        if tracking_fail and bool(self.cfg.full_on_tracking_fail):
            self.last_keyframe = int(t)
            return {"do_sahi": True, "mode": "full", "risk_high": True, "is_anchor": False}

        # 6) risk-based subset/full
        r = float(risk_score)
        if r >= float(self.cfg.risk_tau_full):
            self.last_keyframe = int(t)
            return {"do_sahi": True, "mode": "full", "risk_high": True, "is_anchor": False}

        if r >= float(self.cfg.risk_tau):
            self.last_keyframe = int(t)
            return {"do_sahi": True, "mode": "subset", "risk_high": False, "is_anchor": False}

        return {"do_sahi": False, "mode": "skip", "risk_high": False, "is_anchor": False}
