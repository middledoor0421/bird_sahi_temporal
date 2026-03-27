# sahi_temporal/v5/controller.py
# Python 3.9 compatible. Comments in English only.

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ControllerConfigV5:
    m_escape_soft: int = 2  # M
    subset_min_interval: int = 0  # optional throttle (frames)


class SAHIControllerV5:
    """
    Final v5 policy (per spec):
      - escape_soft singletons are ignored:
          if escape_soft_streak < M -> skip
          else -> subset
      - full is NOT decided here by risk. Full is handled in engine by:
          escape_hard OR subset_fail_streak >= F
    """

    def __init__(self, cfg: Optional[ControllerConfigV5] = None) -> None:
        self.cfg = cfg if cfg is not None else ControllerConfigV5()
        self.last_subset_t = -10**9

    def decide_mode(
        self,
        t: int,
        escape_soft_streak: int,
        force_subset: bool,
    ) -> str:
        """
        Returns: "skip" or "subset"
        """
        if force_subset:
            self.last_subset_t = int(t)
            return "subset"

        if int(escape_soft_streak) >= int(self.cfg.m_escape_soft):
            # Optional throttle
            if int(self.cfg.subset_min_interval) > 0:
                if (int(t) - int(self.last_subset_t)) < int(self.cfg.subset_min_interval):
                    return "skip"
            self.last_subset_t = int(t)
            return "subset"

        return "skip"
