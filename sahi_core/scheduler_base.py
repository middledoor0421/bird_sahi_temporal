# sahi_core/scheduler_base.py
# Base class for tile scheduler.
# Python 3.9 compatible.

from typing import List, Dict


class TileSchedulerBase:
    """Base scheduler: decide which tiles to run SAHI on."""

    def select(self, tiles: List[Dict]) -> List[int]:
        """
        Args:
            tiles: list of tile info dicts (e.g., {"x","y","w","h"})

        Returns:
            list of indices of tiles to use
        """
        raise NotImplementedError()


class FullTileScheduler(TileSchedulerBase):
    """Scheduler that selects all tiles (full SAHI baseline)."""

    def select(self, tiles: List[Dict]) -> List[int]:
        return list(range(len(tiles)))
