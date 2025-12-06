# sahi_ext/budget_scheduler.py
# Budget scheduler: limit number of tiles per frame.
# Python 3.9 compatible.

from typing import List


class BudgetScheduler:
    """
    Simple budget scheduler: limit the number of tiles to max_tiles.
    More advanced logic (utility/cost sorting) can be added later.
    """

    def __init__(self, max_tiles: int) -> None:
        self.max_tiles = max(1, int(max_tiles))

    def select(self, tile_idxs: List[int]) -> List[int]:
        # Currently just keep the first max_tiles indices.
        return tile_idxs[: self.max_tiles]
