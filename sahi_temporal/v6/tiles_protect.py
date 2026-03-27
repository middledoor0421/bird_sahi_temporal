# sahi_temporal/v6/tiles_protect.py
# Python 3.9 compatible. Comments in English only.

from typing import Dict, List, Set, Tuple

from .tracker import Track, bbox_center_xywh


def _point_in_tile(cx: float, cy: float, tile: Dict[str, int]) -> bool:
    x = float(tile["x"])
    y = float(tile["y"])
    w = float(tile["w"])
    h = float(tile["h"])
    return (x <= cx < x + w) and (y <= cy < y + h)


def _tile_ids_covering_point(cx: float, cy: float, tiles: List[Dict[str, int]]) -> Set[int]:
    hits: Set[int] = set()
    for i, t in enumerate(tiles):
        if _point_in_tile(cx, cy, t):
            hits.add(int(i))
    return hits


def _build_grid_index(tiles: List[Dict[str, int]]) -> Tuple[Dict[int, Tuple[int, int]], Dict[Tuple[int, int], int]]:
    xs = sorted(set([int(t["x"]) for t in tiles]))
    ys = sorted(set([int(t["y"]) for t in tiles]))
    x_to_c = {x: i for i, x in enumerate(xs)}
    y_to_r = {y: i for i, y in enumerate(ys)}

    id_to_rc: Dict[int, Tuple[int, int]] = {}
    rc_to_id: Dict[Tuple[int, int], int] = {}
    for tid, t in enumerate(tiles):
        r = y_to_r[int(t["y"])]
        c = x_to_c[int(t["x"])]
        id_to_rc[int(tid)] = (int(r), int(c))
        rc_to_id[(int(r), int(c))] = int(tid)
    return id_to_rc, rc_to_id


def _neighbor_expand(
    seeds: Set[int],
    id_to_rc: Dict[int, Tuple[int, int]],
    rc_to_id: Dict[Tuple[int, int], int],
    hops: int,
) -> Set[int]:
    if hops <= 0 or len(seeds) == 0:
        return set()

    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    out: Set[int] = set()
    frontier: Set[int] = set(seeds)

    for _ in range(int(hops)):
        new_frontier: Set[int] = set()
        for tid in frontier:
            if tid not in id_to_rc:
                continue
            r, c = id_to_rc[tid]
            for dr, dc in dirs:
                key = (int(r + dr), int(c + dc))
                if key in rc_to_id:
                    nid = int(rc_to_id[key])
                    if nid not in out and nid not in seeds:
                        out.add(nid)
                        new_frontier.add(nid)
        frontier = new_frontier
        if len(frontier) == 0:
            break

    return out


def build_protect_tiles(
    tracks: List[Track],
    tiles: List[Dict[str, int]],
    K_protect: int,
    neighbor_hops: int = 1,
) -> Tuple[List[int], Dict[str, object]]:
    """Build protect tiles around active tracks.

    Strategy (initial):
        - Take track centers
        - Map to tile ids
        - Optionally add 4-neighbors (grid adjacency)
        - Fill up to K_protect

    Returns:
        tiles_protect, coverage_meta
    """
    K_protect = max(0, int(K_protect))
    if K_protect == 0 or tiles is None or len(tiles) == 0 or tracks is None or len(tracks) == 0:
        return [], {"covered_track_ids": []}

    id_to_rc, rc_to_id = _build_grid_index(tiles)

    # Prioritize tracks by last_score (desc)
    tracks_sorted = sorted(tracks, key=lambda tr: float(tr.last_score), reverse=True)

    chosen: List[int] = []
    chosen_set: Set[int] = set()
    covered: List[int] = []

    for tr in tracks_sorted:
        if len(chosen) >= K_protect:
            break

        cx, cy = bbox_center_xywh(tr.bbox_xywh)
        seeds = _tile_ids_covering_point(cx, cy, tiles)
        if len(seeds) == 0:
            continue

        # Add center tile(s) first
        for tid in sorted(list(seeds)):
            if len(chosen) >= K_protect:
                break
            if tid in chosen_set:
                continue
            chosen.append(int(tid))
            chosen_set.add(int(tid))

        # Then add neighbors
        neigh = _neighbor_expand(seeds, id_to_rc, rc_to_id, hops=int(neighbor_hops))
        for tid in sorted(list(neigh)):
            if len(chosen) >= K_protect:
                break
            if tid in chosen_set:
                continue
            chosen.append(int(tid))
            chosen_set.add(int(tid))

        covered.append(int(tr.track_id))

    return chosen, {"covered_track_ids": covered}
