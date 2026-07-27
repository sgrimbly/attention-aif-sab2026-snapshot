"""
Shared context-dependent resource layout construction for AffectWorld.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from aif_paper1.affectworld.helpers import Resource


def xy_to_idx(x: int, y: int, width: int) -> int:
    return int(y) * int(width) + int(x)


def idx_to_xy(idx: int, width: int) -> tuple[int, int]:
    return int(idx) % int(width), int(idx) // int(width)


def _sector_bounds(width: int, height: int) -> tuple[tuple[int, int, int, int], ...]:
    x_mid = max(1, int(width) // 2)
    y_mid = max(1, int(height) // 2)
    return (
        (0, x_mid, 0, y_mid),
        (x_mid, int(width), 0, y_mid),
        (0, x_mid, y_mid, int(height)),
        (x_mid, int(width), y_mid, int(height)),
    )


def _ordered_sector_cells(
    *,
    width: int,
    height: int,
    hill_idx: int,
    neutral_positions: set[int],
) -> tuple[list[int], ...]:
    sectors: list[list[int]] = []
    hill_x, hill_y = idx_to_xy(int(hill_idx), int(width))
    for x0, x1, y0, y1 in _sector_bounds(int(width), int(height)):
        cx = 0.5 * float(x0 + max(x0, x1 - 1))
        cy = 0.5 * float(y0 + max(y0, y1 - 1))
        cells: list[tuple[tuple[float, float, int], int]] = []
        for y in range(int(y0), int(y1)):
            for x in range(int(x0), int(x1)):
                idx = xy_to_idx(x, y, int(width))
                if idx in neutral_positions:
                    continue
                dist_sector = abs(float(x) - cx) + abs(float(y) - cy)
                dist_hill = abs(int(x) - int(hill_x)) + abs(int(y) - int(hill_y))
                cells.append(((dist_sector, -float(dist_hill), idx), idx))
        cells.sort(key=lambda item: item[0])
        sectors.append([idx for _, idx in cells])
    return tuple(sectors)


def _interleave_sector_candidates(sector_cells: tuple[list[int], ...], sector_order: Iterable[int]) -> list[int]:
    ordered = [sector_cells[int(idx)] for idx in sector_order]
    out: list[int] = []
    depth = max((len(cells) for cells in ordered), default=0)
    for level in range(depth):
        for cells in ordered:
            if level < len(cells):
                out.append(int(cells[level]))
    return out


def _place_from_pool(
    layout: np.ndarray,
    *,
    ctx: int,
    resource: Resource,
    pool: Iterable[int],
    limit: int,
    used: set[int],
) -> None:
    placed = 0
    for idx in pool:
        pos = int(idx)
        if pos in used:
            continue
        layout[pos, int(ctx)] = int(resource)
        used.add(pos)
        placed += 1
        if placed >= int(limit):
            break


def build_default_context_resource_map(
    *,
    width: int,
    height: int,
    hill_idx: int,
    n_ctx: int = 4,
    lake_frac: float = 0.0,
    neutral_positions: Iterable[int] = (),
) -> np.ndarray:
    """
    Build the maintained default resource map.

    The map is context-dependent: Food, Water, and Sleep rotate across sectors
    so the hill's season cue changes the expected opportunity structure.
    """

    n = int(width) * int(height)
    neutral = {int(hill_idx), *(int(v) for v in neutral_positions if 0 <= int(v) < n)}
    layout = np.full((n, int(n_ctx)), int(Resource.AIR), dtype=np.int8)
    sector_cells = _ordered_sector_cells(width=int(width), height=int(height), hill_idx=int(hill_idx), neutral_positions=neutral)

    food_tiles = 2 if n >= 9 else 1
    water_tiles = max(2 if n >= 9 else 1, int(round(max(0.0, float(lake_frac)) * float(n))))
    sleep_tiles = 1 if n > 1 else 0

    for ctx in range(int(n_ctx)):
        used = set(neutral)
        food_order = (ctx, (ctx + 2) % 4, (ctx + 1) % 4, (ctx + 3) % 4)
        water_order = ((ctx + 1) % 4, (ctx + 3) % 4, ctx, (ctx + 2) % 4)
        sleep_order = ((ctx + 2) % 4, ctx, (ctx + 1) % 4, (ctx + 3) % 4)

        _place_from_pool(
            layout,
            ctx=int(ctx),
            resource=Resource.SLEEP,
            pool=_interleave_sector_candidates(sector_cells, sleep_order),
            limit=int(sleep_tiles),
            used=used,
        )
        _place_from_pool(
            layout,
            ctx=int(ctx),
            resource=Resource.FOOD,
            pool=_interleave_sector_candidates(sector_cells, food_order),
            limit=int(food_tiles),
            used=used,
        )
        _place_from_pool(
            layout,
            ctx=int(ctx),
            resource=Resource.WATER,
            pool=_interleave_sector_candidates(sector_cells, water_order),
            limit=int(water_tiles),
            used=used,
        )

    if neutral:
        layout[list(sorted(neutral)), :] = int(Resource.AIR)
    return layout


__all__ = [
    "build_default_context_resource_map",
    "idx_to_xy",
    "xy_to_idx",
]
