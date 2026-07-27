"""
regimes.py
==========

Helpers for constructing *solvable* experimental regimes.

The goal is to make it easy to validate agent competence in a matched-model
setting before adding hazards, volatility, learning, or attention.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from aif_paper1.affectworld.affect_world_env import AffectWorldEnv
from aif_paper1.affectworld.configs import EnvConfig
from aif_paper1.affectworld.context_layout import build_default_context_resource_map
from aif_paper1.affectworld.helpers import Resource


def _manhattan(a: int, b: int, w: int) -> int:
    """Compute Manhattan distance between two flat grid indices."""

    ax, ay = a % w, a // w
    bx, by = b % w, b // w
    return abs(ax - bx) + abs(ay - by)


def _positions_within_radius(start: int, w: int, h: int, radius: int) -> list[int]:
    """Enumerate cells whose Manhattan distance from `start` is at most `radius`."""

    n = w * h
    if not (0 <= start < n):
        raise ValueError(f"start index out of range: {start}")
    radius = int(radius)
    if radius < 0:
        raise ValueError("radius must be >= 0")
    return [i for i in range(n) if _manhattan(i, start, w) <= radius]


@dataclass(frozen=True)
class SimpleRegime:
    """
    A minimal regime that guarantees FOOD and WATER are nearby.

    Notes:
    - Under `frozen` seasons, resources are placed across *all* contexts to keep
      the matched-model competence bank simple.
    - Under dynamic seasonal modes, the regime keeps the context-dependent base
      map and overlays nearby solvable anchors without collapsing all contexts
      into the same slice.
    """

    start_pos: int
    food_positions: tuple[int, ...]
    water_positions: tuple[int, ...]
    sleep_positions: tuple[int, ...]


def _normalize_positions(
    values: Sequence[int] | None,
    *,
    label: str,
    n_cells: int,
) -> tuple[int, ...] | None:
    """Validate explicit placement lists and return them as canonical tuples."""

    if values is None:
        return None
    out = tuple(int(v) for v in values)
    for pos in out:
        if not (0 <= int(pos) < int(n_cells)):
            raise ValueError(f"{label} position out of range: {pos}")
    if len(set(out)) != len(out):
        raise ValueError(f"{label} positions must be unique.")
    return out


def simple_regime_to_dict(layout: SimpleRegime) -> dict[str, Any]:
    """Serialize a `SimpleRegime` to a JSON-friendly mapping."""

    return {
        "start_pos": int(layout.start_pos),
        "food_positions": [int(p) for p in layout.food_positions],
        "water_positions": [int(p) for p in layout.water_positions],
        "sleep_positions": [int(p) for p in layout.sleep_positions],
    }


def simple_regime_from_dict(data: Mapping[str, Any]) -> SimpleRegime:
    """Parse a `SimpleRegime` from stored JSON-ish data."""

    return SimpleRegime(
        start_pos=int(data["start_pos"]),
        food_positions=tuple(int(p) for p in data.get("food_positions", [])),
        water_positions=tuple(int(p) for p in data.get("water_positions", [])),
        sleep_positions=tuple(int(p) for p in data.get("sleep_positions", [])),
    )


def simple_regime_diagnostics(layout: SimpleRegime, *, width: int, height: int) -> dict[str, Any]:
    """Summarize distances and counts that make a simple regime easy to reason about."""

    x = int(layout.start_pos) % int(width)
    y = int(layout.start_pos) // int(width)

    def _nearest(positions: Sequence[int]) -> int | None:
        """Return distance to the nearest position in the supplied collection."""

        if len(positions) <= 0:
            return None
        return min(_manhattan(int(layout.start_pos), int(pos), int(width)) for pos in positions)

    return {
        "start_pos": int(layout.start_pos),
        "start_xy": [int(x), int(y)],
        "nearest_food_distance": _nearest(layout.food_positions),
        "nearest_water_distance": _nearest(layout.water_positions),
        "nearest_sleep_distance": _nearest(layout.sleep_positions),
        "start_edge_distance": int(min(x, y, int(width) - 1 - x, int(height) - 1 - y)),
        "food_count": int(len(layout.food_positions)),
        "water_count": int(len(layout.water_positions)),
        "sleep_count": int(len(layout.sleep_positions)),
    }


def load_simple_regime_bank(path: str | Path) -> list[dict[str, Any]]:
    """Load a bank of stored layouts and normalize each entry's core fields."""

    bank_path = Path(path).expanduser().resolve()
    payload = json.loads(bank_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        items = payload.get("layouts", [])
    elif isinstance(payload, list):
        items = payload
    else:
        raise ValueError(f"Unsupported layout bank payload type: {type(payload)!r}")

    out: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise ValueError(f"Layout bank entry {idx} must be a mapping.")
        layout = simple_regime_from_dict(item)
        rec = dict(item)
        rec["layout_id"] = str(item.get("layout_id", f"layout{idx:03d}"))
        rec["start_pos"] = int(layout.start_pos)
        rec["food_positions"] = [int(p) for p in layout.food_positions]
        rec["water_positions"] = [int(p) for p in layout.water_positions]
        rec["sleep_positions"] = [int(p) for p in layout.sleep_positions]
        out.append(rec)
    return out


def save_simple_regime_bank(
    path: str | Path,
    *,
    layouts: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Persist a regime bank payload with optional metadata and normalized formatting."""

    bank_path = Path(path).expanduser().resolve()
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": dict(metadata or {}),
        "layouts": [dict(item) for item in layouts],
    }
    bank_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def apply_simple_regime(
    env: AffectWorldEnv,
    *,
    start_pos: Optional[int] = None,
    radius: int = 2,
    n_food: int = 2,
    n_water: int = 2,
    n_sleep: int = 1,
    seed: Optional[int] = None,
    food_positions: Sequence[int] | None = None,
    water_positions: Sequence[int] | None = None,
    sleep_positions: Sequence[int] | None = None,
) -> SimpleRegime:
    """
    Mutate `env.res_map` to a simple, solvable layout and return placements.

    Important:
    - Call this *before* constructing an agent whose A/B matrices are built
      from `env.res_map`.
    """
    n_cells = env.w * env.h
    if start_pos is None:
        start_pos = int(env.rng.integers(n_cells))
    start_pos = int(start_pos)

    rng = np.random.default_rng(seed)
    explicit_food = _normalize_positions(food_positions, label="food", n_cells=n_cells)
    explicit_water = _normalize_positions(water_positions, label="water", n_cells=n_cells)
    explicit_sleep = _normalize_positions(sleep_positions, label="sleep", n_cells=n_cells)

    if str(getattr(env, "season_mode", "frozen")).lower() == "frozen":
        # Fixed-layout competence path: identical resources across contexts.
        env.res_map[:] = Resource.AIR
    else:
        # Dynamic seasonal path: preserve context-specific opportunity structure.
        env.res_map[:] = build_default_context_resource_map(
            width=int(env.w),
            height=int(env.h),
            hill_idx=int(env.hill_idx),
            n_ctx=int(getattr(env, "n_ctx", 4)),
            lake_frac=float(getattr(env, "lake_frac", 0.0)),
        )
    env.res_map[int(env.hill_idx), :] = Resource.AIR
    env.res_map[start_pos, :] = Resource.AIR

    if explicit_food is not None or explicit_water is not None or explicit_sleep is not None:
        food_positions_t = explicit_food or ()
        water_positions_t = explicit_water or ()
        sleep_positions_t = explicit_sleep or ()
        all_positions = tuple(food_positions_t) + tuple(water_positions_t) + tuple(sleep_positions_t)
        if len(set(all_positions)) != len(all_positions):
            raise ValueError("Explicit SimpleRegime positions must be unique across resources.")
        reserved = {int(start_pos), int(env.hill_idx)}
        overlap = reserved.intersection(int(p) for p in all_positions)
        if overlap:
            raise ValueError(f"Explicit SimpleRegime positions overlap reserved tiles: {sorted(overlap)}")
    else:
        candidates = _positions_within_radius(start_pos, env.w, env.h, radius)
        candidates = [p for p in candidates if p not in {start_pos, int(env.hill_idx)}]
        if not candidates:
            raise RuntimeError("No candidate positions for SimpleRegime placement.")

        # Sample without replacement from the candidate set.
        n_food = max(0, int(n_food))
        n_water = max(0, int(n_water))
        n_sleep = max(0, int(n_sleep))
        n_total = n_food + n_water + n_sleep
        if n_total > len(candidates):
            raise RuntimeError(
                f"SimpleRegime placement needs {n_total} unique tiles but only "
                f"{len(candidates)} candidates available (radius={radius})."
            )
        chosen = rng.choice(np.array(candidates, dtype=np.int64), size=n_total, replace=False).tolist()

        food_positions_t = tuple(int(p) for p in chosen[:n_food])
        water_positions_t = tuple(int(p) for p in chosen[n_food : n_food + n_water])
        sleep_positions_t = tuple(int(p) for p in chosen[n_food + n_water :])

    for p in food_positions_t:
        env.res_map[p, :] = Resource.FOOD
    for p in water_positions_t:
        env.res_map[p, :] = Resource.WATER
    for p in sleep_positions_t:
        env.res_map[p, :] = Resource.SLEEP

    # Keep landmarks sane.
    env.res_map[int(env.hill_idx), :] = Resource.AIR
    env.res_map[start_pos, :] = Resource.AIR

    return SimpleRegime(
        start_pos=start_pos,
        food_positions=food_positions_t,
        water_positions=water_positions_t,
        sleep_positions=sleep_positions_t,
    )


def make_env(cfg: EnvConfig) -> AffectWorldEnv:
    """Construct an `AffectWorldEnv` from an `EnvConfig`."""
    return AffectWorldEnv(
        width=cfg.width,
        height=cfg.height,
        year_length=cfg.year_length,
        lake_frac=cfg.lake_frac,
        seed=cfg.seed,
        season_mode=cfg.season_mode,
        season_stay_prob=cfg.season_stay_prob,
        predator_enabled=cfg.predator_enabled,
        predator_prob=cfg.predator_prob,
        mutation_enabled=cfg.mutation_enabled,
        mutation_prob=cfg.mutation_prob,
        resource_field_enabled=cfg.resource_field_enabled,
        resource_field_resources=cfg.resource_field_resources,
        resource_field_include_predator=cfg.resource_field_include_predator,
        resource_field_mode=cfg.resource_field_mode,
        resource_field_tau=cfg.resource_field_tau,
        resource_field_radius=cfg.resource_field_radius,
        resource_field_bins=cfg.resource_field_bins,
        resource_field_obs_true_prob=cfg.resource_field_obs_true_prob,
    )


__all__ = [
    "SimpleRegime",
    "apply_simple_regime",
    "load_simple_regime_bank",
    "make_env",
    "save_simple_regime_bank",
    "simple_regime_diagnostics",
    "simple_regime_from_dict",
    "simple_regime_to_dict",
]
