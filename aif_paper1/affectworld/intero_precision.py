"""
intero_precision.py
===================

Shared helpers for interoceptive precision allocation across need channels.

The project uses four Type-II need channels (hunger, thirst, fatigue, suffocation)
plus an optional "no foreground" attention state. This module centralizes the
precision-budget logic so process sampling and agent likelihood construction stay
aligned.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import jax.numpy as jnp


BudgetMode = Literal["none", "soft", "hard"]

_MODE_TO_INDEX: dict[str, int] = {
    "none": 0,
    "soft": 1,
    "hard": 2,
}


def normalize_budget_mode(mode: str) -> str:
    """Canonicalize and validate the named interoceptive precision-budget mode."""

    m = str(mode).strip().lower()
    if m in {"", "none"}:
        return "none"
    if m not in _MODE_TO_INDEX:
        raise ValueError(f"Unknown intero_budget_mode={mode!r}; expected none|soft|hard")
    return m


def budget_mode_to_index(mode: str) -> int:
    """Map a validated budget mode to the compact integer used in JAX code paths."""

    return int(_MODE_TO_INDEX[normalize_budget_mode(mode)])


def _clip01(x: float) -> float:
    """Clamp a scalar precision-like parameter to the closed unit interval."""

    return float(max(0.0, min(1.0, float(x))))


def _total_budget_default(*, n_channels: int, base_precision: float, att_precision: float) -> float:
    """Reproduce the legacy total precision implied by base and attended channel values."""

    n = max(1, int(n_channels))
    return float(att_precision + max(0, n - 1) * base_precision)


def channel_precisions_np(
    *,
    attended: int,
    n_channels: int,
    base_precision: float,
    att_precision: float,
    mode: str = "none",
    bg_floor: float = 0.05,
    total_budget: float | None = None,
) -> np.ndarray:
    """
    Return per-channel precision probabilities as a numpy vector.

    Parameters
    ----------
    attended:
        Foregrounded channel index (0..n_channels-1), or <0 for no explicit foreground.
    n_channels:
        Number of need channels (typically 4).
    base_precision:
        Baseline precision for unattended channels in unbudgeted mode.
    att_precision:
        Attended precision target in unbudgeted/soft mode.
    mode:
        - "none": unbudgeted (legacy behavior)
        - "soft": conserve a total precision budget with nonzero background
        - "hard": foreground-first allocation with background floor
    bg_floor:
        Minimum background precision in budgeted modes.
    total_budget:
        Optional explicit total budget. If omitted, uses the legacy total:
        att_precision + (n_channels-1)*base_precision.
    """
    n = max(1, int(n_channels))
    m = normalize_budget_mode(mode)
    base = _clip01(base_precision)
    att = _clip01(att_precision)
    floor = _clip01(bg_floor)
    total = _total_budget_default(n_channels=n, base_precision=base, att_precision=att)
    if total_budget is not None and float(total_budget) > 0.0:
        total = float(total_budget)
    total = float(max(0.0, min(float(n), total)))

    has_focus = 0 <= int(attended) < n
    p = np.full((n,), base, dtype=np.float32)

    if m == "none":
        if has_focus:
            p[int(attended)] = np.float32(att)
        return p

    if not has_focus:
        val = float(max(0.0, min(1.0, total / float(n))))
        p[:] = np.float32(val)
        return p

    if n == 1:
        p[0] = np.float32(max(0.0, min(1.0, total)))
        return p

    max_bg = float(max(0.0, min(1.0, total / float(n - 1))))
    floor = min(floor, max_bg)

    if m == "hard":
        bg = floor
        fg = float(max(0.0, min(1.0, total - float(n - 1) * bg)))
        p[:] = np.float32(bg)
        p[int(attended)] = np.float32(fg)
        return p

    # soft: keep a nonzero background while allowing foreground boost.
    fg_max = float(max(0.0, min(1.0, total - float(n - 1) * floor)))
    fg = min(att, fg_max)
    rem = max(0.0, total - fg)
    bg = rem / float(n - 1)
    if bg < floor:
        bg = floor
        fg = float(max(0.0, min(1.0, total - float(n - 1) * bg)))
    p[:] = np.float32(bg)
    p[int(attended)] = np.float32(fg)
    return p


def channel_precisions_jax(
    *,
    attended: jnp.ndarray,
    n_channels: int,
    base_precision: jnp.ndarray,
    att_precision: jnp.ndarray,
    mode_index: int,
    bg_floor: jnp.ndarray,
    total_budget: jnp.ndarray,
) -> jnp.ndarray:
    """
    JAX version of `channel_precisions_np` for traced rollouts.
    """
    n = int(max(1, int(n_channels)))
    base = jnp.clip(jnp.asarray(base_precision, dtype=jnp.float32), 0.0, 1.0)
    att = jnp.clip(jnp.asarray(att_precision, dtype=jnp.float32), 0.0, 1.0)
    floor = jnp.clip(jnp.asarray(bg_floor, dtype=jnp.float32), 0.0, 1.0)
    total = jnp.clip(jnp.asarray(total_budget, dtype=jnp.float32), 0.0, float(n))
    focus = attended.astype(jnp.int32)
    has_focus = (focus >= 0) & (focus < n)

    p_none = jnp.full((n,), base, dtype=jnp.float32)
    p_none = jnp.where(
        has_focus,
        p_none.at[jnp.clip(focus, 0, n - 1)].set(att),
        p_none,
    )

    p_uniform = jnp.full((n,), jnp.clip(total / float(n), 0.0, 1.0), dtype=jnp.float32)
    if n == 1:
        p_budget = jnp.full((1,), jnp.clip(total, 0.0, 1.0), dtype=jnp.float32)
        p_soft = p_budget
        p_hard = p_budget
    else:
        max_bg = jnp.clip(total / float(n - 1), 0.0, 1.0)
        floor = jnp.minimum(floor, max_bg)

        bg_hard = floor
        fg_hard = jnp.clip(total - float(n - 1) * bg_hard, 0.0, 1.0)
        p_hard = jnp.full((n,), bg_hard, dtype=jnp.float32)
        p_hard = p_hard.at[jnp.clip(focus, 0, n - 1)].set(fg_hard)

        fg_max = jnp.clip(total - float(n - 1) * floor, 0.0, 1.0)
        fg_soft = jnp.minimum(att, fg_max)
        rem = jnp.maximum(0.0, total - fg_soft)
        bg_soft = rem / float(n - 1)
        bg_soft = jnp.where(bg_soft < floor, floor, bg_soft)
        fg_soft = jnp.where(
            bg_soft <= floor + 1e-12,
            jnp.clip(total - float(n - 1) * bg_soft, 0.0, 1.0),
            fg_soft,
        )
        p_soft = jnp.full((n,), bg_soft, dtype=jnp.float32)
        p_soft = p_soft.at[jnp.clip(focus, 0, n - 1)].set(fg_soft)

    p_soft = jnp.where(has_focus, p_soft, p_uniform)
    p_hard = jnp.where(has_focus, p_hard, p_uniform)

    if int(mode_index) == 0:
        return p_none
    if int(mode_index) == 1:
        return p_soft
    return p_hard
